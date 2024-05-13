import pandas as pd
import os
from pathlib import Path
from glob import glob
from hashlib import md5
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

def get_files_directory() -> str:
    path = Path(__file__).parent.parent.parent
    data_dir_path = str(path.joinpath("fitness_data", "*.csv"))
    return data_dir_path

def get_all_files_in_directory(dir_path: str) -> list[str]:
    """
    This function will return all the filenames with their corresponding path
    Args:
        dir_path (str): Parent file path where the files are stored

    Returns:
        files: List of strings representing the file paths
    """
    files = glob(dir_path, recursive=True)
    return files

def read_data_into_dataframe(files_list: list, file_type: str) -> pd.DataFrame:
    relevant_files = [file_path for file_path in files_list if file_type in file_path]
    df_list = [pd.read_csv(file).assign(filename=os.path.basename(file), set=i) for i, file in enumerate(relevant_files,1)]
    df = pd.concat(df_list, ignore_index=True)
    # Renaming columns
    if file_type not in ("Accelerometer", "Gyroscope"):
        raise ValueError("Error: Invalid file_type. Correct values are 'Acceleromenter' or 'Gyroscope'.")
    if file_type == "Accelerometer":
        df = df.rename(
            columns={
                "epoch (ms)": "epoch_ms",
                "time (01:00)": "time",
                "elapsed (s)": "elapsed_seconds",
                "x-axis (g)": "x_axis_g",
                "y-axis (g)": "y_axis_g",
                "z-axis (g)": "z_axis_g"
            }
        )
    if file_type == "Gyroscope":
        df = df.rename(
            columns={
                "epoch (ms)": "epoch_ms",
                "time (01:00)": "time",
                "elapsed (s)": "elapsed_seconds",
                "x-axis (deg/s)": "x_axis_deg_s",
                "y-axis (deg/s)": "y_axis_deg_s",
                "z-axis (deg/s)": "z_axis_deg_s"
            }
        )
    return df

def create_id(row) -> pd.Series:
    concat_string = ''.join(row.astype(str))
    sha256_hash = md5(concat_string.encode()).hexdigest()
    return sha256_hash

def extract_features_from_filename_column(df: pd.DataFrame) -> pd.DataFrame:
    df["participant"] = df["filename"].str.split("-").str[0]
    df["label"] = df["filename"].str.split("-").str[1]
    df["category"] = df["filename"].str.split("-").str[2].str.split("_").str[0].str.rstrip("123")
    df["id"] = df.apply(create_id, axis=1)
    return df

def get_datetime_from_epoch(df: pd.DataFrame) -> pd.DataFrame:
    df.index = pd.to_datetime(df["epoch_ms"], unit="ms")
    del df["epoch_ms"]
    del df["time"]
    del df["elapsed_seconds"]
    return df

def insert_to_stg(df: pd.DataFrame, table_schema: str,
                  table_name: str, username: str,
                  password: str, hostname: str,
                  port: int, database: str) -> None:
    conn_string = f"postgresql+psycopg2://{username}:{password}@{hostname}:{port}/{database}"
    engine = create_engine(conn_string)
    try:
        # Truncate table before inserting
        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {table_schema}.{table_name}"))
        df.to_sql(
            name=table_name,
            schema=table_schema,
            con=engine,
            if_exists='append',
            index=True
        )
    except SQLAlchemyError as e:
        raise SQLAlchemyError(f"An error ocurred while inserting the stg data: {e}")
    finally:
        engine.dispose()

if __name__ == '__main__':
    data_dir_path = get_files_directory() # I will need to change this when working in S3
    files_list = get_all_files_in_directory(dir_path=data_dir_path) # And this
    df_acc = read_data_into_dataframe(files_list=files_list, file_type="Accelerometer")
    df_gyr = read_data_into_dataframe(files_list=files_list, file_type="Gyroscope")
    df_acc_with_features = extract_features_from_filename_column(df=df_acc)
    df_gyr_with_features = extract_features_from_filename_column(df=df_gyr)
    df_acc_final = get_datetime_from_epoch(df=df_acc_with_features)
    df_gyr_final = get_datetime_from_epoch(df=df_gyr_with_features)
    # Insert accelerometer to stg table
    insert_to_stg(
        df=df_acc_final,
        table_schema="stg",
        table_name="fitness_tracker_accelerometer",
        username="postgres",
        password="postgres",
        hostname="localhost",
        port=5432,
        database="ml-fitness-tracker"
    )
    # Insert gyroscope to stg table
    insert_to_stg(
        df=df_gyr_final,
        table_schema="stg",
        table_name="fitness_tracker_gyroscope",
        username="postgres",
        password="postgres",
        hostname="localhost",
        port=5432,
        database="ml-fitness-tracker"
    )
    


