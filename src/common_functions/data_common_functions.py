import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from hashlib import md5
from sqlalchemy import create_engine
from sqlalchemy import create_engine, text
from pathlib import Path
from glob import glob
import os

#########################################################################
#########################################################################
############################# Utilites ##################################


def create_id(row) -> pd.Series:
    concat_string = ''.join(row.astype(str))
    sha256_hash = md5(concat_string.encode()).hexdigest()
    return sha256_hash

def resample_frequency(df: pd.DataFrame) -> pd.DataFrame:
    # Sampling rule
    sampling_rule = {
        'x_axis_g': 'mean',
        'y_axis_g': 'mean',
        'z_axis_g': 'mean',
        'category': 'last',
        'label': 'last',
        'participant': 'last',
        'x_axis_deg_s': 'mean',
        'y_axis_deg_s': 'mean',
        'z_axis_deg_s': 'mean',
        'set': 'last'
    }
    # We have to use a frequency that gives us a good amount of data,
    # but not too much that becomes too expensive to compute
    time_rule = '200ms'

    # The dataframe will become to large if we resample everything in one go.
    # So we are going to group by day (this is possible because the index is a date),
    # and perfom the sampling in each of those groups and then concat everything
    days = [group for i, group in df.groupby(pd.Grouper(freq='D'))] # This is a collection of dataframes
    df_resampled = pd.concat([df.resample(rule=time_rule).apply(sampling_rule).dropna() for df in days])
    df_resampled["id"] = df_resampled.apply(create_id, axis=1).astype("string")
    return df_resampled

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

#########################################################################
#########################################################################
#########################################################################



#########################################################################
#########################################################################
############################# Data movement #############################

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

def read_sql_table(table_schema: str,
                   table_name: str,
                   username: str,
                   password: str,
                   hostname: str,
                   port: int,
                   database: str ) -> pd.DataFrame:
    conn_string = f"postgresql+psycopg2://{username}:{password}@{hostname}:{port}/{database}"
    engine = create_engine(conn_string)
    try:
        # Read table
        df = pd.read_sql_table(
            table_name=table_name,
            schema=table_schema,
            con=engine
        )
        df.index = df["epoch_ms"].astype('datetime64[ns]')
        # We want epoch_ms only in the index
        del df["epoch_ms"]
        engine.dispose()
    except SQLAlchemyError as e:
        raise SQLAlchemyError(f"An error ocurred while reading the data: {e}")
    finally:
        engine.dispose()
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

def incremental_insert(df: pd.DataFrame, table_schema: str,
                  table_name: str, username: str,
                  password: str, hostname: str,
                  port: int, database: str) -> None:
    conn_string = f"postgresql+psycopg2://{username}:{password}@{hostname}:{port}/{database}"
    engine = create_engine(conn_string)
    try:
        # Read the IDs in the main table and convert the results into a list
        existing_ids = pd.read_sql_query(
            f"SELECT id FROM {table_schema}.{table_name}",
            con=engine
            )['id'].tolist()
        df_only_new_records = df[~df['id'].isin(existing_ids)]
        # Insert if dataframe is not empty
        if not df_only_new_records.empty:
            df_only_new_records.to_sql(
                name=table_name,
                schema=table_schema,
                con=engine,
                if_exists='append',
                index=True
            )
    except SQLAlchemyError as e:
        raise SQLAlchemyError(f"An error ocurred while inserting the merged data: {e}")
    finally:
        engine.dispose()

#################################################################################
#################################################################################
#################################################################################