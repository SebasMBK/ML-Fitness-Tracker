import pandas as pd
from glob import glob
from hashlib import sha256

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
    df_list = [pd.read_csv(file).assign(filename=file.split("\\")[2]) for file in files_list if file_type in file]
    df = pd.concat(df_list, ignore_index=True)
    # Renaming columns
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

def create_id_sha256(row):
    concat_string = ''.join(row.astype(str))
    sha256_hash = sha256(concat_string.encode()).hexdigest()
    return sha256_hash

def extract_features_from_filename_column(df: pd.DataFrame):
    df["participant"] = df["filename"].str.split("-").str[0]
    df["label"] = df["filename"].str.split("-").str[1]
    df["category"] = df["filename"].str.split("-").str[2].str.split("_").str[0].str.rstrip("123")
    df["id"] = df.apply(create_id_sha256, axis=1)
    return df

if __name__ == '__main__':
    directory = '**/fitness_data/'
    wild_card = '*.csv'
    dir_path = f"{directory}{wild_card}"
    files_list = get_all_files_in_directory(dir_path=dir_path)
    df_acc = read_data_into_dataframe(files_list=files_list, file_type="Accelerometer")
    df_gyr = read_data_into_dataframe(files_list=files_list, file_type="Gyroscope")
    df_acc_with_features = extract_features_from_filename_column(df=df_acc)
    df_gyr_with_features = extract_features_from_filename_column(df=df_gyr)