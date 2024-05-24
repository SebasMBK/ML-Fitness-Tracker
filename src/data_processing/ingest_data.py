from ..common_functions.data_common_functions import get_files_directory, get_all_files_in_directory, read_data_into_dataframe, extract_features_from_filename_column, get_datetime_from_epoch, full_load

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
    full_load(
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
    full_load(
        df=df_gyr_final,
        table_schema="stg",
        table_name="fitness_tracker_gyroscope",
        username="postgres",
        password="postgres",
        hostname="localhost",
        port=5432,
        database="ml-fitness-tracker"
    )
    


