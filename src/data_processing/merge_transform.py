import pandas as pd
from ..common_functions.data_common_functions import read_sql_table, incremental_insert, resample_frequency

if __name__ == '__main__':
    # Read accelerometer data
    df_acc = read_sql_table(
        table_schema="stg",
        table_name="fitness_tracker_accelerometer",
        username="postgres",
        password="postgres",
        hostname="localhost",
        port=5432,
        database="ml-fitness-tracker"
    )
    # Read gyroscope data
    df_gyr = read_sql_table(
        table_schema="stg",
        table_name="fitness_tracker_gyroscope",
        username="postgres",
        password="postgres",
        hostname="localhost",
        port=5432,
        database="ml-fitness-tracker"
    )
    df_merged = pd.concat([df_acc.iloc[:,4:7], df_gyr.iloc[:,1:8]], axis=1)
    df_resampled = resample_frequency(df=df_merged)
    df_resampled["set"] = df_resampled['set'].astype("int")
    # Insert data into table
    incremental_insert(
        df=df_resampled,
        table_schema="merged",
        table_name="fitness_tracker",
        username="postgres",
        password="postgres",
        hostname="localhost",
        port=5432,
        database="ml-fitness-tracker"
    )