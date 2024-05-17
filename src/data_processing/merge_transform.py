import pandas as pd
from ..common_functions.data_common_functions import read_sql_table, create_id, incremental_insert

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