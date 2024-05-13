import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from hashlib import md5

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

def insert_to_merged_table(df: pd.DataFrame, table_schema: str,
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
    insert_to_merged_table(
        df=df_resampled,
        table_schema="merged",
        table_name="fitness_tracker",
        username="postgres",
        password="postgres",
        hostname="localhost",
        port=5432,
        database="ml-fitness-tracker"
    )