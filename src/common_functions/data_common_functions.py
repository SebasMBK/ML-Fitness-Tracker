import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from hashlib import md5
from sqlalchemy import create_engine

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

def create_id(row) -> pd.Series:
    concat_string = ''.join(row.astype(str))
    sha256_hash = md5(concat_string.encode()).hexdigest()
    return sha256_hash