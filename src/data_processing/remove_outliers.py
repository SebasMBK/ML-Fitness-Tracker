import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ..common_functions.outliers_functions import mark_outliers_chauvenet
from ..common_functions.data_common_functions import read_sql_table, incremental_insert, create_id

if __name__ == '__main__':
    df = read_sql_table(
        table_schema="merged",
        table_name="fitness_tracker",
        username="postgres",
        password="postgres",
        hostname="localhost",
        port=5432,
        database="ml-fitness-tracker"
    )
    # Drop ID column
    del df["id"]
    
    # Outliers columns - First 3 are acc data and the other 3 are gyro data
    outlier_columns = [
        'x_axis_g',
        'y_axis_g',
        'z_axis_g',
        'x_axis_deg_s',
        'y_axis_deg_s',
        'z_axis_deg_s'
    ]

    # Removing outliers
    df_outliers_removed = df.copy()
    for col in outlier_columns:
        for label in df["label"].unique():
            ds = mark_outliers_chauvenet(df[df["label"] == label], columns=[col])
            # Replace outliers values with NaN
            ds.loc[ds[f"{col}_outlier"], col] = np.nan
            # Update the values in the dataframe
            df_outliers_removed.loc[(df_outliers_removed["label"] == label), col] = ds[col]

    # Insert ID into dataframe for incremental load
    df_outliers_removed["id"] = df_outliers_removed.apply(create_id, axis=1).astype("string")
    
    # Insert into table
    incremental_insert(
        df=df_outliers_removed,
        table_schema="outliers",
        table_name="fitness_tracker_chauvenet",
        username="postgres",
        password="postgres",
        hostname="localhost",
        port=5432,
        database="ml-fitness-tracker"
    )

