import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ..common_functions.outliers_functions import mark_outliers_lof, mark_outliers_chauvenet, mark_outliers_iqr, plot_binary_outliers
from ..common_functions.data_common_functions import read_sql_table

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
    
    # Outliers columns - First 3 are acc data and the other 3 are gyro data
    outlier_columns = [
        'x_axis_g',
        'y_axis_g',
        'z_axis_g',
        'x_axis_deg_s',
        'y_axis_deg_s',
        'z_axis_deg_s'
    ]

    # Plotting the outliers
    plt.style.use("fivethirtyeight")
    plt.rcParams["figure.figsize"] = (20,5)
    plt.rcParams["figure.dpi"] = 100
    # Accelerometer data
    df[outlier_columns[:3] + ["label"]].boxplot(by="label", figsize=(20,10), layout=(1,3))
    # Gyroscope data
    df[outlier_columns[3:] + ["label"]].boxplot(by="label", figsize=(20,10), layout=(1,3))

    # Mark outliers using IQR
    df_marked_outliers_iqr = mark_outliers_iqr(df, columns=outlier_columns)
    # Show outliers vs non-outliers in a more contrasting way
    # and it seems like there is some data that should be investigated
    # if they truly are outliers or not
    for col in outlier_columns:
        plot_binary_outliers(
            dataset=df_marked_outliers_iqr,
            col=col,
            outlier_col=f"{col}_outlier",
            reset_index=True
        )
    
    # Chauvenetes method to detect outliers
    # This method assumes that the data has a normal distribution
    # so first there is the need to check if this is the case for
    # this data
    # Accelerometer data
    df[outlier_columns[:3] + ["label"]].plot.hist(by="label", figsize=(20,20), layout=(3,3))
    # Gyroscope data
    df[outlier_columns[3:] + ["label"]].plot.hist(by="label", figsize=(20,20), layout=(3,3))
    # It is not perfectly normalized, but for this project is enough
    # Now the Chauvenete method will be applied
    # It will be noticeable how there are many outliers and this is for the rest data
    # that is not normally distributed
    df_marked_outliers_chauv = mark_outliers_chauvenet(df, columns=outlier_columns)
    for col in outlier_columns:
        plot_binary_outliers(
            dataset=df_marked_outliers_chauv,
            col=col,
            outlier_col=f"{col}_outlier",
            reset_index=True
        )
    
    # Local outlier factor method to detect outliers
    # This method uses the distance to detect outliers and
    # it is a unsupervised learning method. A model will be trained
    # and it will be used to mark the outliers
    # The results show that some of the values in the bottom/upper and in the middle
    # are also marked as outliers
    df_marked_outliers_lof, outliers, X_scores = mark_outliers_lof(dataset=df, columns=outlier_columns)
    for col in outlier_columns:
        plot_binary_outliers(
            dataset=df_marked_outliers_lof,
            col=col,
            outlier_col=f"outlier_lof",
            reset_index=True
        )
    
    # Until this point, the data has been analyze as whole, now
    # the analysis will be done by label
    # This will help into really see how well the outlier detection method
    # performs for our data
    label = "bench"
    dataset_iqr = mark_outliers_iqr(df[df["label"] == label], columns=outlier_columns)
    dataset_chauvenet = mark_outliers_chauvenet(df[df["label"] == label], columns=outlier_columns)
    dataset_lof, outliers, X_scores = mark_outliers_lof(dataset=df[df["label"] == label], columns=outlier_columns)
    for col in outlier_columns:
        plot_binary_outliers(
            dataset=dataset_iqr,
            col=col,
            outlier_col=f"{col}_outlier",
            reset_index=True
        )
    for col in outlier_columns:
        plot_binary_outliers(
            dataset=dataset_chauvenet,
            col=col,
            outlier_col=f"{col}_outlier",
            reset_index=True
        )
    for col in outlier_columns:
        plot_binary_outliers(
            dataset=dataset_lof,
            col=col,
            outlier_col=f"outlier_lof",
            reset_index=True
        )
    
    # Now, a decision has to be made about what method is going to be used
    # and for now the decision is to use chauvenet because of the previous results
    df_outliers_removed = df.copy()
    for col in outlier_columns:
        for label in df["label"].unique():
            ds = mark_outliers_chauvenet(df[df["label"] == label], columns=[col])
            # Replace outliers values with NaN
            ds.loc[ds[f"{col}_outlier"], col] = np.nan
            # Update the values in the dataframe
            df_outliers_removed.loc[(df_outliers_removed["label"] == label), col] = ds[col]
            
            n_outliers_removed = len(ds) - len(ds[col].dropna())
            print(f"Removed {n_outliers_removed} from {col} for {label}") 
df_outliers_removed.info()