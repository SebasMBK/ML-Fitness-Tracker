from ..common_functions.feature_engineering_functions import FourierTransformation, LowPassFilter, PrincipalComponentAnalysis, NumericalAbstraction
from ..common_functions.data_common_functions import read_sql_table, full_load
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

if __name__ == '__main__':
    # Load the data
    df = read_sql_table(
            table_schema="outliers",
            table_name="fitness_tracker_chauvenet",
            username="postgres",
            password="postgres",
            hostname="localhost",
            port=5432,
            database="ml-fitness-tracker"
    )

    predictor_columns = list(df.columns[4:10])

    # Imputate NaN values after outlier detection
    for col in predictor_columns:
        df[col] = df[col].interpolate()
    
    # Calculate duration of the set for noise reduction
    for set in df["set"].unique():
        set_start = df[df["set"] == set].index[0]
        set_end = df[df["set"] == set].index[-1]
        set_duration = set_end - set_start
        # Adding a column called duration
        df.loc[(df["set"] == set), "duration"] = set_duration.seconds

    # Mean of the duration of the set by category
    df_duration_by_cat = df.groupby(["category"])["duration"].mean()


    # Reducing the noise of each repetition. During training
    # the adjustments of the hands, bars, etc. can come up in
    # the data, so that needs to be filtered.
    # Using Butterworth lowpass filter
    df_lowpass = df.copy()
    lowpass = LowPassFilter()
    # In a previous step (resample frequency), the frequency used was
    # 200ms, so for 1000ms that is 5 entries
    fs = 1000/200
    cutoff = 1.3 # This value is obtained via experimenting using visualization to see the results
    for col in predictor_columns:            
        df_lowpass = lowpass.low_pass_filter(
            data_table=df_lowpass,
            col=col,
            sampling_frequency=fs,
            cutoff_frequency=cutoff
        )
        # Overwriting the original columns with the lowpass ones
        df_lowpass[col] = df_lowpass[f"{col}_lowpass"]
        del df_lowpass[f"{col}_lowpass"]
    
    # PCA to reduce the complexity of the data
    df_pca = df_lowpass.copy()
    pca = PrincipalComponentAnalysis()
    # This is used to visualize the variance when selecting the
    # number of variables/features for the PCA process. The method
    # used was the elbow method.
    pc_values = pca.determine_pc_explained_variance(
        data_table=df_pca,
        cols=predictor_columns
    )
    df_pca = pca.apply_pca(
        data_table=df_pca,
        cols=predictor_columns,
        number_comp=3 # This was chosen using the elbow method using pc_values
    )

    # To help the model generalize better, the three values (x, y and z) of the accelerometer
    # and gyroscope will be converted into a single scalar value per each device
    # to make it impartial to any device orientation and can handle dynamic re-orientations.
    # The technique used will be sum of squares.
    df_squared = df_pca.copy()
    acc_r = df_squared["x_axis_g"] ** 2 + df_squared["y_axis_g"] ** 2 + df_squared["z_axis_g"] ** 2
    gyr_r = df_squared["x_axis_deg_s"] ** 2 + df_squared["y_axis_deg_s"] ** 2 + df_squared["z_axis_deg_s"] ** 2
    df_squared["acc_r"] = np.sqrt(acc_r)
    df_squared["gyr_r"] = np.sqrt(gyr_r)

    # Calculating the rolling average for the dataset to obtain more data from the dataset
    # similar to window functions
    df_rolling = df_squared.copy()
    numabs = NumericalAbstraction()
    window_size = int(1000/200)
    predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

    # A subset of the data is needed to not mix different sets data
    df_rolling_list = []
    for set in df_rolling["set"].unique():
        subset = df_rolling[df_rolling["set"] == set].copy()
        for col in predictor_columns:
            subset = numabs.abstract_numerical(subset, [col], window_size, "mean")
            subset = numabs.abstract_numerical(subset, [col], window_size, "std")
        df_rolling_list.append(subset)
    df_rolling = pd.concat(df_rolling_list)

    # Frequency abstraction. Usefull to obtain insights and components from
    # frequency data
    df_frequency = df_rolling.copy().reset_index()
    freqabs = FourierTransformation()
    frequency = int(1000/200)
    window_size = int(2800/200)
    df_frequency_list = []
    for s in df_frequency["set"].unique():
        subset = df_frequency[df_frequency["set"] == s].reset_index(drop=True).copy()
        subset = freqabs.abstract_frequency(subset, predictor_columns, window_size, frequency)
        df_frequency_list.append(subset)
    df_frequency = pd.concat(df_frequency_list).set_index("epoch_ms", drop=True)

    # Dealing with overlapping windows to avoid overfiting
    df_frequency = df_frequency.dropna()
    # To avoid overlaping, 50% of the data will be dropped by skipping every other row
    df_frequency = df_frequency.iloc[::2]

    # Clustering
    df_cluster = df_frequency.copy()
    cluster_columns = ["x_axis_g", "y_axis_g", "z_axis_g"]
    k = 5 # This is obtained using the inertias with the elbow method
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    subset = df_cluster[cluster_columns]
    df_cluster["cluster"] = kmeans.fit_predict(subset)

    # Insert into table
    full_load(
        df=df_cluster,
        table_schema="clean",
        table_name="fitness_tracker",
        username="postgres",
        password="postgres",
        hostname="localhost",
        port=5432,
        database="ml-fitness-tracker"
    )





