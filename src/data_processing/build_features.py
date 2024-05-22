from ..common_functions.feature_engineering_functions import LowPassFilter, PrincipalComponentAnalysis, NumericalAbstraction
from ..common_functions.data_common_functions import read_sql_table
import matplotlib.pyplot as plt
import numpy as np

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




