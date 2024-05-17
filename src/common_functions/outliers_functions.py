import pandas as pd
import matplotlib.pyplot as plt
import scipy
import math
from sklearn.neighbors import LocalOutlierFactor

def mark_outliers_iqr(dataset: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        columns (list): The columns you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    for col in columns:
        Q1 = dataset[col].quantile(0.25)
        Q3 = dataset[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
            dataset[col] > upper_bound
        )

    return dataset

def mark_outliers_chauvenet(dataset: pd.DataFrame, columns: list[str], C: int=2) -> pd.DataFrame:
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    
    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        columns (list): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption 
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    for col in columns:
        # Compute the mean and standard deviation.
        mean = dataset[col].mean()
        std = dataset[col].std()
        N = len(dataset.index)
        criterion = 1.0 / (C * N)

        # Consider the deviation for the data points.
        deviation = abs(dataset[col] - mean) / std

        # Express the upper and lower bounds.
        low = -deviation / math.sqrt(C)
        high = deviation / math.sqrt(C)
        prob = []
        mask = []

        # Pass all rows in the dataset.
        for i in range(0, len(dataset.index)):
            # Determine the probability of observing the point
            prob.append(
                1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
            )
            # And mark as an outlier when the probability is below our criterion.
            mask.append(prob[i] < criterion)
        dataset[col + "_outlier"] = mask
    return dataset

def mark_outliers_lof(dataset: pd.DataFrame, columns: list[str], n=20) -> pd.DataFrame:
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        columns (list): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.
    
    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """
    
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores

def plot_binary_outliers(dataset: pd.DataFrame, col: str, outlier_col: str, reset_index: bool) -> None:
    """ Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()