import pandas as pd 
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression


from trading_robot.data_collection.data_collector import DataCollector, mt5
from trading_robot.future_inzenering.talib_indicators import TLIndicators
from trading_robot.future_inzenering.basic_future_inz import BasicFuture
from trading_robot.utils.logger import log_message


class BasicFeatureSelector:
    """
    Class for feature selection based on various methods:
    - Removal of features with low variance
    - Feature selection based on Spearman correlation with the target variable
    - Feature selection based on Pearson correlation between them
    - Feature selection based on mutual information with the target variable
    """

    def remove_features_low_variance(self, data: pd.DataFrame, threshold: float = 0.0) -> list:
        """
        Removes columns with variance below a certain threshold.

        Parameters:
        -----------
        data : pd.DataFrame
        Input DataFrame.

        threshold : float, default 0.0
        Variance threshold. Columns with variance below this value will be removed.

        Returns:
        -----------
        - list - a list of feature names that have low correlation with other features.
        """
        log_message("Starting feature removal based on variance.")

        # Calculate variance for each column
        variance = data.var()

        # Select columns with variance above the threshold
        variance_above_threshold = variance[variance > threshold].index.tolist()

        log_message(f"Features with variance above threshold {threshold}: {variance_above_threshold}")

        # Return DataFrame only with the required columns
        return variance_above_threshold

    def filter_features_by_spearman_corr(self, data: pd.DataFrame, target: str = "Close",
                                        p_value_threshold: float = .05,
                                        corr_threshold: float = .1) -> list:
        """
        Function to filter features by Spearman correlation with the target variable.

        Spearman's rank correlation coefficient: Measures the degree of monotonic relationship between 
        two variables. Monotonic relationship means that as one variable increases, the other variable 
        either always increases or always decreases, but not necessarily in a linear fashion.

        p-value: Helps to assess the statistical significance of the correlation. 
        This value indicates the probability that the observed correlation occurred by chance. 
        If the p-value is less than a chosen threshold (e.g. 0.05), then the correlation is 
        considered statistically significant.

        Parameters:
        - data: pd.DataFrame - source data with features and target variable.
        - target: str - name of the target variable in the dataframe (default is "Close").
        - p_value_threshold: float - threshold for the p-value of the correlation (default is 0.05).
        - corr_threshold: float - threshold for the absolute value of the correlation (default is 0.05).

        Returns:
        - list - a list of feature names that have low correlation with other features.
        """

        log_message(f"Filtering features by Spearman correlation with target '{target}'.")

        if target not in data.columns:
            logger.error(f"Target variable '{target}' not found in the data.")
            raise ValueError(f"Target variable '{target}' not found in the data.")

        
        Y = data[target]
        data = data.drop(columns=target)

        correlation_df = {}

        # Calculate the Spearman correlation for each feature
        for column in data.columns:
            if data[column].isnull().any() or Y.isnull().any():
                correlation_df[column] = {"corr": None, "p_value": None}
                continue

            corr, p_value = spearmanr(data[column], Y)
            correlation_df[column] = {"corr": corr, "p_value": p_value}

        correlation_df = pd.DataFrame(correlation_df).T

        # Filter by correlation values ​​and p-values
        filtered_df = correlation_df[(correlation_df['corr'].abs() > corr_threshold) &
        (correlation_df['p_value'] < p_value_threshold)]

        filtered_data = data[filtered_df.index]
        filtered_data = filtered_data.join(Y)

        log_message(f"Selected features by Spearman correlation: {filtered_df.index.tolist() + [target]}")
        
        return filtered_df.index.tolist() + [target]

    def filter_correlated_features(self, data: pd.DataFrame, corr_threshold: float = 0.8) -> list:
        """
        A function for filtering features based on the Pearson correlation between them.
        Returns a list of features that have low correlation with others.

        Parameters:
        - data: pd.DataFrame - source data with features.
        - corr_threshold: float - correlation threshold above which features will be excluded.

        Returns:
        - list - a list of feature names that have low correlation with other features.
        """
        log_message(f"Filtering features by Pearson correlation with threshold {corr_threshold}.")

        # Compute the Pearson correlation matrix
        corr_matrix = data.corr(method='pearson')
        
        # Set the diagonal to zero (self-correlation)
        np.fill_diagonal(corr_matrix.values, 0)
        
        # Identifying highly correlated feature pairs
        high_corr_features = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > corr_threshold:
                    colname = corr_matrix.columns[i]
                    high_corr_features.add(colname)
        
        all_features = set(data.columns)
        low_corr_features = all_features - high_corr_features
        
        log_message(f"Selected features with low correlation: {list(low_corr_features)}")

        return list(low_corr_features)

    def select_features_by_mutual_info(self, data: pd.DataFrame, target: str = "Close", mi_threshold: float = 0.1) -> list:
        """
        A function for selecting lag features based on their mutual information with the target variable.

        Parameters:
        - data: pd.DataFrame - source data containing lag features and the target variable.
        - target: str - name of the target variable in the dataframe.
        - mi_threshold: float - mutual information threshold for lag selection.

        Returns:
        - list - a list of feature names that have low correlation with other features.
        """
        log_message(f"Selecting features by mutual information with target '{target}' and threshold {mi_threshold}.")

        X = data.drop(columns=[target])
        target = data[target]
        
        mi = mutual_info_regression(X, target)
        
        # Create a dictionary with feature names and their mutual information
        mi_series = pd.Series(mi, index=X.columns)
        
        # Filter features by mutual information threshold
        selected_features = mi_series[mi_series >= mi_threshold].index.tolist()
        
        log_message(f"Selected features by mutual information: {selected_features}")
        
        return selected_features


if __name__ == "__main__":

    # Getting Data
    data_col = DataCollector()
    data = data_col.get_historical_data(symbol="EURUSD", timeframe= mt5.TIMEFRAME_M5 )

    # Added Indicators
    bf = BasicFuture()
    data = bf.create_lag_features(data, column="Open", end=1)
    data = bf.create_lag_features(data, column="High", end=1)
    data = bf.create_lag_features(data, column="Low", end=1)
    data = bf.create_lag_features(data, column="Volume", end=1)
    data = bf.create_lag_features(data, column="Close")


    data = bf.double_exponential_smoothing(data)
    data = bf.exponential_smoothing(data)
    data = bf.triple_exponential_smoothing(data)
    data = bf.confidence_interval(data, column="FLClose")

    tl = TLIndicators(data=data, Close="FLClose", High="FLHigh" , Low="FLLow", 
                    Open="FLOpen", Volume="FLVolume")

    data = tl.all_talib_indicators()

    # Select indicators 
    bfs = BasicFeatureSelector()
    target = "Close"

    feture = bfs.remove_features_low_variance(data)
    feture = bfs.filter_features_by_spearman_corr(data[feture], target=target)
    feture = bfs.select_features_by_mutual_info(data[feture], target=target)

    data = data[feture]
    columns = [col for col in data.columns if "lag" not in col and "FL" not in col and "Close" not in col]
    lag_col = [col for col in data.columns if "lag" in col.lower() or "FL" in col]

    feture = bfs.filter_correlated_features(data[columns])
    feture = feture + lag_col
    print(feture)