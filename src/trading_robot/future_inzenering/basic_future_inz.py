import pandas as pd 
import numpy as np

from trading_robot.data_collection.data_collector import DataCollector, mt5
from trading_robot.utils.logger import log_message


class HoltWinters:
    """
    Holt-Winters model with Brutlag method for anomaly detection.
    """

    def _initial_trend(self) -> float:
        """Initializes the trend based on the first few seasons."""
        sum_trend = 0.0
        for i in range(self._slen):
            sum_trend += float(self._series.iloc[i + self._slen] - self._series.iloc[i]) / self._slen
        return sum_trend / self._slen  

    def _initial_seasonal_components(self) -> dict:
        """Initializes seasonal components."""
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self._series) / self._slen)

        # Calculate seasonal averages
        for j in range(n_seasons):
            season_avg = np.mean(self._series.iloc[self._slen * j:self._slen * j + self._slen])
            season_averages.append(season_avg)

        # Calculate initial seasonal values
        for i in range(self._slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += self._series.iloc[self._slen * j + i] - season_averages[j]
            seasonals[i] = sum_of_vals_over_avg / n_seasons

        return seasonals

    def triple_exponential_smoothing(self, data:pd.DataFrame, column:str= "FLClose", slen:int=12, 
                                     alpha:float=.1, beta:float=.1, gamma:float=0.9, n_preds:int=1, 
                                     scaling_factor:float = 1.96) -> pd.DataFrame:
        """
        Performs triple exponential smoothing and returns a DataFrame with the results,
        predicted values, upper and lower bounds, and model components.

        Triple exponential smoothing includes:

        1. **Level**: the smoothed value of the time series at the current time.

        2. **Trend**: the change in level relative to the previous value.

        3. **Seasonality**: cyclical fluctuations in the data that repeat at a certain periodicity.

        Parameters:
        - data (pd.DataFrame): The time series data containing the column with the time series to analyze.
        - column (str): The name of the column in `data` that contains the time series. Defaults to "Close".
        - slen (int): Season length (number of periods in one season). Determines the periodicity of seasonal fluctuations. Defaults to 24.
        - alpha (float): The smoothing coefficient for the level. The value must be be in the range (0, 1]. Defines the weight for the time series level. Default is 0.5.
        - beta (float): Trend smoothing coefficient. The value must be in the range (0, 1]. Defines the weight for trend. Default is 0.5.
        - gamma (float): Seasonality smoothing coefficient. The value must be in the range (0, 1]. Defines the weight for seasonal variations. Default is 0.5.
        - n_preds (int): Number of steps ahead to forecast. Defines the forecast horizon. Default is 1.
        - scaling_factor (float): Brute-Lag confidence interval width. Defines how wide the confidence interval boundary will be. Default is 1.96 (for a 95% confidence interval).

        Returned value:
        - **Actual** (pd.Series): The actual values ​​of the time series.
        - **Forecast** (pd.Series): The predicted values ​​of the time series as calculated by the model. These are the values ​​the model expects based on the current data and previous estimates.
        - **Predicted_Deviation** (pd.Series): The deviation forecast values ​​calculated by the Brutlag algorithm. This value shows the uncertainty of the forecast.
        - **Upper_Bound** (pd.Series): The upper bound of the forecast confidence interval. It is calculated as the forecast value plus the deviation value multiplied by the scaling factor.
        - **Lower_Bound** (pd.Series): The lower bound of the forecast confidence interval. It is calculated as the forecast value minus the deviation value multiplied by the scaling factor.
        - **Smooth** (pd.Series): Smoothed values ​​of the time series level. These values ​​take into account only the level without trend and seasonality.
        - **Trend** (pd.Series): Estimated trend at each step. It is the change in level over time, taking into account only trends.
        - **Season** (pd.Series): Seasonal components of the model at each step. They show cyclical fluctuations based on pre-calculated seasonal values.

        Notes:
        - The method produces forecasts for both historical data (within the original time series) and future values ​​(within the forecast horizon, `n_preds`).
        - The bias and confidence interval bounds are increased with each forecast step, accounting for forecast uncertainty.
        - The seasonal components and trend are calculated using historical data and the initial values ​​set in the `_initial_trend` and `_initial_seasonal_components` methods.

        The returned DataFrame contains all the listed columns with data corresponding to the length of the time series plus the forecast horizon.

        Return value:
        - **pd.DataFrame**: DataFrame containing columns with actual and forecast values, biases, interval bounds, and model components.

        Example usage: 
        ```python model = HoltWinters() 
        result_df = model.triple_exponential_smoothing() print(result_df.head(15)) ``` 
        """



        self._series = data[column]
        self._column = column
        self._slen = slen
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._n_preds = n_preds
        self._scaling_factor = scaling_factor
        
        result = []
        smooth = []
        trend = []
        season = []
        predicted_deviation = []
        upper_bound = []
        lower_bound = []

        seasonals = self._initial_seasonal_components()
        last_smooth = self._series.iloc[0]
        current_trend = self._initial_trend()

        for i in range(len(self._series) + self._n_preds):
            if i == 0:
                smooth_val = self._series.iloc[0]
                result.append(self._series.iloc[0])
                smooth.append(smooth_val)
                trend.append(current_trend)
                season.append(seasonals[i % self._slen])

                predicted_deviation.append(0)

                upper_bound.append(result[0] + self._scaling_factor * predicted_deviation[0])
                lower_bound.append(result[0] - self._scaling_factor * predicted_deviation[0])

                continue

            if i >= len(self._series):
                m = i - len(self._series) + 1
                forecast = (smooth_val + m * current_trend) + seasonals[i % self._slen]
                result.append(forecast)

                # Увеличиваем неопределенность во время прогноза
                predicted_deviation.append(predicted_deviation[-1] * 1.01)
            else:
                value = self._series.iloc[i]
                last_smooth, smooth_val = smooth_val, self._alpha * (value - seasonals[i % self._slen]) + (1 - self._alpha) * (smooth_val + current_trend)
                current_trend = self._beta * (smooth_val - last_smooth) + (1 - self._beta) * current_trend
                seasonals[i % self._slen] = self._gamma * (value - smooth_val) + (1 - self._gamma) * seasonals[i % self._slen]
                forecast = smooth_val + current_trend + seasonals[i % self._slen]
                result.append(forecast)

                # Отклонение рассчитывается по алгоритму Брутлага
                predicted_deviation.append(self._gamma * np.abs(self._series.iloc[i] - result[i]) + (1 - self._gamma) * predicted_deviation[-1])

            upper_bound.append(result[-1] + self._scaling_factor * predicted_deviation[-1])
            lower_bound.append(result[-1] - self._scaling_factor * predicted_deviation[-1])

            smooth.append(smooth_val)
            trend.append(current_trend)
            season.append(seasonals[i % self._slen])

        df = pd.DataFrame({
            'Forecast': result[:len(self._series)],
            'Predicted_Deviation': predicted_deviation[:len(self._series)],
            'Upper_Bound': upper_bound[:len(self._series)],
            'Lower_Bound': lower_bound[:len(self._series)],
            'Smooth': smooth[:len(self._series)],
            'Trend': trend[:len(self._series)],
            'Season': season[:len(self._series)]
        }, index=self._series.index)

        # Merge original data with results
        return pd.merge(data, df, how='inner', left_index=True, right_index=True)

class BasicFuture(HoltWinters):

    def __init__(self) -> None:
        super().__init__()

    def create_lag_features(self, data: pd.DataFrame, start: int = 1, end: int | None = None, 
                            column: str = "Close") -> pd.DataFrame:
        """
        Creates lagged features for the specified column of the time series.

        Parameters:
        data (pd.DataFrame): The original DataFrame containing the time series.
        start (int): The starting lag (default 1).
        end (int | None): The ending lag. If None, calculated as half the square root of the series length.
        column (str): The name of the column for which to create lagged features (default "Close").

        Returns:
        pd.DataFrame: The DataFrame with the lagged features added.
        """
        log_message(f"Creating lag features for column '{column}' with lags from {start} to {end if end else 'auto'}.")

        df = data.copy()

        if end is None:
            n = len(data)
            end = int((n / np.sqrt(n)) / 2)

        for lag in range(start, end + 1):
            df[f'lag_{lag}'] = df[column].shift(lag)

        df = df.rename(columns={"lag_1": f"FL{column}"})

        log_message(f"Lag features created with {end - start + 1} lags.")

        return df.dropna()

    def exponential_smoothing(self, data: pd.DataFrame, alpha: float = .9, column: str = "FLClose") -> pd.DataFrame:
        """
        Performs exponential smoothing on a time series.

        Parameters:
        data (pd.DataFrame): The original DataFrame containing the time series.
        alpha (float): The smoothing factor (0 < alpha <= 1).
        column (str): The name of the column to smooth (defaults to "Close").

        Returns:
        pd.DataFrame: A DataFrame with a column of smoothed values added.
        """
        log_message(f"Applying exponential smoothing with alpha={alpha} on column '{column}'.")

        data = data.copy()
        data["ES"] = data[column].ewm(alpha=alpha, adjust=False).mean()

        log_message("Exponential smoothing applied successfully.")

        return data

    def double_exponential_smoothing(self, data: pd.DataFrame, alpha: float = .9, beta: float = .1, 
                                    column: str = "FLClose") -> pd.DataFrame:
        """
        Performs double exponential smoothing on a time series.

        Parameters:
        data (pd.DataFrame): The original DataFrame containing the time series.
        alpha (float): Smoothing factor for the level (0 < alpha <= 1).
        beta (float): Smoothing factor for the trend (0 < beta <= 1).
        column (str): The name of the column to smooth (defaults to "Close").

        Returns:
        pd.DataFrame: DataFrame with the smoothed column and forecast appended.
        """
        log_message(f"Applying double exponential smoothing with alpha={alpha} and beta={beta} on column '{column}'.")

        data = data.copy()
        series = data[column].values
        n = len(series)
        result = [series[0]]

        # Initialization of level and trend
        level = series[0]
        trend = series[1] - series[0]

        for t in range(1, n):
            value = series[t]
            last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            result.append(level + trend)

        # Processing the last forecast
        result.append(level + trend)

        df = data.copy()
        df["DES"] = result[:n]

        log_message("Double exponential smoothing applied successfully.")

        return df

    def confidence_interval(self, data: pd.DataFrame, n: int = 24, column: str = "FLClose", dropna: bool = True) -> pd.DataFrame:
        """
        Calculates the confidence interval for the moving average of the specified column.

        Parameters:
        data (pd.DataFrame): The original DataFrame containing the time series.
        n (int): The window size for the moving standard deviation (default is 24).
        column (str): The name of the column for which to calculate the confidence interval (default is "Close").
        dropna (bool): If True, removes rows with NaN values (default is True).

        Returns:
        pd.DataFrame: DataFrame with columns added for the upper and lower bounds of the confidence interval.
        """
        log_message(f"Calculating confidence interval with window size {n} for column '{column}'.")

        df = data.copy()

        if len(df) < n:
            log_message(f"Data length ({len(df)}) is less than window size ({n}).")
            raise ValueError(f"Data length ({len(df)}) is less than window size ({n}).")

        rolling_std = df[column].rolling(window=n).std()

        df['Conf_upper_interval'] = df[column] + 1.96 * rolling_std
        df['Conf_lower_interval'] = df[column] - 1.96 * rolling_std

        if dropna:
            df.dropna(inplace=True)

        log_message("Confidence interval calculated successfully.")

        return df

    def all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a series of indicator methods to the input DataFrame and returns the DataFrame with added features.

        Args:
            data (pd.DataFrame): Input DataFrame with the original data.

        Returns:
            pd.DataFrame: DataFrame with original and new features generated by the indicator methods.
        """
        log_message("Applying all indicators to the data.")

        data = data.copy()

        class_methods = ['confidence_interval', 'double_exponential_smoothing', 
                          'exponential_smoothing', 'triple_exponential_smoothing']

        data = self.create_lag_features(data)
        original_columns = data.columns
        new_columns = []

        for method in class_methods:
            try:
                m = getattr(self, method)
                method_result = m(data)
                new_columns.append(method_result.drop(columns=original_columns))
                log_message(f"Applied method: {method}.")
            except Exception as e:
                log_message(f"Error applying method '{method}': {e}")

        new_data = pd.concat(new_columns, axis=1)
        merged_data = pd.merge(data, new_data, 
                               left_index=True, right_index=True,
                               how='inner')

        log_message("All indicators applied successfully.")

        return merged_data


if __name__ == "__main__":
    bf = BasicFuture()
    data_col = DataCollector()

    data = data_col.get_historical_data(symbol="EURUSD", timeframe= mt5.TIMEFRAME_M5 )  
    data = bf.all_indicators(data)
    print(data)
    
