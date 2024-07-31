import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from connectors.data_collector import DataCollector, mt5

import pandas as pd 
import numpy as np




def create_lag_features(data:pd.DataFrame, start:int=1, end:int|None=None, column:str="Close"):
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

    df = data.copy()

    if end is None:
        n = len(data)
        end = int((n / np.sqrt(n)) / 2)

    for lag in range(start, end ):
        df[f'lag_{lag}'] = df[column].shift(lag)
        
    return df.dropna()


if __name__ == "__main__":
    
    data_col = DataCollector()
    data = data_col.get_historical_data(symbol="EURUSD", timeframe= mt5.TIMEFRAME_M5 )


    lag_data = create_lag_features(data) 
    print(lag_data.head())