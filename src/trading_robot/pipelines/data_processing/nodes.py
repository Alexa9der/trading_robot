import pandas as pd

# Import classes from the appropriate modules
from trading_robot.utils.logger import log_message
from trading_robot.data_collection.data_collector import DataCollector
from trading_robot.future_inzenering.basic_future_inz import BasicFuture
from trading_robot.future_inzenering.talib_indicators import TLIndicators
from trading_robot.future_selection.basic_feature_selector import BasicFeatureSelector



def nodes_load_data() -> pd.DataFrame:

    log_message("Starting data loading nodes.")

    loader = DataCollector()
    data = loader.get_historical_data('EURUSD')

    log_message(f"Data loaded nodes successfully. Data keys: {list(data.keys())}")

    return data

def nodes_inzener_features(data: pd.DataFrame) -> pd.DataFrame:

    log_message("Starting feature engineering nodes.")

    bf = BasicFuture()

    for column in ["Open", "High", "Low", "Volume"]:
        data = bf.create_lag_features(data, column=column, end=1)

    data = bf.create_lag_features(data, column="Close")
    data = bf.double_exponential_smoothing(data)
    data = bf.exponential_smoothing(data)
    data = bf.triple_exponential_smoothing(data)
    data = bf.confidence_interval(data, column="FLClose")

    tl = TLIndicators(data=data)
    data = tl.all_talib_indicators()

    log_message("Feature engineering nodes completed.")
                
    return data

def nodes_select_features(data: pd.DataFrame) -> pd.DataFrame:

    log_message("Starting feature selection nodes.")

    bfs = BasicFeatureSelector()
    target = "Close"

    feture = bfs.remove_features_low_variance(data)
    feture = bfs.filter_features_by_spearman_corr(data[feture], target=target)
    feture = bfs.select_features_by_mutual_info(data[feture], target=target)

    selected_data = data[feture]
    columns = [col for col in selected_data.columns if "lag" not in col and "FL" not in col and "Close" not in col]
    lag_col = [col for col in selected_data.columns if "lag" in col.lower() or "FL" in col]

    feture = bfs.filter_correlated_features(data[columns])
    feture = feture + lag_col + [target]

    log_message(f"Final nodes selected features: {feture}")

    return data[feture]


    
