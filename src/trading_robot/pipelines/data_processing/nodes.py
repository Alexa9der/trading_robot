from dotenv import load_dotenv
load_dotenv()

from typing import Dict, Any

# Import classes from the appropriate modules
from trading_robot.utils.logger import log_message
from trading_robot.data_collection.data_collector import DataCollector
from trading_robot.future_inzenering.basic_future_inz import BasicFuture
from trading_robot.future_inzenering.talib_indicators import TLIndicators
from trading_robot.future_selection.basic_feature_selector import BasicFeatureSelector
from trading_robot.feture_split.time_series_split import TimeSeriesSplits

# from model.model_evaluator import TimeSeriesModelEvaluator
# from analysis.analyze_liquidity import AnalyzeLiquidity
# from strategies.strategy import Strategy
# from strategies.tuner import SMATuner
# from traiding.risk_manager import RiskManager
# from traiding.trading import Trading
# from utils.message import PushbulletMagic


def nodes_load_data() -> Dict[str, Any]:

    log_message("Starting data loading nodes.")

    loader = DataCollector()
    data = loader.get_historical_data('EURUSD')

    log_message(f"Data loaded nodes successfully. Data keys: {list(data.keys())}")

    return data

def nodes_inzener_features(data: Dict[str, Any]) -> Dict[str, Any]:

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

def nodes_select_features(data: Dict[str, Any]) -> Dict[str, Any]:

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

def nodes_train_test_split(data: Dict[str, Any]) -> Dict[str, Any]:

    log_message("Starting train-test split nodes.")

    splitter = TimeSeriesSplits()
    X_train, y_train, X_test, y_test = splitter.train_test_split(data, "Close")

    log_message("Train-test split nodes completed.")
    log_message(f"Training data size: X_train: {X_train.shape}, y_train: {y_train.shape}")
    log_message(f"Testing data size: X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, y_train, X_test, y_test



if __name__ == "__main__":
    data = nodes_load_data()
    data = nodes_inzener_features(data)
    data = nodes_select_features(data)
    X_train, y_train, X_test, y_test = nodes_train_test_split(data)
