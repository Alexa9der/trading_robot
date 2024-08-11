from dotenv import load_dotenv
load_dotenv()

from typing import Dict, Any

# Import classes from the appropriate modules
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

    loader = DataCollector()
    data = loader.get_historical_data('EURUSD')

    return data

def nodes_inzener_features(data: Dict[str, Any]) -> Dict[str, Any]:

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

    tl = TLIndicators(data=data)
    data = tl.all_talib_indicators()

    return data

def nodes_select_features(data: Dict[str, Any]) -> Dict[str, Any]:

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

    return data[feture]

def nodes_train_test_split(data: Dict[str, Any]) -> Dict[str, Any]:

    splitter = TimeSeriesSplits()
    X_train, y_train, X_test, y_test = splitter.train_test_split(data, "Close")

    return X_train, y_train, X_test, y_test



if __name__ == "__main__":
    data = nodes_load_data()
    data = nodes_inzener_features(data)
    data = nodes_select_features(data)
    X_train, y_train, X_test, y_test = nodes_train_test_split(data)
