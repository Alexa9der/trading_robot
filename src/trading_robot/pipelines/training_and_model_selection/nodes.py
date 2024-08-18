from typing import Dict, Any, Tuple
import pandas as pd

# Import classes from the appropriate modules
from trading_robot.utils.logger import log_message
from trading_robot.feture_split.time_series_split import TimeSeriesSplits
from trading_robot.ml_src.model_evaluator import TimeSeriesModelEvaluator

from trading_robot.data_collection.data_collector import DataCollector


def nodes_train_test_split(data: pd.DataFrame) ->  Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

    log_message("Starting train-test split nodes.")

    splitter = TimeSeriesSplits()
    X_train, y_train, X_test, y_test = splitter.train_test_split(data, "Close")



    log_message("Train-test split nodes completed.")
    log_message(f"Training data size: X_train: {X_train.shape}, y_train: {y_train.shape}")
    log_message(f"Testing data size: X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, y_train, X_test, y_test



def nodes_ml_tune(X_train: pd.DataFrame,  y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:

    tsem = TimeSeriesModelEvaluator()

    results = tsem.tune_and_log_models(X=X_train, y=y_train,
                                       X_test=X_test, y_test=y_test,
                                       deploy=True, mlflow_log_metrics=True)
    
    return results




def nodes_load_data() -> pd.DataFrame:

    log_message("Starting data loading nodes.")

    loader = DataCollector()
    data = loader.get_historical_data('EURUSD')

    log_message(f"Data loaded nodes successfully. Data keys: {list(data.keys())}")

    return data


data = nodes_load_data()
nodes_train_test_split(data)