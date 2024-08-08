import sys
import os

import pandas as pd 
import numpy as np

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from data_collection.data_collector import DataCollector



class TimeSeriesModelEvaluator:
    """
    A class for evaluating models and selecting hyperparameters, taking into account time series.

    Attributes:
    N_SPLITS: Int - the number of folds for TimeserIssplit.
    Scaler: Standardscaler - an object for scaling signs.
    TSCV: TimeserIssplit - Cross -Washing for time series.

    Methods:
    Timeseriescvscore (Model, Data, Y_col):
    Evaluates the model using cross-novels for temporary rows.

    Timeseries_grid_search (Model, Param_grid, Data, Y_col):
    It searches for hyperparameters of the model using Gridsearchcv for time rows.
    """

    def __init__(self, n_splits: int = 5, data_scaler : bool = True):
        """
        Class for evaluating models and selecting hyperparameters for temporary rows.

        Options:
        N_SPLITS: Int - the number of folds for TimeserIssplit.
        """
        self.n_splits = n_splits
        self.scaler = StandardScaler()
        self.tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
    def timeseriesCVscore(self, model, data: pd.DataFrame, y_col: str) -> float:
        """
        Assesses the quality of the model using cross-novels for temporary rows.

        Options:
        Model: A learning model having FIT and Predict methods.
        Data: PD.Dataframe - data containing signs and target variable.
        y_col: str - the name of the column with the target variable.

        Returns:
        Float is a medium -sequatratic error averaged by folds.
        """
        errors = []

        for train_index, test_index in self.tscv.split(data):
            train, test = data.iloc[train_index], data.iloc[test_index]

            X_train = train.drop(columns=[y_col]).values
            y_train = train[y_col].values

            X_test = test.drop(columns=[y_col]).values
            y_test = test[y_col].values

           # We scalit signs
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            # Teaching the model and making a forecast
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # We count the error (rmse) and add to the list
            error = root_mean_squared_error(y_test, predictions)
            errors.append(error)

        return np.mean(errors)

    def timeseries_grid_search(self, model, param_grid, data: pd.DataFrame, y_col: str):
        """
        It searches for hyperparameters of the model using Gridsearchcv for time rows.

        Options:
        Model: a model for learning (for example, RandomForestregressor).
        PARAM_GRID: DICT - Hyperparameter grid for search.
        Data: PD.Dataframe - data containing signs and target variable.
        y_col: str - the name of the column with the target variable.

        Returns:
        Gridsearchcv - an object with the best model parameters.
        """
        X = data.drop(columns=[y_col]).values
        X = self.scaler.fit_transform(X)
        y = data[y_col]

        scorer = make_scorer(root_mean_squared_error)

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                                   cv=self.tscv, scoring=scorer, n_jobs=-1)

        grid_search.fit(X, y)

        return grid_search



if __name__ == "__main__":
    # Data collection
    data_col = DataCollector()
    data = data_col.get_historical_data(symbol="EURUSD")

    # Initialize TimeSeriesModelEvaluator and RandomForestRegressor model
    evaluator = TimeSeriesModelEvaluator(n_splits=5)
    model = RandomForestRegressor()

    # Define hyperparameter grid for search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15]
    }

    # Hyperparameter search
    best_model = evaluator.timeseries_grid_search(model, param_grid, data, y_col='Close')

    # Cross-validation using the best model
    rmse = evaluator.timeseriesCVscore(model=best_model.best_estimator_, data=data, y_col='Close')

    # Output results
    print(f"Best parameters: {best_model.best_params_}/n")
    print(f"Best RMSE result on validation data: {best_model.best_score_}/n")
    print(f"RMSE on cross-validation: {rmse}")

