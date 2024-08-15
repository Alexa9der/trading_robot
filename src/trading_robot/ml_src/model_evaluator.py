import pandas as pd 
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Union

import mlflow
import mlflow.sklearn

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator

from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score

from trading_robot.data_collection.data_collector import DataCollector
from trading_robot.utils.logger import log_message
from trading_robot.ml_src.mlflow_deploy_menadger import ModelDeploymentManager

import warnings
warnings.filterwarnings("ignore")



class TimeSeriesModelEvaluator(ModelDeploymentManager):
    """
    A class for evaluating models and selecting hyperparameters specifically for time series data.

    Attributes:
    ----------
    n_splits : int
        The number of folds for TimeSeriesSplit.
    scaler : StandardScaler
        An object for scaling features.

    Methods:
    -------
    timeseriesCVscore(model, data, y_col, cv=5) -> float
        Evaluates the model using cross-validation for time series.

    timeseries_grid_search(model, param_grid, X, y, cv=5, model_name=None) -> GridSearchCV
        Searches for model hyperparameters using GridSearchCV for time series.

    tune_and_log_models(X, y, models=None, param_grids=None, cv=5, deploy=True) -> dict
        Iterates over models, selects hyperparameters, and logs the best models in MLflow with RMSE.
        Optionally deploys the best model to production.

    rmse(y_true, y_pred) -> float
        Calculates Root Mean Squared Error (RMSE).

    __log_additional_metrics(grid_search, X, y, model_name)
        Logs additional metrics and data to MLflow.

    __model() -> dict
        Returns a dictionary of models available for evaluation.

    __param() -> dict
        Returns a dictionary of parameter grids for hyperparameter tuning.
    """

    def __init__(self):
        """
        Initializes the TimeSeriesModelEvaluator with a StandardScaler for feature scaling.
        """
        super().__init__()
        self.scaler = StandardScaler()

    def timeseriesCVscore(self, model, data: pd.DataFrame, y_col: str, cv: int = 5) -> float:
        """
        Assesses the quality of the model using time series cross-validation.

        Parameters:
        ----------
        model : BaseEstimator
            A learning model having fit and predict methods.
        data : pd.DataFrame
            DataFrame containing features and the target variable.
        y_col : str
            The name of the column containing the target variable.
        cv : int, optional
            The number of folds for cross-validation. Default is 5.

        Returns:
        -------
        float
            The average root mean squared error (RMSE) across folds.
        """
        log_message(f"Evaluating model with time series cross-validation using target column '{y_col}'.")

        errors = []
        for train_index, test_index in TimeSeriesSplit(n_splits=cv).split(data):
            train, test = data.iloc[train_index], data.iloc[test_index]

            X_train = train.drop(columns=[y_col]).values
            y_train = train[y_col].values

            X_test = test.drop(columns=[y_col]).values
            y_test = test[y_col].values

            # Train the model and make predictions
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Calculate the error (RMSE) and add to the list
            error = self.rmse(y_test, predictions)
            errors.append(error)

        mean_error = np.mean(errors)
        
        log_message(f"Mean RMSE across folds: {mean_error}")

        return mean_error

    def timeseries_grid_search(self, model : Dict[str, BaseEstimator], param_grid : Dict[str, List[Union[float, int, str]]],
                               X: pd.DataFrame, y: pd.Series,
                               cv: int = 5, model_name: str | None = None):
        """
        Searches for the best hyperparameters for the model using GridSearchCV with time series cross-validation.

        Parameters:
        ----------
        model : BaseEstimator
            A model instance for learning (e.g., RandomForestRegressor).
        param_grid : dict
            Dictionary with parameters to search through.
        X : pd.DataFrame
            DataFrame containing features.
        y : pd.Series
            Series containing the target variable.
        cv : int, optional
            The number of folds for cross-validation. Default is 5.
        model_name : str, optional
            The name of the model being tuned. Default is None.

        Returns:
        -------
        GridSearchCV
            The GridSearchCV object containing the best model parameters and scores.
        """

        log_message(f"Starting grid search with TimeSeriesSplit.")

        scorer = make_scorer(self.rmse)

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                                   cv=TimeSeriesSplit(n_splits=cv), 
                                   scoring=scorer, n_jobs=-1)

        grid_search.fit(X, y)

        log_message(f"Model name: {model_name}")
        log_message(f"Best parameters found: {grid_search.best_params_}")
        log_message(f"Best score achieved: {grid_search.best_score_}")

        # Log additional metrics and data
        self.__log_additional_metrics(grid_search, X, y, model_name)
        
        return grid_search
    
    def tune_and_log_models(self, X: pd.DataFrame, y: pd.Series, models=None, param_grids=None, cv=5, deploy=True):
        """
        Tunes models, selects hyperparameters, and logs the best models in MLflow with RMSE. Optionally deploys the best model.

        Parameters:
        ----------
        X : pd.DataFrame
            DataFrame containing features.
        y : pd.Series
            Series containing the target variable.
        models : dict, optional
            Dictionary of model names and instances. Default is None, which uses built-in models.
        param_grids : dict, optional
            Dictionary of parameter grids for each model. Default is None, which uses built-in parameters.
        cv : int, optional
            The number of folds for cross-validation. Default is 5.
        deploy : bool, optional
            Whether to deploy the best model after tuning. Default is True.

        Returns:
        -------
        dict
            Dictionary containing RMSE values for each model.
        """
        if models is None:
            models = self.__model()

        if param_grids is None:
            param_grids = self.__param()

        results = {}

        # Convert data types to avoid issues with NaN in int columns
        X = X.astype(np.float64)

        # Scale features and retain feature names
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

        example_input = X_scaled.tail(1).to_dict(orient='records')[0]

        selected_model_rmse = np.inf
        selected_model = None
        selected_model_name = None
        parent_run_name = f"Model_Tuning_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        
        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            for model_name, candidate_model in models.items():
                with mlflow.start_run(run_name=model_name, nested=True):
                    param_grid = param_grids.get(model_name, {})

                    grid_search = self.timeseries_grid_search(
                        model=candidate_model,
                        param_grid=param_grid,
                        X=X_scaled,
                        y=y,
                        cv=cv,
                        model_name=model_name
                    )

                    best_model = grid_search.best_estimator_
                    best_rmse = grid_search.best_score_

                    results[model_name] = {'rmse': best_rmse}

                    # Select the model with the lowest RMSE
                    if best_rmse < selected_model_rmse:
                        selected_model_rmse = best_rmse
                        selected_model = best_model
                        selected_model_name = model_name

            # Register and deploy the best model if required
            if deploy and selected_model is not None:

                # The best model should be registered and deployed
                # Here we assume `selected_model` has the necessary attributes to register
                self.register_and_deploy_best_model(best_model=selected_model, model_name=selected_model_name, example_input=example_input)

        return results

    def rmse(self, y_true: pd.Series, y_pred: pd.Series):
        """
        Calculates Root Mean Squared Error (RMSE) between true and predicted values.

        Parameters:
        ----------
        y_true : pd.Series
            Series containing true values.
        y_pred : pd.Series
            Series containing predicted values.

        Returns:
        -------
        float
            The RMSE value.
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def __log_additional_metrics(self, grid_search, X : pd.DataFrame, y: pd.Series, 
                                 model_name : str):
        """
        Logs additional metrics and data to MLflow.

        Parameters:
        ----------
        grid_search : GridSearchCV
            The GridSearchCV object containing cross-validation results.
        X : pd.DataFrame
            DataFrame containing features.
        y : pd.Series
            Series containing the target variable.
        model_name : str
            Name of the model being logged.
        """

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Log sample predictions
        predictions = best_model.predict(X)

        # Logging an example of the model's results
        prediction_df = pd.DataFrame({'y_true': y, 'y_pred': predictions})
        prediction_df.index = prediction_df.index.astype(str)
        mlflow.log_dict(prediction_df.head(10).to_dict(), "sample_predictions.json")

        # log metric
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)

        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("rmse", grid_search.best_score_)

        # Log model
        example_input = X.tail(1).to_dict(orient='records')[0]
        mlflow.sklearn.log_model(best_model, "model", input_example=example_input)
        mlflow.log_params(best_params)

        # Log all cross-validation results
        results_df = pd.DataFrame(grid_search.cv_results_)
        mlflow.log_dict(results_df.to_dict(), "cv_results.json")

        # Log feature importances
        if hasattr(best_model, 'feature_importances_'):
            feature_importances = best_model.feature_importances_
            importance_dict = dict(zip(X.columns, feature_importances))
            mlflow.log_dict(importance_dict, "feature_importances.json")

            plt.figure(figsize=(10, 6))
            plt.barh(X.columns, feature_importances)
            plt.xlabel("Feature Importance")
            plt.title(f"Feature Importance for {model_name}")
            plt.savefig("feature_importances.png")
            mlflow.log_artifact("feature_importances.png")
            plt.close()

        # Log scaler parameters
        if hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
            mlflow.log_dict({
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist()
            }, "scaler_params.json")

        # Log dataset metadata
        mlflow.log_param("n_samples", X.shape[0])
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("last_data_update", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        # Log library versions
        mlflow.log_param("pandas_version", pd.__version__)
        mlflow.log_param("sklearn_version", sklearn.__version__)

    def __model(self):
        """
        Provides a dictionary of models available for evaluation.

        Returns:
        -------
        dict
            Dictionary of model names and their instances.
        """
        models = {
            'linear': LinearRegression(n_jobs=-1),
            'ridge': Ridge(random_state=42),
            'lasso': Lasso(random_state=42),
            'svr': SVR(),
            'decision_tree': DecisionTreeRegressor(random_state=42),
            'random_forest': RandomForestRegressor(n_jobs=-1, random_state=42),
            'gbr': GradientBoostingRegressor(random_state=42),
            'knn': KNeighborsRegressor(n_jobs=-1),
            'polynomial': Pipeline([
                ('poly', PolynomialFeatures()),
                ('linear', LinearRegression(n_jobs=-1))
            ])
        }

        return models 
    
    def __param(self):
        """
        Provides a dictionary of parameter grids for hyperparameter tuning.

        Returns:
        -------
        dict
            Dictionary of parameter grids for each model.
        """

        params = {
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                'max_iter': [1000, 5000, 10000, 15000]
            },
            'svr': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto'],
                'epsilon': [0.001, 0.01, 0.1, 1]
            },
            'decision_tree': {
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            },
            'gbr': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 10, 15],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]  # 1: Manhattan, 2: Euclidean
            },
            'polynomial': {
                'poly__degree': [2, 3, 4],
                'linear__fit_intercept': [True, False],
                'linear__normalize': [True, False]
            }
        }

        return params


if __name__ == "__main__":

    data_col = DataCollector()
    data = data_col.get_historical_data(symbol="EURUSD")

    tsem = TimeSeriesModelEvaluator()

    X = data.drop(columns="Close")
    y = data["Close"]

    models = {
        'decision_tree': DecisionTreeRegressor(random_state=42),
        'ridge': Ridge(random_state=42),
        'lasso': Lasso(random_state=42),
    }

    params = {
        'ridge': {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        },
        'lasso': {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
            'max_iter': [10000, 15000, 20000]
        },
        'decision_tree': {
            'max_depth': [None, 10, 20, 30, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }
    }

    results = tsem.tune_and_log_models(X, y, models=models, param_grids=params, deploy=True)
    print(results)



