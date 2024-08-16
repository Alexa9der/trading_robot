import pandas as pd 
import numpy as np
from datetime import datetime
from typing import Dict, List, Union, Optional

import mlflow
import mlflow.sklearn

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

from sklearn.metrics import make_scorer, mean_squared_error

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

    """

    def __init__(self, scaler = None):
        """
        Initializes the TimeSeriesModelEvaluator with a StandardScaler for feature scaling.
        """
        super().__init__()
        self.scaler = scaler if scaler is not None else StandardScaler()

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

    def timeseries_grid_search(self, model : BaseEstimator, 
                               param_grid : Dict[str, List[Union[float, int, str]]],
                               X: pd.DataFrame, y: pd.Series, cv: int = 5):
        """
        Searches for the best hyperparameters for the model using GridSearchCV with time series cross-validation.

        Parameters:
        ----------
        model : BaseEstimator
            A model instance for learning (e.g., RandomForestRegressor).
        param_grid : Dict[str, List[Union[float, int, str]]]
            Dictionary with parameters to search through, where the keys are the parameter names and 
            the values are lists of parameter settings to try.
        X : pd.DataFrame
            DataFrame containing features for training the model.
        y : pd.Series
            Series containing the target variable for training the model.
        cv : int, optional
            The number of folds for time series cross-validation. Default is 5.
        Returns:
        -------
        GridSearchCV
            The GridSearchCV object containing the best model parameters and scores, along with other 
            details about the cross-validation process.

        Notes:
        ------
        - The function uses TimeSeriesSplit for cross-validation, which is suitable for time series data
        where the order of observations is important.
        - A custom RMSE (Root Mean Square Error) scorer is used to evaluate the models.
        - If mlflow_log_metrics is set to True, additional metrics and details are logged to MLflow 
        to help track the performance of the model with the chosen parameters.
        
        Example:
        --------
        >>> model = RandomForestRegressor()
        >>> param_grid = {
        >>>     'n_estimators': [100, 200],
        >>>     'max_depth': [10, 20, None],
        >>> }
        >>> grid_search_result = timeseries_grid_search(model, param_grid, X, y, cv=5)
        >>> print(grid_search_result.best_params_)
        """
        
        scorer = make_scorer(self.rmse)

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                                   cv=TimeSeriesSplit(n_splits=cv), 
                                   scoring=scorer, n_jobs=-1)

        grid_search.fit(X, y)

        log_message(f"Best parameters found: {grid_search.best_params_}")
        log_message(f"Best score achieved: {grid_search.best_score_}")
        
        return grid_search
    
    def tune_and_log_models(self, X: pd.DataFrame, y: pd.Series, 
                            models : None | Dict[str, BaseEstimator] = None, 
                            param_grids : None | Dict[str, Dict[str, List[Union[float, int, str]]]] = None, 
                            cv=5, mlflow_log_metrics : bool = True, deploy=True):
        """
        Tunes multiple models by selecting the best hyperparameters, logs the models and their metrics to MLflow, 
        and optionally deploys the best model.

        Parameters:
        ----------
        X : pd.DataFrame
            DataFrame containing the features used for training the models.
        y : pd.Series
            Series containing the target variable for training the models.
        models : dict, optional
            A dictionary where keys are model names and values are instances of the models (e.g., {'ridge': Ridge(), 'lasso': Lasso()}).
            Default is None, which will use predefined models.
        param_grids : dict, optional
            A dictionary where keys are model names and values are parameter grids for those models. 
            Each parameter grid is a dictionary where the keys are parameter names and the values are lists of parameter values to try.
            Default is None, which will use predefined parameter grids.
        cv : int, optional
            The number of folds for cross-validation. Default is 5.
        deploy : bool, optional
            Whether to deploy the best model after hyperparameter tuning and model selection. Default is True.
        mlflow_log_metrics : bool, optional
            If True, logs additional metrics and model parameters to MLflow for tracking. Default is True.

        Returns:
        -------
        dict
            A dictionary where keys are model names and values are dictionaries containing the RMSE values 
            ('rmse' key) for the corresponding models.

        Notes:
        ------
        - This function iterates over the provided models, applies grid search with time series cross-validation, 
        and logs the best model for each algorithm to MLflow.
        - The best model across all candidates, based on RMSE, can be automatically deployed if `deploy` is set to True.
        - The function also handles data preprocessing, including scaling of features, to ensure consistency 
        during the model training and evaluation processes.
        - The example input data is captured from the last row of the scaled features, which is used in the model registration process.

        Example:
        --------
        >>> from sklearn.linear_model import Ridge, Lasso
        >>> model_dict = {'ridge': Ridge(), 'lasso': Lasso()}
        >>> param_grid_dict = {
        >>>     'ridge': {'alpha': [0.1, 1.0, 10.0]},
        >>>     'lasso': {'alpha': [0.001, 0.01, 0.1]}
        >>> }
        >>> results = tune_and_log_models(X, y, models=model_dict, param_grids=param_grid_dict, cv=5, deploy=True)
        >>> print(results)
        """
        if models is None:
            models = self.__model()

        if param_grids is None:
            param_grids = self.__param()

        X = self.__transform(X)

        results = {}
        selected_model_rmse = np.inf
        selected_model = None
        selected_model_name = None
        run_id = None

        parent_run_name = f"Model_Tuning_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        log_message(f"Starting model tuning with parent run name: {parent_run_name}")

        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            for model_name, candidate_model in models.items():
                with mlflow.start_run(run_name=model_name, nested=True):

                    log_message(f"Starting grid search with TimeSeriesSplit for model '{model_name}'.")

                    param_grid = param_grids.get(model_name, {})

                    grid_search = self.timeseries_grid_search(
                        model=candidate_model,
                        param_grid=param_grid,
                        X=X,
                        y=y,
                        cv=cv,
                    )

                    best_model = grid_search.best_estimator_
                    best_rmse = grid_search.best_score_

                    results[model_name] = {'rmse': best_rmse}

                    # Select the model with the lowest RMSE
                    if best_rmse < selected_model_rmse:
                        selected_model_rmse = best_rmse
                        selected_model = best_model
                        selected_model_name = model_name
                        active_run = mlflow.active_run()
                        run_id = active_run.info.run_id

                    # Log additional metrics and data
                    if mlflow_log_metrics:
                        log_message(f"Logging additional metrics to MLflow.")
                        self._log_additional_metrics(grid_search, X, y, model_name, self.scaler)

            # Register and deploy the best model if required
            if deploy and selected_model is not None:
                example_input = X.tail(1).to_dict(orient='records')[0]


                log_message(f"Deploying the best model: {selected_model_name}")
                # The best model should be registered and deployed
                # Here we assume `selected_model` has the necessary attributes to register
                self.register_and_deploy_best_model(run_id=run_id, best_model=selected_model, model_name=selected_model_name, 
                                                    example_input=example_input, scaler= self.scaler )

        log_message("Model tuning and logging complete")
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

    def __is_fitted(self, transformer: BaseEstimator):
        """
        The function checks if the transformer from Scikit-learn is trained.

        :param transformer: transformer object (e.g. StandardScaler, MinMaxScaler, PCA)
        :return: True if the transformer is trained, otherwise False
        """
        fitted_attributes = {
            'StandardScaler': 'n_samples_seen_',
            'MinMaxScaler': 'min_',
            'MaxAbsScaler': 'scale_',
            'RobustScaler': 'center_',
            'Normalizer': 'n_features_in_',  
            'PCA': 'components_',
            'KernelPCA': 'dual_coef_',
            'IncrementalPCA': 'components_',
            'TruncatedSVD': 'components_',
            'NMF': 'components_',
            'FactorAnalysis': 'components_',
            'DictionaryLearning': 'components_',
            'FastICA': 'components_',
            'GaussianRandomProjection': 'components_',
            'SparseRandomProjection': 'components_',
            'Binarizer': 'threshold',  
            'QuantileTransformer': 'n_quantiles_',
            'PowerTransformer': 'lambdas_',
            'FunctionTransformer': 'n_features_in_',  
        }
        
        transformer_name = transformer.__class__.__name__
        attribute = fitted_attributes.get(transformer_name)
        
        if attribute:
            return hasattr(transformer, attribute)
        else:
            raise ValueError(f"Unknown transformer: {transformer_name}")
        
    def __transform(self, X:  pd.DataFrame):
        """
        Applies scaling to the input data if the scaler has not been fitted.
        
        This method ensures that all data types in the DataFrame are converted to float64 to
        avoid issues with NaN values in integer columns. If the scaler has already been fitted, 
        it simply returns the input data as is. Otherwise, it fits the scaler to the input data, 
        scales the data, and returns the scaled DataFrame with the original column names.
        
        :param X: pd.DataFrame - The input data to be transformed.
        :return: pd.DataFrame - The transformed (scaled) data.
        """
        
        # Convert data types to avoid issues with NaN in integer columns
        X = X.astype(np.float64)

        # Check if the scaler has already been fitted
        if self.__is_fitted(self.scaler):
            # If the scaler is fitted, return the data as is
            X_scaled = X
        else:
            # If the scaler is not fitted, fit and transform the data
            # Convert the scaled data back into a DataFrame, retaining original column names
            X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

        # Return the scaled data
        return X_scaled

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

    results = tsem.tune_and_log_models(X, y, models=models, param_grids=params, 
                                       deploy= True, mlflow_log_metrics = True)
    print(results)



