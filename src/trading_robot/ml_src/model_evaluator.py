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

from sklearn.metrics import make_scorer

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

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

    __models = {
        'xgboost': XGBRegressor(random_state=42),
        'catboost': CatBoostRegressor(random_state=42, silent=True),
        'random_forest': RandomForestRegressor(n_jobs=-1, random_state=42),
        'gbr': GradientBoostingRegressor(random_state=42),
        'svr': SVR(),
        'polynomial': Pipeline([
            ('poly', PolynomialFeatures()),
            ('linear', LinearRegression(n_jobs=-1))
        ]),
        'linear': LinearRegression(n_jobs=-1),
        'ridge': Ridge(random_state=42, max_iter=15000),
        'lasso': Lasso(random_state=42, max_iter=15000),
        'knn': KNeighborsRegressor(n_jobs=-1),
        
    }

    __params = {
                'catboost': {
                    'depth': range(4, 11),  
                    'learning_rate': np.arange(0.01, 0.1, 0.01),
                    'iterations': range(100, 1100, 100),
                    'l2_leaf_reg': np.arange(1, 10, 2),
                    'border_count': range(32, 129, 32)
                },
                'xgboost': {
                    'n_estimators': range(100, 1100, 100),
                    'max_depth': range(3, 10, 2),
                    'learning_rate': np.arange(0.01, 0.1, 0.01),
                    'subsample': np.arange(0.5, 1.1, 0.1),
                    'colsample_bytree': np.arange(0.5, 1.1, 0.1),
                    'gamma': [0, 0.1, 0.2, 0.3],
                    'reg_alpha': [0, 0.01, 0.1, 1],
                    'reg_lambda': [1, 0.1, 0.01, 0]
                },
                'random_forest': {
                    'n_estimators': range(100, 1100, 100), 
                    'max_depth': range(1, 16, 2), 
                    'min_samples_split': range(2, 12, 2), 
                    'min_samples_leaf': range(2, 8, 2),
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'bootstrap': [True, False]
                },
                'gbr': {
                    'n_estimators': range(100, 1100, 100),
                    'max_depth': range(1, 11, 2),
                    'subsample': np.arange(0.5, 1.1, 0.1),
                    'min_samples_split': range(2, 8, 2),
                    'min_samples_leaf': range(2, 8, 2),
                    'learning_rate': np.arange(0.1, 1.1, 0.1),
                    'max_features': ['auto', 'sqrt', 'log2'],
                    "criterion": ["friedman_mse", "mse", "mae"],
                    'loss': ['ls', 'lad', 'huber', 'quantile']
                },
                'svr': {
                    'C': [0.01, 0.1, 1, 10, 100, 1000],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto'],
                    'epsilon': [0.001, 0.01, 0.1, 1]
                },
                'polynomial': {
                    'poly__degree': range(2, 8, 2),  
                    'linear__fit_intercept': [True, False],
                    "poly__interaction_only": [True, False]
                },
                'ridge': {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                    'fit_intercept': [True, False]
                },
                'lasso': {
                    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                    'fit_intercept': [True, False]
                },
                'knn': {
                    'n_neighbors': range(2, 17, 2),
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]  # 1: Manhattan, 2: Euclidean
                },
            }

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
                            X_test: pd.DataFrame, y_test: pd.Series, 
                            models : None | Dict[str, BaseEstimator] = None, 
                            param_grids : None | Dict[str, Dict[str, List[Union[float, int, str]]]] = None, 
                            cv=5, mlflow_log_metrics : bool = True, deploy=True):
        """
            Tunes hyperparameters for multiple models, logs them and their metrics to MLflow, 
            and optionally deploys the best model.

            Parameters:
            ----------
            X : pd.DataFrame
                DataFrame containing the features used for training the models.
            y : pd.Series
                Series containing the target variable for training the models.
            X_test : pd.DataFrame
                DataFrame containing the features used for testing the models.
            y_test : pd.Series
                Series containing the target variable for testing the models.
            models : dict, optional
                A dictionary where keys are model names and values are instances of the models 
                (e.g., {'ridge': Ridge(), 'lasso': Lasso()}). Default is None, which will use predefined models.
            param_grids : dict, optional
                A dictionary where keys are model names and values are parameter grids for those models. 
                Each parameter grid is a dictionary where the keys are parameter names and the values are lists of parameter values to try. 
                Default is None, which will use predefined parameter grids.
            cv : int, optional
                The number of folds for cross-validation. Default is 5.
            mlflow_log_metrics : bool, optional
                If True, logs additional metrics and model parameters to MLflow for tracking. Default is True.
            deploy : bool, optional
                If True, the best model is automatically deployed after hyperparameter tuning and model selection. 
                Default is True.

            Returns:
            -------
            dict
                A dictionary where keys are model names and values are dictionaries containing the RMSE values 
                ('rmse' key) for the corresponding models.

            Notes:
            ------
            - This method iterates over the provided models, applies Grid Search with TimeSeriesSplit for cross-validation, 
            and logs the best model for each algorithm to MLflow.
            - The best model across all candidates, based on RMSE, can be automatically deployed if `deploy` is set to True.
            - The method also handles data preprocessing, including scaling of features, to ensure consistency 
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
            >>> results = tune_and_log_models(X_train, y_train, X_test, y_test, models=model_dict, param_grids=param_grid_dict, cv=5, deploy=True)
            >>> print(results)
            """
        if models is None:
            models = self.__models

        if param_grids is None:
            param_grids = self.__params

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

                    # Get the parameter grid for the current model
                    param_grid = param_grids.get(model_name, {})

                    # Grid Search with TimeSeriesSplit
                    grid_search = self.timeseries_grid_search(
                        model=candidate_model,
                        param_grid=param_grid,
                        X=X,
                        y=y,
                        cv=cv,
                    )

                    # Best model after Grid Search
                    best_model = grid_search.best_estimator_

                    # Scale test data similarly to training data
                    X_test = pd.DataFrame(self.scaler.fit_transform(X_test), columns=X_test.columns)
                    
                    # Predict on test data
                    predict = best_model.predict(X_test)  

                    # Calculate RMSE for predictions on test data
                    rmse = self.rmse(y_test, predict)

                    # Store RMSE results for the current model
                    results[model_name] = {'rmse': rmse}

                    # Select the model with the lowest RMSE
                    if rmse < selected_model_rmse:
                        selected_model_rmse = rmse
                        selected_model = best_model
                        selected_model_name = model_name
                        active_run = mlflow.active_run()
                        run_id = active_run.info.run_id

                    # Log additional metrics and data
                    if mlflow_log_metrics:
                        log_message(f"Logging additional metrics to MLflow.")
                        self._log_additional_metrics(grid_search, X_test, y_test, model_name, self.scaler)

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

    # Create an instance of the DataCollector class
    data_col = DataCollector()

    # Fetch historical data for the EURUSD symbol
    data = data_col.get_historical_data(symbol="EURUSD")

    # Define split point (e.g., 80% of the data for training and 20% for testing)
    split_index = int(len(data) * 0.8)  # Calculate the index to split the data

    # Split the data into training and testing sets
    train = data.iloc[:split_index]  
    test = data.iloc[split_index:]   

    # Separate features (X) and target variable (y) from the dataset
    X_train = train.drop(columns="Close")
    y_train = train["Close"]               

    X_test = test.drop(columns="Close")
    y_test = test["Close"]               

    # Define a dictionary of models to be tuned
    models = {
        'decision_tree': DecisionTreeRegressor(random_state=42),  
        'ridge': Ridge(random_state=42),                          
        'lasso': Lasso(random_state=42),                          
    }

    # Define a dictionary of hyperparameter grids for each model
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

    # Create an instance of TimeSeriesModelEvaluator to handle model evaluation and tuning
    tsem = TimeSeriesModelEvaluator()

    # Tune models, log results to MLflow, and deploy the best model if needed
    results = tsem.tune_and_log_models(X=X_train, 
                                       y=y_train,
                                       X_test=X_test, 
                                       y_test=y_test, 
                                    #    models=models, 
                                    #    param_grids=params, 
                                       deploy=True, 
                                       mlflow_log_metrics=True)


    print(results)


