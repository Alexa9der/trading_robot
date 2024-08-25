import numpy as np
import pandas as pd
from datetime import datetime

import mlflow
import mlflow.catboost
import mlflow.xgboost
from mlflow.exceptions import MlflowException

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import  mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from trading_robot.data_collection.data_collector import DataCollector
from trading_robot.utils.logger import log_message

import logging
logging.getLogger("mlflow").setLevel(logging.DEBUG)

class ModelDeploymentManager:
    """
    A class to manage the lifecycle of machine learning models using MLflow.
    This includes registering models, transitioning stages, and deploying models.

    Attributes:
    client: MlflowClient - an instance of MLflow client for interacting with MLflow Model Registry.
    """

    def __init__(self):
        self.client = mlflow.tracking.MlflowClient()

    def register_and_deploy_best_model(self, run_id, best_model: BaseEstimator, model_name: str, 
                                        example_input: pd.DataFrame, scaler: StandardScaler):
        """
        Registers and deploys the best model found by GridSearchCV.

        Parameters:
        best_model: sklearn.base.BaseEstimator - The best model found.
        model_name: str - Name of the model to register and deploy.
        example_input: pd.DataFrame - Input data example.
        scaler: StandardScaler - Scaler object to log as an artifact.
        """
        try:
            log_message(f"Registering and deploying the best model: {model_name}")

            # Log model to MLflow
            model_uri = f"runs:/{run_id}/model"

            # Log the model as an artifact
            self.__log_model(best_model, model_name, input_example=example_input)

            log_message(f"Model logged to {model_uri}")

            # Log scaler parameters as an artifact
            # Log scaler parameters
            scaler_param = self.scaler_param(scaler=scaler)
            if scaler_param:
                mlflow.log_dict(scaler_param, "scaler_param.json")
            log_message(f"Scaler parameters logged as 'scaler_param.json'")

            # Try to create a registered model
            try:
                self.client.create_registered_model(model_name)
            except MlflowException as e:
                if "already exists" in str(e):
                    print(f"Model '{model_name}' already exists.")
                else:
                    raise e

            # Create model version
            model_version_info = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id,
                description="Best model found during hyperparameter tuning"
            )

            model_version = model_version_info.version
            log_message(f"Model version {model_version} created for '{model_name}'.")

        except MlflowException as e:
            print(f"MLflow Exception: {e}")
        except Exception as e:
            print(f"Unexpected Exception: {e}")

    def _log_additional_metrics(self, grid_search, X : pd.DataFrame, y: pd.Series, 
                                 model_name : str, scaler):
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
        try:
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Log sample predictions
            predictions = best_model.predict(X)

            # Logging an example of the model's results
            prediction_df = pd.DataFrame({'y_true': y, 'y_pred': predictions})
            prediction_df.index = prediction_df.index.astype(str)
            mlflow.log_dict(prediction_df.head(5).to_dict(), "sample_predictions.json")

            # log metric
            r2 = r2_score(y, predictions)
            mae = mean_absolute_error(y, predictions)
            rmse = self.rmse(y, predictions)

            mlflow.log_metric("R2", r2)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("rmse", rmse)

            # Log model
            example_input = X.tail(1).to_dict(orient='records')[0]
            mlflow.sklearn.log_model(best_model, "model", input_example=example_input)

             # Log model parameters
            mlflow.log_params(best_params)

            # Log scaler parameters
            scaler_param = self.__scaler_params(scaler=scaler)
            if scaler_param:
                mlflow.log_dict(scaler_param, "scaler_param.json")

            # Log dataset metadata
            mlflow.log_param("n_samples", X.shape[0])
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("last_data_update", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

            # # Log library versions
            # mlflow.log_param("pandas_version", pd.__version__)
            # mlflow.log_param("sklearn_version", sklearn.__version__)

        except Exception as e:
            print(f"Error logging additional metrics and data: {str(e)}")
            raise

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

    def __scaler_params(self, scaler):

        scaler_param = {}

        if hasattr(scaler, 'mean_'):
            scaler_param['mean'] = scaler.mean_.tolist()
        if hasattr(scaler, 'scale_'):
            scaler_param['scale'] = scaler.scale_.tolist()
        if hasattr(scaler, 'var_'):
            scaler_param['var'] = scaler.var_.tolist()
        if hasattr(scaler, 'min_'):
            scaler_param['min'] = scaler.min_.tolist()
        if hasattr(scaler, 'data_min_'):
            scaler_param['data_min'] = scaler.data_min_.tolist()
        if hasattr(scaler, 'data_max_'):
            scaler_param['data_max'] = scaler.data_max_.tolist()
        if hasattr(scaler, 'center_'):
            scaler_param['center'] = scaler.center_.tolist()
        if hasattr(scaler, 'scale_'):
            scaler_param['scale'] = scaler.scale_.tolist()
        if hasattr(scaler, 'quantiles_'):
            scaler_param['quantiles'] = scaler.quantiles_.tolist()

        return scaler_param

    def __log_model(self, model, model_name, input_example=None):
        """
        Логирует модель в MLflow в зависимости от типа модели.

        :param model: Объект модели для логирования.
        :param model_name: Название модели в MLflow.
        :param input_example: Пример входных данных для логирования модели (необязательно).
        """
        with mlflow.start_run() as run:
            if isinstance(model, CatBoostRegressor):
                mlflow.catboost.log_model(model, model_name, input_example=input_example)
                print(f"CatBoost model saved in run {run.info.run_id}")

            elif isinstance(model, XGBRegressor):
                mlflow.xgboost.log_model(model, model_name, input_example=input_example)
                print(f"XGBoost model saved in run {run.info.run_id}")

            elif hasattr(model, 'predict'):
                # Предполагаем, что это модель scikit-learn
                mlflow.sklearn.log_model(model, model_name, input_example=input_example)
                print(f"Scikit-learn model saved in run {run.info.run_id}")


if __name__ == "__main__":
    
    # 1. Data collection
    data_col = DataCollector()
    data = data_col.get_historical_data(symbol="EURUSD")

    X = data.drop(columns="Close")
    y = data["Close"]

    # 2. Prepare the scaler and apply it to the data
    scaler = StandardScaler()
    X = X.astype(np.float64)
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 3. Define the model and parameters for GridSearchCV
    model = Ridge()
    param_grid = {'alpha': [0.1, 1.0, 10.0]}
    grid_search = GridSearchCV(model, param_grid, cv=5)

    # 4. Find the best model
    grid_search.fit(X_scaled, y)
    best_model = grid_search.best_estimator_

    # 5. Create an MLflowModelManager object
    model_manager = ModelDeploymentManager()
    
    with mlflow.start_run() as active_run:

        #6. Register and deploy the best model
        model_manager.register_and_deploy_best_model(
            best_model=best_model,
            model_name="ridge",
            example_input=X_scaled.head(1).to_dict(orient='records')[0],  
            scaler=scaler  
        )

    print("Model registered and deployed successfully.")

