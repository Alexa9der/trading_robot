import numpy as np
import pandas as pd
from datetime import datetime

import mlflow
from mlflow.exceptions import MlflowException

import sklearn
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import  mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from trading_robot.data_collection.data_collector import DataCollector
from trading_robot.utils.logger import log_message



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
            mlflow.sklearn.log_model(best_model, "model", input_example=example_input)
            log_message(f"Model logged to {model_uri}")

            # Log scaler parameters as an artifact
            scaler_params = {
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist()
            }
            mlflow.log_dict(scaler_params, "scaler_params.json")
            log_message(f"Scaler parameters logged as 'scaler_params.json'")

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

            # # Log feature importances
            # if hasattr(best_model, 'feature_importances_'):
            #     feature_importances = best_model.feature_importances_
            #     importance_dict = dict(zip(X.columns, feature_importances))
            #     mlflow.log_dict(importance_dict, "feature_importances.json")

            #     plt.figure(figsize=(10, 6))
            #     plt.barh(X.columns, feature_importances)
            #     plt.xlabel("Feature Importance")
            #     plt.title(f"Feature Importance for {model_name}")
            #     plt.savefig("feature_importances.png")
            #     mlflow.log_artifact("feature_importances.png")
            #     plt.close()

            # Log scaler parameters
            if hasattr(scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
                mlflow.log_dict({
                    'mean': scaler.mean_.tolist(),
                    'scale': scaler.scale_.tolist()
                }, "scaler_params.json")

            # Log dataset metadata
            mlflow.log_param("n_samples", X.shape[0])
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("last_data_update", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

            # Log library versions
            mlflow.log_param("pandas_version", pd.__version__)
            mlflow.log_param("sklearn_version", sklearn.__version__)

        except Exception as e:
            print(f"Error logging additional metrics and data: {str(e)}")
            raise



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

