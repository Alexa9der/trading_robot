import numpy as np
import pandas as pd

import mlflow
import mlflow.pyfunc

from trading_robot.utils.logger import log_message


class MLflowModelManager:
    """
    A class for managing the loading and usage of models from the MLflow Model Registry.
    Automatically selects the latest registered model.
    """

    def __init__(self, model_name=None):
        """
        Initializes the model manager.

        Parameters:
        model_name: str - Optional, the name of the registered model in the MLflow Model Registry.
        """
        self.model_name = model_name
        self.client = mlflow.tracking.MlflowClient()

    def get_latest_model_name(self) -> str:
        """
        Retrieves the name of the latest registered model from the MLflow Model Registry.

        Returns:
        str - The name of the latest registered model.
        """
        models = self.client.search_registered_models()
        
        if not models:
            raise ValueError("No registered models found in MLflow Model Registry.")
        
        # Select the model with the latest registration date
        latest_model = max(models, key=lambda model: model.creation_timestamp)
        log_message(f"Latest registered model: {latest_model.name}")  # Debug output
        
        return latest_model.name

    def get_latest_model_version(self, model_name: str) -> str:
        """
        Retrieves the URI of the latest version of a registered model.

        Parameters:
        model_name: str - The name of the registered model in the MLflow Model Registry.

        Returns:
        str - The URI of the latest model version.
        """
        versions = self.client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            raise ValueError(f"Model with name '{model_name}' not found.")
        
        # Find the version with the highest number
        latest_version = max(versions, key=lambda v: v.version)
        log_message(f"Latest model version: {latest_version.version}")  # Debug output
        
        # Return the model URI
        return f"models:/{model_name}/{latest_version.version}"

    def load_model(self):
        """
        Loads the latest version of a model from the MLflow Model Registry.

        Returns:
        mlflow.pyfunc.PythonModel - The loaded model.
        """
        if self.model_name is None:
            self.model_name = self.get_latest_model_name()

        model_uri = self.get_latest_model_version(self.model_name)
        log_message(f"Loading model from URI: {model_uri}")  # Debug output
        try:
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

    def predict(self, X: pd.DataFrame):
        """
        Makes predictions based on the loaded model.

        Parameters:
        X: pd.DataFrame - The dataset for predictions.

        Returns:
        np.ndarray - The model's predictions.
        """
        model = self.load_model()
        return model.predict(X)

# Example usage of the class
if __name__ == "__main__":
    # Create an instance of the class without specifying a model name
    model_manager = MLflowModelManager()

    latest_model = model_manager.load_model()
    print(latest_model)

