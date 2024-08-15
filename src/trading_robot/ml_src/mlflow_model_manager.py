import json
import os
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
        log_message(f"Latest registered model: {latest_model.name}")  
        
        return latest_model.name

    def get_latest_model_version_info(self, model_name: str) -> dict:
            """
            Retrieves the latest version info of a registered model, including the run_id.

            Parameters:
            model_name: str - The name of the registered model in the MLflow Model Registry.

            Returns:
            dict - A dictionary containing version and run_id.
            """
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            if not versions:
                raise ValueError(f"Model with name '{model_name}' not found.")
            
            # Find the version with the highest number
            latest_version = max(versions, key=lambda v: v.version)
            log_message(f"Latest model version: {latest_version.version}")  # Debug output
            
            # Return model version info as a dictionary
            return {
                "version": latest_version.version,
                "run_id": latest_version.run_id
            }

    def load_model(self):
        """
        Loads the latest version of a model from the MLflow Model Registry.

        Returns:
        mlflow.pyfunc.PythonModel - The loaded model.
        """
        if self.model_name is None:
            self.model_name = self.get_latest_model_name()


        version_info = self.get_latest_model_version_info(self.model_name)
        latest_version = version_info["version"]
        model_uri = f"models:/{self.model_name}/{latest_version}"

        log_message(f"Loading model from URI: {model_uri}")  
        try:
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

    def load_scaler_params(self) -> dict:
        """
        Loads scaler parameters from a JSON artifact in MLflow.

        Returns:
        dict - The scaler parameters, including 'mean' and 'scale'.
        """
        if self.model_name is None:
            self.model_name = self.get_latest_model_name()

        version_info = self.get_latest_model_version_info(self.model_name)
        run_id = version_info["run_id"]

        # Create the artifact URI
        artifact_path = "scaler_params.json"
        artifact_uri = f"runs:/{run_id}/{artifact_path}"
        log_message(f"Loading scaler parameters from artifact URI: {artifact_uri}")


        local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)

        log_message(f"local_file_path: {local_path}")

        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Scaler parameters file not found at {local_path}")
        
        # Read the downloaded JSON file
        with open(local_path, 'r') as file:
            scaler_params = json.load(file)
        return scaler_params

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

    scaler_params = model_manager.load_scaler_params()
    print(scaler_params)

    # Create a StandardScaler object
    scaler = StandardScaler()

    # Set the parameters manually
    scaler.mean_ = np.array(scaler_params['mean'])
    scaler.scale_ = np.array(scaler_params['scale'])
