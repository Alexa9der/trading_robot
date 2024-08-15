import mlflow
from mlflow.exceptions import MlflowException
from sklearn.model_selection import GridSearchCV



class ModelDeploymentManager:
    """
    A class to manage the lifecycle of machine learning models using MLflow.
    This includes registering models, transitioning stages, and deploying models.

    Attributes:
    client: MlflowClient - an instance of MLflow client for interacting with MLflow Model Registry.
    """

    def __init__(self):
        self.client = mlflow.tracking.MlflowClient()

    def register_and_deploy_best_model(self, best_model : GridSearchCV, model_name : str, example_input : pd.DataFrame):
        """
        Registers and deploys the best model found by GridSearchCV.

        Parameters:
        best_model: sklearn.base.BaseEstimator - The best model found.
        model_name: str - Name of the model to register and deploy.
        input_example: pd. Input data example
        """
        try:
            # Ensure MLflow is active
            active_run = mlflow.active_run()
            if active_run is None:
                raise RuntimeError("MLflow is not running. Ensure that mlflow.start_run() is called.")

            # Log model to MLflow
            model_uri = f"runs:/{active_run.info.run_id}/model"

            # Log the model as an artifact
            mlflow.sklearn.log_model(best_model, "model", input_example=example_input)
            
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
                run_id=active_run.info.run_id,
                description="Best model found during hyperparameter tuning"
            )
            model_version = model_version_info.version
            print(f"Model version {model_version} created for '{model_name}'.")

        except MlflowException as e:
            print(f"MLflow Exception: {e}")



