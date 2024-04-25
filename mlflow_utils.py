import os
import mlflow
from typing import Any

def set_mlflow_tracking_uri(tracking_uri: str = None) -> None:
    if "AWS_ACCESS_KEY_ID" not in os.environ or "AWS_SECRET_ACCESS_KEY" not in os.environ or "MLFLOW_S3_ENDPOINT_URL" not in os.environ:
        raise EnvironmentError("Required environment variables are not set")

    # Directly use the environment variables
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ["AWS_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["AWS_SECRET_ACCESS_KEY"]
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.environ["MLFLOW_S3_ENDPOINT_URL"]

def create_mlflow_experiment(experiment_name: str, artifact_location: str, tags: dict[str, Any]) -> str:
    """
    Create a new mlflow experiment with the given name, artifact location, and tags.
    """
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
            tags=tags
        )
    except mlflow.exceptions.MlflowException as e:
        print("Error: {}".format(e))
        if "already exists" in str(e):
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        else: 
            experiment_id = ""
    
    return experiment_id

def get_mlflow_experiment_by_name(experiment_name: str) -> Any:
    """
    Retrieve the experiment id of the given experiment name.
    """
    experiment= mlflow.get_experiment_by_name(experiment_name)
    return experiment

def get_mlflow_experiment_by_id(experiment_id: str) -> Any:
    """
    Retrieve the experiment name of the given experiment id.
    """
    experiment = mlflow.get_experiment(experiment_id)
    return experiment

def delete_mlflow_experiment(experiment_id: str) -> None:
    """
    Delete the experiment with the given id.
    """
    mlflow.delete_experiment(experiment_id)