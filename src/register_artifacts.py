import logging
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd


logger = logging.getLogger("src.register_artifacts")
    
client = MlflowClient()


def get_best_run(experiment_id: str, parent_run_id: str) -> pd.Series:
    """Get the best child run based on test accuracy for a given parent run.
    
    Args:
        client: MLflow client instance
        parent_run_id: ID of the parent run
        
    Returns:
        The best run as a pandas Series
    """
    # Get all child runs for the parent
    child_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        order_by=["metrics.test_accuracy DESC"],
        max_results=1000
    )       
    # Return the run with highest test accuracy
    return child_runs[0]

def register_model() -> None:
    """Register the model that was logged during training."""

    logger.info("Registering model from latest MLflow run")

    # Get the experiment ID for the 'ml_classification' experiment
    experiment_id = client.get_experiment_by_name("ml_classification").experiment_id

    # Get the latest run from the experiment
    latest_run = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )[0]
    
    # Check if the latest run has a parent run
    run_id = latest_run.info.run_id
    parent_run_id = latest_run.data.tags.get('mlflow.parentRunId')
    
    if parent_run_id:
        logger.info(f"Latest run has parent run ID: {parent_run_id}")
        best_run = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
            order_by=["metrics.test_accuracy DESC"],
            max_results=1
        )[0]
        run_id = best_run.info.run_id
        logger.info(f"Using best run {run_id} with test_accuracy: {best_run.data.metrics['test_accuracy']}")

    # Register the model from the run
    logger.info("Registering model")
    try:
        client.create_registered_model("model")
    except mlflow.exceptions.MlflowException:
        logger.debug("Model already exists")

    model_uri = f"runs:/{run_id}/model"
    client.create_model_version(
        name="model",
        source=model_uri,
        run_id=run_id
    )
    logger.info("Registered model successfully")


def main() -> None:
    """Main function to orchestrate the model registration process."""
    register_model()
    logger.info("Model registration completed")


if __name__ == "__main__":
    main()