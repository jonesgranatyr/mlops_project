import json
import logging
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


logger = logging.getLogger("src.model_training.train_model")


def load_data() -> pd.DataFrame:
    """Load the feature-engineered training data.

    Returns:
        pd.DataFrame: A dataframe containing the training data.
    """
    train_path = "data/processed/train_processed.csv"
    logger.info(f"Loading feature data from {train_path}")
    train_data = pd.read_csv(train_path)
    return train_data


def load_params() -> dict[str, float | int]:
    """Load model hyperparameters for the train stage from params.yaml.

    Returns:
        dict[str, int | float]: dictionary containing model hyperparameters.
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["train"]


def prepare_data(
    train_data: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, OneHotEncoder]:
    """Prepare data for neural network training by separating features and target, and encoding labels.

    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Test dataset.

    Returns:
        tuple containing:
            pd.DataFrame: Training features
            np.ndarray: Encoded training labels
            OneHotEncoder: Fitted label encoder
    """
    # Separate features and target for train data
    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]

    # One-hot encode the target variable
    encoder = OneHotEncoder(sparse_output=False)
    y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1))

    return X_train, y_train_encoded, encoder


def create_model(
    input_shape: int, num_classes: int, params: dict[str, int | float]
) -> tf.keras.Model:
    """Create a Keras Dense Neural Network model.

    Args:
        input_shape (int): Number of input features.
        num_classes (int): Number of target classes.
        params (dict[str, int | float]): Model hyperparameters.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    model = Sequential(
        [
            Dense(
                params["hidden_layer_1_neurons"],
                activation="relu",
                input_shape=(input_shape,),
            ),
            Dropout(params["dropout_rate"]),
            Dense(
                params["hidden_layer_2_neurons"],
                activation="relu",
            ),
            Dropout(params["dropout_rate"]),
            Dense(num_classes, activation="softmax"),
        ]
    )

    optimizer = Adam(learning_rate=params["learning_rate"])

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def save_training_artifacts(model: tf.keras.Model, encoder: OneHotEncoder) -> None:
    """Save model artifacts to disk.

    Args:
        model (tf.keras.Model): Trained Keras model.
        encoder (OneHotEncoder): Fitted label encoder.
    """
    artifacts_dir = "artifacts"
    models_dir = "models"
    model_path = os.path.join(models_dir, "model.keras")
    encoder_path = os.path.join(artifacts_dir, "[target]_one_hot_encoder.joblib")

    # Save the model
    logger.info(f"Saving model to {model_path}")
    model.save(model_path)

    # Save the encoder for inference
    logger.info(f"Saving encoder to {encoder_path}")
    joblib.dump(encoder, encoder_path)


def train_model(train_data: pd.DataFrame, params: dict[str, int | float]) -> None:
    """Train a Keras model, logging metrics and artifacts with MLflow.

    Args:
        train_data (pd.DataFrame): Training dataset.
        params (dict[str, int | float]): Model hyperparameters.
    """
    # Set up MLflow experiment
    mlflow.set_experiment("ml_classification")

    # Set up Keras autolog
    mlflow.keras.autolog()

    # Setting MLflow if we are running a DVC experiment
    is_experiment = os.getenv("DVC_EXP_NAME") is not None
    extra_args = {}
    if is_experiment:
        runs = mlflow.search_runs(
            experiment_ids=[os.getenv("MLFLOW_EXPERIMENT_ID")],
            filter_string="tags.dvc_exp = 'True'",
            order_by=["start_time DESC"],
        )
        if runs.empty:
            with mlflow.start_run() as parent_run:
                mlflow.set_tag("dvc_exp", True)
                parent_run_id = parent_run.info.run_id
        else:
            parent_run_id = runs.iloc[0].run_id
        run_name = os.getenv("DVC_EXP_NAME")
        extra_args = {
            "parent_run_id": parent_run_id,
            "run_name": run_name,
            "nested": True,
        }

    with mlflow.start_run(**extra_args):
        # Log parameters to MLflow
        mlflow.log_params(params)

        tf.keras.utils.set_random_seed(params.pop("random_seed"))

        # Log preprocessing artifacts
        mlflow.log_artifact("artifacts/[features]_mean_imputer.joblib")
        mlflow.log_artifact("artifacts/[features]_scaler.joblib")

        # Prepare the data
        X_train, y_train, encoder = prepare_data(train_data)

        # Create the model
        model = create_model(
            input_shape=X_train.shape[1], num_classes=y_train.shape[1], params=params
        )

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        # Train the model with validation split
        logger.info("Training model...")
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=[early_stopping],
        )

        save_training_artifacts(model, encoder)

        # Log the encoder
        mlflow.log_artifact("artifacts/[target]_one_hot_encoder.joblib")

        # Save training metrics to a file
        metrics = {
            metric: float(history.history[metric][-1]) for metric in history.history
        }
        metrics_path = "metrics/training.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # # Log metrics to MLflow
        # mlflow.log_metrics(metrics)


def main() -> None:
    """Main function to orchestrate the model training process."""
    train_data = load_data()
    params = load_params()
    train_model(train_data, params)
    logger.info("Model training completed")


if __name__ == "__main__":
    main()
