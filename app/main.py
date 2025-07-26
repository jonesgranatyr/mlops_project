import io
import logging
import os

import joblib
import mlflow
import pandas as pd
from flask import Flask, render_template, request
from sklearn.datasets import load_breast_cancer
from mlflow.tracking import MlflowClient

logger = logging.getLogger("app.main")


class ModelService:
    def __init__(self) -> None:
        self._load_artifacts()

    def _load_artifacts(self):
        """Load the registered model from MLflow Model Registry and related artifacts from its run."""
        
        # Load model from registry
        logger.info("Loading registered model from MLflow Model Registry")
        self.model = mlflow.keras.load_model("models:/model/latest")

        # Get run_id from model version metadata
        client = MlflowClient()
        run_id = client.get_registered_model("model").latest_versions[0].run_id

        # Load related artifacts
        logger.info(f"Loading artifacts from run {run_id}")
        artifacts_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="")

        imputer_path = os.path.join(artifacts_dir, "[features]_mean_imputer.joblib")
        self.features_imputer = joblib.load(imputer_path)
        scaler_path = os.path.join(artifacts_dir, "[features]_scaler.joblib")
        self.features_scaler = joblib.load(scaler_path)
        encoder_path = os.path.join(artifacts_dir, "[target]_one_hot_encoder.joblib")
        self.target_encoder = joblib.load(encoder_path)
        
        logger.info("Successfully loaded model and related artifacts")

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Make predictions using the full pipeline.

        Args:
            features: DataFrame containing the input features

        Returns:
            Series containing the predictions
        """
        # Apply transformations in sequence
        X_imputed = self.features_imputer.transform(features)
        X_scaled = self.features_scaler.transform(X_imputed)

        # Get model predictions
        y_pred = self.model.predict(X_scaled)

        # Decode predictions
        y_decoded = self.target_encoder.inverse_transform(y_pred)

        return pd.Series(y_decoded.ravel(), index=features.index)


def create_routes(app: Flask) -> None:
    """Create all routes for the application."""

    @app.route("/")
    def index() -> str:
        """Serve the HTML upload interface."""
        return render_template("index.html")

    @app.route("/upload", methods=["POST"])
    def upload() -> str:
        """Handle CSV file upload, validate features, and return predictions."""
        file = request.files["file"]
        if not file.filename.endswith(".csv"):
            return render_template("index.html", error="Please upload a CSV file")

        try:
            # Read CSV content
            content = file.read().decode("utf-8")
            features = pd.read_csv(io.StringIO(content))

            # Validate column names against breast cancer dataset
            expected_features = load_breast_cancer().feature_names
            missing_cols = [
                col for col in expected_features if col not in features.columns
            ]
            if missing_cols:
                return render_template(
                    "index.html",
                    error=f"Missing required columns: {', '.join(missing_cols)}",
                )
            features = features[expected_features]

            # Make predictions
            predictions = app.model_service.predict(features)

            # Format predictions for display
            result = pd.DataFrame(
                {"Sample": range(1, len(predictions) + 1), "Prediction": predictions}
            ).to_string(index=False)

            return render_template("index.html", predictions=result)

        except Exception as e:
            logger.error(
                f"Error processing file: {e}", exc_info=True
            )  # Added exc_info for better logging
            return render_template(
                "index.html",
                error=f"Error processing file: {str(e)}",  # Ensure e is string
            )


# Create and configure Flask app at module level
app = Flask(__name__)
app.model_service = ModelService()
create_routes(app)
logger.info("Application initialized with model service and routes")


def main() -> None:
    """Run the Flask development server."""
    app.run(host="0.0.0.0", port=5001)


if __name__ == "__main__":
    main()
