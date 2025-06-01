import mlflow
import os
import sys
from pathlib import Path

# Fix import path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.config import Config

'''This script uploads and registers local model weights to MLflow.'''

# === CONFIG ===
config = Config()
MLFLOW_TRACKING_URI = config.MLFLOW_TRACKING_URI
EXPERIMENT_NAME = "Manual-Upload"

# Use paths from Config
LOCAL_MODEL_WEIGHTS = config.get_model_paths()

# Use MLflow artifact config from Config
MLFLOW_MODEL_ARTIFACTS = config.MLFLOW_MODEL_ARTIFACTS

# === SETUP ===
if not config.MLFLOW_ENABLED:
    raise RuntimeError("MLflow is not enabled in configuration")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

# === UPLOAD AND REGISTER EACH MODEL ===
run_ids = {}

for model_key, file_path in LOCAL_MODEL_WEIGHTS.items():
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"⚠️ Skipping {model_key}: file not found at {file_path}")
        continue

    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        # Use artifact path from config
        artifact_path = MLFLOW_MODEL_ARTIFACTS[model_key]["artifact_path"].split("/")[0]  # Get base path (e.g., "model")
        
        mlflow.log_artifact(str(file_path), artifact_path=artifact_path)
        
        # Register the model with version from config
        model_name = model_key
        version = MLFLOW_MODEL_ARTIFACTS[model_key].get("version", "1")
        model_uri = f"runs:/{run_id}/{artifact_path}/{file_path.name}"
        
        try:
            result = mlflow.register_model(model_uri, model_name)
            print(f"✅ Registered {model_key} as MLflow Model: {model_name} (version {result.version})")
            run_ids[model_key] = {
                "run_id": run_id,
                "artifact_path": artifact_path,
                "model_name": model_name,
                "model_version": result.version
            }
        except Exception as e:
            print(f"⚠️ Failed to register {model_key}: {e}")
            run_ids[model_key] = {
                "run_id": run_id,
                "artifact_path": artifact_path,
                "model_name": model_name,
                "model_version": None,
                "error": str(e)
            }

# === PRINT SUMMARY ===
print("\n=== Model Upload & Registration Summary ===")
for model_key, info in run_ids.items():
    artifact_path = MLFLOW_MODEL_ARTIFACTS[model_key]["artifact_path"]
    print(
        f"{model_key} -> run_id='{info['run_id']}', "
        f"artifact_path='{artifact_path}', "
        f"model_name='{info['model_name']}', "
        f"model_version='{info['model_version']}'"
    )
