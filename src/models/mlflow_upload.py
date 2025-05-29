import mlflow
import os
from pathlib import Path

# === CONFIG ===
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Manual-Upload")
MODEL_FILES = {
    "yolo_text_detect": "./models/text_detection/weights/best.pt",
    "yolo_text_detect_v2": "./models/text_detection/weights/bestv2.pt",
    "yolo_corner_detect": "./models/corner_detection/weights/29_03_25-YOLOv11n-Corner-best_metrics.pt"
}

# === SETUP ===
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

# === UPLOAD AND REGISTER EACH MODEL ===
run_ids = {}

for model_key, file_path in MODEL_FILES.items():
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"⚠️ Skipping {model_key}: file not found at {file_path}")
        continue

    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        artifact_path = "model"
        mlflow.log_artifact(str(file_path), artifact_path=artifact_path)
        # Register the model
        model_name = model_key
        # Use mlflow.pyfunc to register a generic model artifact
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
    print(
        f"{model_key} -> run_id='{info['run_id']}', "
        f"artifact_path='{info['artifact_path']}/{model_key}.pt', "
        f"model_name='{info['model_name']}', "
        f"model_version='{info['model_version']}'"
    )
