import os
import torch
import mlflow
from ultralytics import YOLO
from pathlib import Path
from src.config import get_config

'''
Train YOLO Corner Detection Model
This script trains a YOLO model for corner detection using the Ultralytics YOLO library.
It supports downloading model weights from MLflow if available, otherwise falls back to local weights.
It logs training parameters and metrics to MLflow.
It also ensures the necessary directories exist and sets up the MLflow tracking URI.
'''

# === CONFIG ===
config = get_config()
MLFLOW_TRACKING_URI = config.MLFLOW_TRACKING_URI
MLFLOW_ENABLED = config.MLFLOW_ENABLED
MODEL_KEY = "yolo_text"
EXPERIMENT_NAME = "YOLO-Text-Detection"

ARTIFACT_INFO = config.MLFLOW_MODEL_ARTIFACTS.get(MODEL_KEY, {})
RUN_ID = ARTIFACT_INFO.get("run_id")
MODEL_VERSION = ARTIFACT_INFO.get("version", "1")
ARTIFACT_PATH = ARTIFACT_INFO.get("artifact_path")

DATA_PATH = os.path.join(str(config.YOLO_TEXT_MODEL_DATASET), "main", "data.yaml")
EPOCHS = 100
IMGSZ = 640

LOCAL_DIR = "./.mlflow_downloads"
os.makedirs(LOCAL_DIR, exist_ok=True)

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id
    
# === Load YOLO Weights ===
model_path = None

# Try downloading from MLflow if enabled
if MLFLOW_ENABLED:
    try:
        if RUN_ID:
            # Download from specific run
            print(f"Downloading weights from MLflow run: {RUN_ID}")
            model_path = mlflow.artifacts.download_artifacts(
                run_id=RUN_ID,
                artifact_path=ARTIFACT_PATH.rstrip("/"),
                dst_path=LOCAL_DIR
            )
        else:
            # Get run ID from Model Registry version
            client = mlflow.tracking.MlflowClient()
            try:
                model_version = client.get_model_version(MODEL_KEY, MODEL_VERSION)
                version_run_id = model_version.run_id
                print(f"Found run ID {version_run_id} for {MODEL_KEY} version {MODEL_VERSION}")
                
                # Download using the run ID and artifact path
                model_path = mlflow.artifacts.download_artifacts(
                    run_id=version_run_id,
                    artifact_path=ARTIFACT_PATH.rstrip("/"),
                    dst_path=LOCAL_DIR
                )
                print(f"Downloaded model path: {model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to get model version: {e}")
        
        model_path = Path(model_path)
        if model_path.is_dir():
            pt_files = list(model_path.glob("*.pt"))
            if pt_files:
                model_path = pt_files[0]
            else:
                raise FileNotFoundError(f"No .pt file found in directory: {model_path}")
        elif not model_path.suffix == '.pt':
            raise ValueError(f"Downloaded file is not a .pt file: {model_path}")
            
        print(f"✅ Downloaded weights from MLflow: {model_path}")
    except Exception as e:
        print(f"⚠️ Could not download weights from MLflow: {e}")
        model_path = None

# Try local weights if MLflow download fails
if not model_path or not Path(model_path).exists():
    try:
        local_paths = config.get_model_paths()
        model_path = local_paths.get(MODEL_KEY)
        if model_path and Path(model_path).exists():
            print(f"Using local weights: {model_path}")
        else:
            print("⚠️ No local weights found")
            model_path = None
    except Exception as e:
        print(f"⚠️ Error accessing local weights: {e}")
        model_path = None

# Select device
device = "cuda" if torch.cuda.is_available() and not config.FORCE_CPU else "cpu"
print(f"Using device: {device}")

# Start MLflow run for training
with mlflow.start_run(experiment_id=experiment_id) as run:
    print(f"Started MLflow run: {run.info.run_id}")
    
    # Initialize YOLO model
    if model_path:
        print(f"Loading pretrained weights from: {model_path}")
        model = YOLO(str(model_path))
    else:
        print("Starting training from scratch with YOLO base model")
        model = YOLO()  # Initialize new model
    
    model.to(device)
    
    # Log parameters
    mlflow.log_params({
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "device": device,
        "base_model": str(model_path) if model_path else "scratch"
    })

    # Train
    results = model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        device=device
    )

    # Log metrics
    metrics = {
        "final_precision": results.results_dict.get("metrics/precision(B)", 0),
        "final_recall": results.results_dict.get("metrics/recall(B)", 0),
        "final_map": results.results_dict.get("metrics/mAP50(B)", 0)
    }
    mlflow.log_metrics(metrics)

    # Log model artifacts
    mlflow.log_artifact(results.save_dir)
    
    print(f"✅ Training completed. Results saved to MLflow run: {run.info.run_id}")