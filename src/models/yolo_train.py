import os
import torch
import mlflow
import argparse
from ultralytics import YOLO
from pathlib import Path

# Fix import path
import sys
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.config import get_config

'''
Train YOLO Corner Detection Model
This script trains a YOLO model for corner detection using the Ultralytics YOLO library.
It supports downloading model weights from MLflow if available, otherwise falls back to local weights.
It logs training parameters and metrics to MLflow.
It also ensures the necessary directories exist and sets up the MLflow tracking URI.
'''

from ultralytics import settings

# Update a setting
settings.update({"mlflow": True})

# Add argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO Corner Detection Model')
    
    # MLflow related args for model loading
    parser.add_argument('--model-key', type=str, default="yolo_corner_detect",
                      help='Model key in MLflow registry (also used as experiment name)')
    parser.add_argument('--run-id', type=str, default=None,
                      help='MLflow run ID to load weights from')
    parser.add_argument('--load-version', type=str, default="0",
                      help='Model version in MLflow registry (use "latest" or "0" for latest version)')
    parser.add_argument('--load-alias', type=str, default=None,
                      help='Alias to load model weights from (e.g., "production", "staging")')
    
    # MLflow related args for model saving
    parser.add_argument('--save-alias', type=str, default=None,
                      help='Alias to set for the newly trained model (e.g., "production", "staging")')
    parser.add_argument('--save-tags', type=str, default=None,
                      help='Tags to set for the new model version in format "key1=value1,key2=value2"')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=1,
                      help='Number of epochs to train')
    parser.add_argument('--img-size', type=int, default=640,
                      help='Image size for training')
    parser.add_argument('--data-path', type=str, default=None,
                      help='Path to data.yaml file')
    
    # Other args
    parser.add_argument('--local-dir', type=str, default=None,
                      help='Local directory for saving training results (default: ./.temp/training/{model_key})')
    parser.add_argument('--force-cpu', action='store_true',
                      help='Force using CPU even if CUDA is available')
    
    return parser.parse_args()

# Replace the config section with
def get_config_with_args(args):
    config = get_config()
    
    # Override config with command line args
    MODEL_KEY = args.model_key
    EXPERIMENT_NAME = MODEL_KEY  # Use model key as experiment name
    ARTIFACT_INFO = config.MLFLOW_MODEL_ARTIFACTS.get(MODEL_KEY, {})
    RUN_ID = args.run_id or ARTIFACT_INFO.get("run_id")
    ARTIFACT_PATH = ARTIFACT_INFO.get("artifact_path")
    EPOCHS = args.epochs
    IMGSZ = args.img_size
    FORCE_CPU = args.force_cpu
    
    # Handle "latest" version flag or 0
    MODEL_VERSION = args.load_version
    if MODEL_VERSION in ["latest", "0"]:
        try:
            client = mlflow.tracking.MlflowClient()
            # Get all versions and find the latest one
            versions = client.search_model_versions(f"name = '{MODEL_KEY}'")
            if versions:
                # Sort by version number and get the latest
                latest_version = max(versions, key=lambda x: int(x.version))
                MODEL_VERSION = str(latest_version.version)
                print(f"Using latest model version: {MODEL_VERSION}")
            else:
                print("⚠️ No versions found, falling back to default")
                MODEL_VERSION = ARTIFACT_INFO.get("version", "1")
        except Exception as e:
            print(f"⚠️ Could not get latest version, falling back to default: {e}")
            MODEL_VERSION = ARTIFACT_INFO.get("version", "1")
    else:
        MODEL_VERSION = MODEL_VERSION or ARTIFACT_INFO.get("version", "1")
        
    MODEL_ALIAS = args.save_alias 
    
    # Parse tags if provided
    MODEL_TAGS = {}  # Initialize outside if block
    if args.save_tags:
        try:
            tag_pairs = args.save_tags.split(',')
            for pair in tag_pairs:
                key, value = pair.split('=')
                MODEL_TAGS[key.strip()] = value.strip()
        except Exception as e:
            print(f"⚠️ Error parsing tags: {e}")
    
    # Use provided data path or default from config
    if args.data_path:
        DATA_PATH = args.data_path
    else:
        DATA_PATH = os.path.join(str(config.YOLO_CORNER_MODEL_DATASET), "main", "data.yaml")
    
    # Set up local directory
    LOCAL_DIR = args.local_dir if args.local_dir else f"./.temp/training/{MODEL_KEY}"
    
    # Clear the directory if it exists
    local_dir_path = Path(LOCAL_DIR)
    if local_dir_path.exists():
        import shutil
        shutil.rmtree(local_dir_path)
    local_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Cleared and recreated local directory: {LOCAL_DIR}")
    
    return {
        'MLFLOW_TRACKING_URI': config.MLFLOW_TRACKING_URI,
        'MLFLOW_ENABLED': config.MLFLOW_ENABLED,
        'MODEL_KEY': MODEL_KEY,
        'EXPERIMENT_NAME': EXPERIMENT_NAME,
        'RUN_ID': RUN_ID,
        'MODEL_VERSION': MODEL_VERSION,
        'ARTIFACT_PATH': ARTIFACT_PATH,
        'DATA_PATH': DATA_PATH,
        'EPOCHS': EPOCHS,
        'IMGSZ': IMGSZ,
        'LOCAL_DIR': LOCAL_DIR,
        'MODEL_ALIAS': MODEL_ALIAS,
        'MODEL_TAGS': MODEL_TAGS,  # Now MODEL_TAGS is always defined
        'FORCE_CPU': FORCE_CPU,
        'LOAD_ALIAS': args.load_alias,  # For loading weights
        'SAVE_ALIAS': args.save_alias,  # For saving new model
        'SAVE_TAGS': MODEL_TAGS,  # For saving new model
    }

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Get configuration
    cfg = get_config_with_args(args)
    config = get_config()
    
    # Create local directory
    os.makedirs(cfg['LOCAL_DIR'], exist_ok=True)

    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri(cfg['MLFLOW_TRACKING_URI'])

    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(cfg['EXPERIMENT_NAME'])
    if experiment is None:
        experiment_id = mlflow.create_experiment(cfg['EXPERIMENT_NAME'])
    else:
        # Check if experiment is deleted
        if experiment.lifecycle_stage == 'deleted':
            # Create new experiment with timestamp to avoid conflicts
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_name = f"{cfg['EXPERIMENT_NAME']}_{timestamp}"
            print(f"Previous experiment was deleted, creating new one: {new_name}")
            experiment_id = mlflow.create_experiment(new_name)
        else:
            experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_id=experiment_id)

    # === Load YOLO Weights ===
    model_path = None

    # Try downloading from MLflow if enabled
    if cfg['MLFLOW_ENABLED']:
        try:
            download_path = None
            
            # 1. Try Run ID first if provided
            if cfg['RUN_ID']:
                print(f"Attempting download from run ID: {cfg['RUN_ID']}")
                try:
                    download_path = mlflow.artifacts.download_artifacts(
                        run_id=cfg['RUN_ID'],
                        artifact_path=cfg['ARTIFACT_PATH'].rstrip("/"),
                        dst_path=cfg['LOCAL_DIR']
                    )
                except Exception as e:
                    print(f"⚠️ Failed to download from run ID: {e}")
            
            # 2. Try alias if run ID failed and alias is provided
            if not download_path and cfg['LOAD_ALIAS']:
                print(f"Attempting download using alias: {cfg['LOAD_ALIAS']}")
                try:
                    client = mlflow.tracking.MlflowClient()
                    alias_version = client.get_model_version_by_alias(cfg['MODEL_KEY'], cfg['LOAD_ALIAS'])
                    if alias_version:
                        version_run_id = alias_version.run_id
                        print(f"Found run ID {version_run_id} for alias '{cfg['LOAD_ALIAS']}'")
                        download_path = mlflow.artifacts.download_artifacts(
                            run_id=version_run_id,
                            artifact_path=cfg['ARTIFACT_PATH'].rstrip("/"),
                            dst_path=cfg['LOCAL_DIR']
                        )
                except Exception as e:
                    print(f"⚠️ Failed to download using alias: {e}")
            
            # 3. Try model version if both run ID and alias failed
            if not download_path:
                print(f"Attempting download from version: {cfg['MODEL_VERSION']}")
                try:
                    client = mlflow.tracking.MlflowClient()
                    model_version = client.get_model_version(cfg['MODEL_KEY'], cfg['MODEL_VERSION'])
                    version_run_id = model_version.run_id
                    print(f"Found run ID {version_run_id} for version {cfg['MODEL_VERSION']}")
                    download_path = mlflow.artifacts.download_artifacts(
                        run_id=version_run_id,
                        artifact_path=cfg['ARTIFACT_PATH'].rstrip("/"),
                        dst_path=cfg['LOCAL_DIR']
                    )
                except Exception as e:
                    print(f"⚠️ Failed to download from version: {e}")
            
            if download_path:
                print(f"✅ Successfully downloaded to: {download_path}")
                download_path = Path(download_path)
                # Check possible weight locations
                weight_locations = [
                    download_path,  # Direct path
                    download_path / "weights",  # weights subdirectory
                    download_path.parent / "weights"  # parent/weights directory
                ]
                
                for location in weight_locations:
                    if location.is_dir():
                        pt_files = list(location.glob("*.pt"))
                        if pt_files:
                            # Try to find best.pt first
                            best_weights = next((f for f in pt_files if f.name == "best.pt"), None)
                            if best_weights:
                                model_path = best_weights
                            else:
                                # Fall back to first file alphabetically
                                model_path = sorted(pt_files)[0]
                            break
                
                if model_path is None:
                    raise FileNotFoundError(f"No .pt file found in any of: {[str(p) for p in weight_locations]}")
                elif not model_path.suffix == '.pt':
                    raise ValueError(f"Found file is not a .pt file: {model_path}")
                
                print(f"✅ Found weights at: {model_path}")

        except Exception as e:
            print(f"⚠️ All MLflow download attempts failed: {e}")
            model_path = None

    # Try local weights if MLflow download fails
    if not model_path or not Path(model_path).exists():
        try:
            local_paths = config.get_train_model_paths()
            model_path = local_paths.get(cfg['MODEL_KEY'])
            if model_path and Path(model_path).exists():
                print(f"Using local weights: {model_path}")
            else:
                print("⚠️ No local weights found")
                model_path = None
        except Exception as e:
            print(f"⚠️ Error accessing local weights: {e}")
            model_path = None

    # Select device
    device = "cuda" if torch.cuda.is_available() and not cfg['FORCE_CPU'] else "cpu"
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

        # Train with MLflow logging enabled
        results = model.train(
            data=cfg['DATA_PATH'],
            epochs=cfg['EPOCHS'],
            imgsz=cfg['IMGSZ'],
            device=device,
            pretrained=True if not model_path else False,
            cache='disk',
        )
        
        # Get best model path
        best_model_path = str(Path(results.save_dir) / "weights" / "best.pt")
        if Path(best_model_path).exists():
            # Register the model using the YOLO-logged artifact
            model_uri = f"runs:/{run.info.run_id}/train/weights/best.pt"
            try:
                registered_model = mlflow.register_model(
                    model_uri=model_uri,
                    name=cfg['MODEL_KEY']
                )
                print(f"✅ Model registered successfully as: {cfg['MODEL_KEY']} version {registered_model.version}")
                
                # Set alias if provided
                if cfg['SAVE_ALIAS']:
                    client = mlflow.tracking.MlflowClient()
                    client.set_registered_model_alias(
                        name=cfg['MODEL_KEY'],
                        alias=cfg['SAVE_ALIAS'],
                        version=registered_model.version
                    )
                    print(f"✅ Set '{cfg['SAVE_ALIAS']}' alias for version {registered_model.version}")
                
                # Set tags if provided
                if cfg['SAVE_TAGS']:
                    client = mlflow.tracking.MlflowClient()
                    for key, value in cfg['SAVE_TAGS'].items():
                        client.set_model_version_tag(
                            name=cfg['MODEL_KEY'],
                            version=registered_model.version,
                            key=key,
                            value=value
                        )
                    print(f"✅ Set tags for version {registered_model.version}: {cfg['SAVE_TAGS']}")
                
                # Update config with new version info
                if hasattr(config, 'update_model_artifact_info'):
                    config.update_model_artifact_info(
                        cfg['MODEL_KEY'],
                        run_id=run.info.run_id,
                        version=str(registered_model.version),
                        artifact_path="train/weights"
                    )
                    
            except Exception as e:
                print(f"⚠️ Failed to register model: {e}")
        else:
            print(f"⚠️ Best model weights not found at: {best_model_path}")

        print(f"✅ Training completed. Results saved to MLflow run: {run.info.run_id}")
