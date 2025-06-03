import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Export YOLO model to ONNX format')
    
    # Required arguments
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input .pt model file')
    
    # Optional arguments
    parser.add_argument('--output', type=str, default=None,
                      help='Path to save the ONNX model (default: same directory as input with .onnx extension)')
    parser.add_argument('--imgsz', type=int, default=640,
                      help='Image size for ONNX export (default: 640)')
    parser.add_argument('--batch', type=int, default=1,
                      help='Batch size for ONNX export (default: 1)')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device to use for export (default: cpu)')
    
    return parser.parse_args()

def export_to_onnx(input_path: str, output_path: str = None, imgsz: int = 640, 
                  batch: int = 1, device: str = 'cpu') -> str:
    """
    Export YOLO model to ONNX format.
    
    Args:
        input_path (str): Path to input .pt model
        output_path (str, optional): Path to save ONNX model
        imgsz (int, optional): Image size. Defaults to 640.
        batch (int, optional): Batch size. Defaults to 1.
        device (str, optional): Device to use. Defaults to 'cpu'.
    
    Returns:
        str: Path to exported ONNX model
    """
    try:
        # Convert paths to Path objects
        input_path = Path(input_path)
        
        # Validate input path
        if not input_path.exists():
            raise FileNotFoundError(f"Input model not found: {input_path}")
        if input_path.suffix != '.pt':
            raise ValueError(f"Input file must be a .pt file, got: {input_path.suffix}")
        
        # Set default output path if not provided
        if output_path is None:
            output_path = input_path.with_suffix('.onnx')
        else:
            output_path = Path(output_path)
            # Ensure output has .onnx extension
            if output_path.suffix != '.onnx':
                output_path = output_path.with_suffix('.onnx')
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load YOLO model
        print(f"Loading YOLO model from: {input_path}")
        model = YOLO(str(input_path))
        
        # Export to ONNX
        print(f"Exporting to ONNX format...")
        model.export(
            format='onnx',
            imgsz=imgsz,
            batch=batch,
            device=device,
            simplify=True,  # Simplify ONNX model
            dynamic=True,  # Enable dynamic axes for batch size and image size
            nms=True,  # Adds Non-Maximum Suppression (NMS), for accurate and efficient detection post-processing.
            #half=True,  # FP16 export for GPU
        )
        
        # The exported model will be in the same directory as the input with _saved_model suffix
        onnx_path = input_path.parent / (input_path.stem + '.onnx')
        
        # Move to desired output location if specified
        if output_path != onnx_path:
            from shutil import move
            move(str(onnx_path), str(output_path))
            
        print(f"✅ Model exported successfully to: {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"❌ Error exporting model: {e}")
        raise

if __name__ == "__main__":
    args = parse_args()
    export_to_onnx(
        input_path=args.input,
        output_path=args.output,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )