"""
Simple test script for YOLOv5 model loading with PyTorch 2.7
"""

import os
import sys
import torch
from pathlib import Path
import traceback

# Add YOLOv5 directory to path
yolov5_dir = Path('yolov5-bk').absolute()
if yolov5_dir.exists():
    sys.path.insert(0, str(yolov5_dir))
    os.chdir(str(yolov5_dir))
    print(f"Changed working directory to: {os.getcwd()}")
else:
    print(f"YOLOv5 directory not found at {yolov5_dir}")

# Try to load the model directly
try:
    from models.experimental import attempt_load
    print("Successfully imported YOLOv5 modules")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Find the model file
    weights_path = None
    possible_paths = [
        "runs/train/exp/weights/best.pt",
        "../runs/train/exp/weights/best.pt",
        "../../runs/train/exp/weights/best.pt",
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            weights_path = path
            break
    
    if not weights_path:
        print("Model file not found. Please specify the correct path.")
        sys.exit(1)
    
    print(f"Loading model from: {weights_path}")
    model = attempt_load(weights_path, device="cpu")
    print(f"Model loaded successfully! Type: {type(model)}")
    
    if hasattr(model, 'names'):
        print(f"Model classes: {model.names}")
    
    if hasattr(model, 'stride'):
        print(f"Model stride: {model.stride}")
    
    print("Test successful!")
    sys.exit(0)
    
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    sys.exit(1)
