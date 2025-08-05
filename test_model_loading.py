#!/usr/bin/env python3
"""
Test script to verify that the YOLOv5 model loads correctly with PyTorch 2.7
"""

import os
import sys
import torch
from pathlib import Path

# Find YOLOv5 directory
yolov5_path = os.path.abspath("yolov5-bk")
if not os.path.exists(yolov5_path):
    yolov5_path = os.path.abspath(".")

# Add YOLOv5 root directory to the beginning of the path
sys.path.insert(0, yolov5_path)
print(f"Added to Python path: {yolov5_path}")

# Change directory to YOLOv5 for relative paths
os.chdir(yolov5_path)
print(f"Changed working directory to: {os.getcwd()}")

# Import YOLOv5 modules 
try:
    from models.experimental import attempt_load
    print("Successfully imported YOLOv5 modules")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current sys.path: {sys.path}")
    print("Attempting alternative import method...")
    
    # Try direct import
    import models.experimental
    print("Direct import successful")

def test_model_loading():
    """Test loading a YOLOv5 model"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Try to load a model
    try:
        # Try several possible model paths
        possible_paths = [
            "runs/train/exp/weights/best.pt",
            "../runs/train/exp/weights/best.pt",
            "yolov5/runs/train/exp/weights/best.pt"
        ]
        
        weights_path = None
        for path in possible_paths:
            if os.path.exists(path):
                weights_path = path
                break
                
        if not weights_path:
            print(f"Model not found in any of the expected paths. Please provide the correct path.")
            return False
            
        print(f"Loading model from {weights_path}")
        model = attempt_load(weights_path, device="cpu")
        print(f"Model loaded successfully: {type(model)}")
        print(f"Model stride: {model.stride}")
        print(f"Model names: {model.names}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    print(f"Test result: {'SUCCESS' if success else 'FAILED'}")
