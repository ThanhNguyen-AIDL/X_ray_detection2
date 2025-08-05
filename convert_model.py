"""
Convert YOLOv5 model to be compatible with PyTorch 2.7+
This script addresses the PosixPath and weights_only issues
"""

import os
import sys
import torch
import pickle
from pathlib import Path
import argparse

def find_model_path():
    """Find YOLOv5 model path"""
    possible_paths = [
        "yolov5/runs/train/exp/weights/best.pt",
        "runs/train/exp/weights/best.pt",
        "../runs/train/exp/weights/best.pt",
        "yolov5/runs/train/exp/weights/last.pt",
        "runs/train/exp/weights/last.pt"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return path
    
    return None

def convert_model(model_path, output_path=None):
    """Convert YOLOv5 model to be compatible with PyTorch 2.7+"""
    if output_path is None:
        # Create new filename with _converted suffix
        p = Path(model_path)
        output_path = str(p.with_stem(f"{p.stem}_converted"))
    
    print(f"Loading model from: {model_path}")
    
    try:
        # Load the model using a custom unpickler
        class PathFixingUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'pathlib' and name == 'PosixPath':
                    from pathlib import WindowsPath
                    return WindowsPath
                return super().find_class(module, name)
        
        # Load the model file directly with our custom unpickler
        with open(model_path, 'rb') as f:
            model_data = PathFixingUnpickler(f).load()
            
        print(f"Model loaded successfully")
        print(f"Model keys: {list(model_data.keys() if isinstance(model_data, dict) else [])}")
        
        # Save the model with the same structure but as a compatible file
        print(f"Saving converted model to: {output_path}")
        torch.save(model_data, output_path)
        
        print(f"Model successfully converted and saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error converting model: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLOv5 model for PyTorch 2.7+ compatibility")
    parser.add_argument('--input', type=str, help='Path to input model file')
    parser.add_argument('--output', type=str, help='Path to output model file')
    args = parser.parse_args()
    
    model_path = args.input if args.input else find_model_path()
    
    if not model_path:
        print("Error: Could not find model path. Please specify with --input")
        sys.exit(1)
        
    converted_path = convert_model(model_path, args.output)
    
    if converted_path:
        print("\nTo use the converted model, update the model path in your code to:")
        print(f"  {converted_path}")
        sys.exit(0)
    else:
        print("Model conversion failed.")
        sys.exit(1)
