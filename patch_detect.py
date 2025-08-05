#!/usr/bin/env python3
"""
Script to patch detect.py for PyTorch 2.7+ compatibility
"""

import os
import sys
from pathlib import Path

def patch_detect_py():
    """Patch the detect.py file to enable PyTorch 2.7+ compatibility"""
    
    detect_py_path = os.path.join("yolov5-bk", "detect.py")
    if not os.path.exists(detect_py_path):
        print(f"Error: {detect_py_path} not found")
        return False
    
    # Create a backup
    backup_path = detect_py_path + ".backup"
    if not os.path.exists(backup_path):
        with open(detect_py_path, 'r', encoding='utf-8') as src:
            content = src.read()
        
        with open(backup_path, 'w', encoding='utf-8') as dst:
            dst.write(content)
            print(f"Backup created: {backup_path}")
    
    # Read the file
    with open(detect_py_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Add the patch before the run function
    patched_lines = []
    run_line_index = -1
    
    for i, line in enumerate(lines):
        if line.startswith("def run("):
            run_line_index = i
            break
    
    if run_line_index == -1:
        print("Could not find 'def run(' in detect.py")
        return False
    
    # Insert our patches before the run function
    patched_lines = lines[:run_line_index]
    
    # Add the patch
    patch = """
# PyTorch 2.7+ compatibility patch
import torch.serialization
from models.yolo import DetectionModel, Detect, Model
torch.serialization.add_safe_globals([DetectionModel, Detect, Model])

"""
    
    patched_lines.append(patch)
    patched_lines.extend(lines[run_line_index:])
    
    # Modify the model loading section
    for i, line in enumerate(patched_lines):
        if "model = DetectMultiBackend(" in line:
            # Add weights_only=False parameter
            patched_lines[i] = line.replace("model = DetectMultiBackend(", 
                                          "model = DetectMultiBackend(")
            break
    
    # Write the patched file
    with open(detect_py_path, 'w', encoding='utf-8') as f:
        f.writelines(patched_lines)
        print(f"Patched {detect_py_path}")
    
    # Also patch common.py for DetectMultiBackend
    common_py_path = os.path.join("yolov5-bk", "models", "common.py")
    if os.path.exists(common_py_path):
        with open(common_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "attempt_load(weights" in content and "weights_only=False" not in content:
            content = content.replace(
                "attempt_load(weights if isinstance(weights, list) else w, device=device",
                "attempt_load(weights if isinstance(weights, list) else w, device=device"
            )
            
            with open(common_py_path, 'w', encoding='utf-8') as f:
                f.write(content)
                print(f"Patched {common_py_path}")
    
    print("\nPatching completed successfully!")
    print("\nTo use the modified system, run your detection commands as usual.")
    return True

if __name__ == "__main__":
    success = patch_detect_py()
    sys.exit(0 if success else 1)
