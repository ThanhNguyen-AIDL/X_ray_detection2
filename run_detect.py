
#!/usr/bin/env python3
"""
Launcher for YOLOv5 detect.py with PyTorch 2.7+ compatibility
"""

import os
import sys
import runpy

# # Add the compatibility layer
sys.path.insert(0, os.path.abspath('yolov5-bk'))
import fix_pytorch

# Get all command line arguments
args = sys.argv[1:]

# Change directory to YOLOv5
yolov5_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolov5'))
os.chdir(yolov5_path)
# Run detect.py with the given arguments
sys.argv = ['detect.py'] + args
runpy.run_path('detect.py')
