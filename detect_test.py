#!/usr/bin/env python3
"""
YOLOv5 Detection Test Script
Standalone script to run object detection on medical X-ray images
"""

import os
import sys
import subprocess
import argparse
import glob
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from IPython.display import Image, display
import pandas as pd
import numpy as np

# Configuration
YOLOV5_DIR = '/yolov5-bk'
DATA_DIR = 'd:/AITest/AIXRAY/data'
WEIGHTS_PATH = './runs/train/exp/weights/best.pt'
CONFIG_PATH = 'data/vinbigdata.yaml'

# Medical condition classes
CLASSES = [
    'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 
    'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass', 
    'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 
    'Pulmonary fibrosis'
]

def setup_environment():
    """Setup the working environment and check dependencies"""
    print("Setting up environment...")
    
    # Change to YOLOv5 directory
    os.chdir(YOLOV5_DIR)
    print(f"Working directory: {os.getcwd()}")
    
    # Check if weights exist
    if not os.path.exists(WEIGHTS_PATH):
        print(f"⚠️  Warning: Weights file not found at {WEIGHTS_PATH}")
        print("You need to train the model first or use pre-trained weights")
        return False
    
    # Check if config exists
    if not os.path.exists(CONFIG_PATH):
        print(f"⚠️  Warning: Config file not found at {CONFIG_PATH}")
        return False
    
    print("✅ Environment setup complete")
    return True

def run_detection(source_path, conf_threshold=0.25, img_size=640, save_txt=True, save_conf=True):
    """
    Run YOLOv5 detection on specified source
    
    Args:
        source_path (str): Path to image, directory, or video
        conf_threshold (float): Confidence threshold for detections
        img_size (int): Input image size
        save_txt (bool): Save results as txt files
        save_conf (bool): Save confidence scores in txt files
    
    Returns:
        tuple: (success, output_dir)
    """
    print(f"Running detection on: {source_path}")
    
    detect_cmd = [
        sys.executable, 'detect.py',
        '--weights', WEIGHTS_PATH,
        '--img', str(img_size),
        '--source', source_path,
        '--conf', str(conf_threshold),
        '--project', 'runs/detect',
        '--name', 'test_results',
        '--exist-ok'
    ]
    
    if save_txt:
        detect_cmd.append('--save-txt')
    if save_conf:
        detect_cmd.append('--save-conf')
    
    print(f"Command: {' '.join(detect_cmd)}")
    
    try:
        result = subprocess.run(detect_cmd, capture_output=True, text=True, check=True)
        print("✅ Detection completed successfully!")
        print(result.stdout)
        return True, 'runs/detect/test_results'
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running detection: {e}")
        print(f"stderr: {e.stderr}")
        return False, None
    except FileNotFoundError:
        print(f"❌ Source not found: {source_path}")
        return False, None

def display_results(results_dir):
    """Display detection results"""
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    print(f"\n📁 Results in: {results_dir}")
    
    # List all files in results directory
    print("\nFiles in detection directory:")
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path):
            print(f"  📁 {item}/")
            for subitem in os.listdir(item_path):
                print(f"    📄 {subitem}")
        else:
            print(f"  📄 {item}")
    
    # Find and display images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    found_images = []
    
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                found_images.append(os.path.join(root, file))
    
    if found_images:
        print(f"\n🖼️  Found {len(found_images)} result image(s):")
        for img_path in found_images:
            print(f"  📷 {img_path}")
            
            # Display image using matplotlib
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.figure(figsize=(12, 8))
                    plt.imshow(img_rgb)
                    plt.title(f"Detection Results: {os.path.basename(img_path)}")
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show()
                    break  # Show first image only
            except Exception as e:
                print(f"Could not display {img_path}: {e}")
    
    # Check for label files
    labels_dir = os.path.join(results_dir, 'labels')
    if os.path.exists(labels_dir):
        print(f"\n🏷️  Label files found in {labels_dir}:")
        label_files = os.listdir(labels_dir)
        
        for i, label_file in enumerate(label_files[:3]):  # Show first 3 files
            print(f"  📄 {label_file}")
            
            # Read and display contents
            label_path = os.path.join(labels_dir, label_file)
            try:
                with open(label_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        print(f"    Content: {content}")
                        # Parse detections
                        lines = content.split('\n')
                        print(f"    Detected {len(lines)} object(s):")
                        for line in lines:
                            parts = line.split()
                            if len(parts) >= 6:
                                class_id = int(parts[0])
                                confidence = float(parts[5]) if len(parts) > 5 else 0.0
                                class_name = CLASSES[class_id] if class_id < len(CLASSES) else f"Class_{class_id}"
                                print(f"      - {class_name}: {confidence:.3f}")
                    else:
                        print("    (empty - no detections)")
            except Exception as e:
                print(f"    Could not read: {e}")

def test_single_image(image_path, conf_threshold=0.25):
    """Test detection on a single image"""
    print(f"\n🔍 Testing single image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    success, results_dir = run_detection(image_path, conf_threshold)
    if success:
        display_results(results_dir)
    return success

def test_directory(directory_path, conf_threshold=0.25):
    """Test detection on all images in a directory"""
    print(f"\n📁 Testing directory: {directory_path}")
    
    if not os.path.exists(directory_path):
        print(f"❌ Directory not found: {directory_path}")
        return False
    
    # Find all image files
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(directory_path, pattern)))
    
    if not image_files:
        print(f"❌ No image files found in {directory_path}")
        return False
    
    print(f"Found {len(image_files)} image(s)")
    
    success, results_dir = run_detection(directory_path, conf_threshold)
    if success:
        display_results(results_dir)
    return success

def run_validation(conf_threshold=0.25, iou_threshold=0.45):
    """Run validation on test dataset"""
    print(f"\n🧪 Running validation with conf={conf_threshold}, iou={iou_threshold}")
    
    val_cmd = [
        sys.executable, 'val.py',
        '--weights', WEIGHTS_PATH,
        '--data', CONFIG_PATH,
        '--img', '640',
        '--conf', str(conf_threshold),
        '--iou', str(iou_threshold),
        '--save-txt',
        '--task', 'test',
        '--project', 'runs/val',
        '--name', 'test_validation',
        '--exist-ok'
    ]
    
    print(f"Command: {' '.join(val_cmd)}")
    
    try:
        result = subprocess.run(val_cmd, capture_output=True, text=True, check=True)
        print("✅ Validation completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running validation: {e}")
        print(f"stderr: {e.stderr}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='YOLOv5 Detection Test Script')
    parser.add_argument('--source', type=str, default='d:/AITest/AIXRAY/data/images/test/0391d2388a2442f14d055d5089a747c6.jpg',
                       help='Source image/directory path')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--mode', type=str, choices=['single', 'directory', 'validation'], default='single',
                       help='Test mode: single image, directory, or validation')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation on test dataset')
    
    args = parser.parse_args()
    
    print("🚀 YOLOv5 Medical X-ray Detection Test")
    print("=" * 50)
    
    # Setup environment
    if not setup_environment():
        return
    
    # Run based on mode
    if args.mode == 'single':
        test_single_image(args.source, args.conf)
    elif args.mode == 'directory':
        test_directory(args.source, args.conf)
    elif args.mode == 'validation':
        run_validation(args.conf)
    
    if args.validate:
        run_validation(args.conf)
    
    print("\n✨ Test completed!")

if __name__ == "__main__":
    main()
