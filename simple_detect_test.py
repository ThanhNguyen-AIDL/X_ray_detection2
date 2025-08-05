#!/usr/bin/env python3
"""
Simple YOLOv5 Detection Test Script
Easy-to-use script for testing YOLOv5 on medical X-ray images
"""

import os
import sys
import subprocess
import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    """Main detection test function"""
    print("🚀 YOLOv5 Medical X-ray Detection Test")
    print("=" * 50)
    
    # Configuration
    yolov5_dir = '/yolov5-bk'
    weights_path = './runs/train/exp/weights/best.pt'
    config_path = 'data/vinbigdata.yaml'
    
    # Test image path
    test_image = 'd:/AITest/AIXRAY/data/images/test/0391d2388a2442f14d055d5089a747c6.jpg'
    
    # Medical condition classes
    classes = [
        'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 
        'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass', 
        'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 
        'Pulmonary fibrosis'
    ]
    
    # Setup environment
    print("Setting up environment...")
    os.chdir(yolov5_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Check if files exist
    if not os.path.exists(weights_path):
        print(f"❌ Weights file not found: {weights_path}")
        print("Please train the model first or use pre-trained weights")
        return
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    print("✅ Files found, starting detection...")
    
    # Run detection
    detect_cmd = [
        sys.executable, 'detect.py',
        '--weights', weights_path,
        '--img', '640',
        '--source', test_image,
        '--conf', '0.25',
        '--data', 'data/vinbigdata.yaml',  # Use custom dataset config
        '--save-txt',
        '--save-conf',
        '--project', 'runs/detect',
        '--name', 'simple_test',
        '--exist-ok'
    ]
    
    print(f"Command: {' '.join(detect_cmd)}")
    
    try:
        print("Running detection...")
        result = subprocess.run(detect_cmd, capture_output=True, text=True, check=True)
        print("✅ Detection completed successfully!")
        
        # Show results
        results_dir = 'runs/detect/simple_test'
        print(f"\n📁 Results saved in: {results_dir}")
        
        # Display detected image
        import glob
        result_images = glob.glob(f"{results_dir}/*.jpg") + glob.glob(f"{results_dir}/*.png")
        
        if result_images:
            img_path = result_images[0]
            print(f"📷 Displaying result: {img_path}")
            
            # Load and display image
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(12, 8))
                plt.imshow(img_rgb)
                plt.title("YOLOv5 Detection Results")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
        
        # Show detection labels
        labels_dir = f"{results_dir}/labels"
        if os.path.exists(labels_dir):
            label_files = os.listdir(labels_dir)
            if label_files:
                label_path = os.path.join(labels_dir, label_files[0])
                print(f"\n🏷️  Reading detections from: {label_files[0]}")
                
                try:
                    with open(label_path, 'r') as f:
                        content = f.read().strip()
                        if content:
                            lines = content.split('\n')
                            print(f"Found {len(lines)} detection(s):")
                            for i, line in enumerate(lines):
                                parts = line.split()
                                if len(parts) >= 6:
                                    class_id = int(parts[0])
                                    confidence = float(parts[5])
                                    class_name = classes[class_id] if class_id < len(classes) else f"Class_{class_id}"
                                    print(f"  {i+1}. {class_name}: {confidence:.3f}")
                        else:
                            print("No detections found")
                except Exception as e:
                    print(f"Error reading labels: {e}")
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Detection failed: {e}")
        print(f"Error output: {e.stderr}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print("\n✨ Test completed!")

if __name__ == "__main__":
    main()
