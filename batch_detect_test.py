#!/usr/bin/env python3
"""
Batch YOLOv5 Detection Test Script
Process multiple images and generate a summary report
"""

import os
import sys
import subprocess
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import glob
from pathlib import Path
import json
from datetime import datetime

def batch_detection_test():
    """Run detection on multiple test images and generate report"""
    print("🚀 YOLOv5 Batch Detection Test")
    print("=" * 50)
    
    # Configuration
    yolov5_dir = '/yolov5-bk'
    weights_path = './runs/train/exp/weights/best.pt'
    test_dir = 'd:/AITest/AIXRAY/data/images/test'
    
    # Medical condition classes
    classes = [
        'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 
        'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass', 
        'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 
        'Pulmonary fibrosis'
    ]
    
    # Setup
    os.chdir(yolov5_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Check requirements
    if not os.path.exists(weights_path):
        print(f"❌ Weights not found: {weights_path}")
        return
    
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    # Find test images
    image_patterns = ['*.jpg', '*.jpeg', '*.png']
    test_images = []
    for pattern in image_patterns:
        test_images.extend(glob.glob(os.path.join(test_dir, pattern)))
    
    if not test_images:
        print(f"❌ No test images found in {test_dir}")
        return
    
    print(f"📷 Found {len(test_images)} test images")
    
    # Run batch detection
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_name = f"batch_test_{timestamp}"
    
    detect_cmd = [
        sys.executable, 'detect.py',
        '--weights', weights_path,
        '--img', '640',
        '--source', test_dir,
        '--conf', '0.25',
        '--save-txt',
        '--save-conf',
        '--project', 'runs/detect',
        '--name', results_name,
        '--exist-ok'
    ]
    
    print(f"Running batch detection...")
    print(f"Command: {' '.join(detect_cmd)}")
    
    try:
        result = subprocess.run(detect_cmd, capture_output=True, text=True, check=True)
        print("✅ Batch detection completed!")
        
        # Analyze results
        results_dir = f'runs/detect/{results_name}'
        analyze_batch_results(results_dir, classes, test_images)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Batch detection failed: {e}")
        print(f"Error: {e.stderr}")

def analyze_batch_results(results_dir, classes, test_images):
    """Analyze batch detection results and generate report"""
    print(f"\n📊 Analyzing results from: {results_dir}")
    
    labels_dir = os.path.join(results_dir, 'labels')
    
    if not os.path.exists(labels_dir):
        print("❌ No labels directory found")
        return
    
    # Collect detection statistics
    detection_stats = {
        'total_images': len(test_images),
        'images_with_detections': 0,
        'total_detections': 0,
        'class_counts': {cls: 0 for cls in classes},
        'confidence_scores': [],
        'results_per_image': []
    }
    
    label_files = os.listdir(labels_dir)
    print(f"📄 Processing {len(label_files)} label files...")
    
    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        image_name = os.path.splitext(label_file)[0]
        
        try:
            with open(label_path, 'r') as f:
                content = f.read().strip()
                
                if content:
                    detection_stats['images_with_detections'] += 1
                    lines = content.split('\n')
                    num_detections = len(lines)
                    detection_stats['total_detections'] += num_detections
                    
                    image_results = {
                        'image_name': image_name,
                        'num_detections': num_detections,
                        'detections': []
                    }
                    
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 6:
                            class_id = int(parts[0])
                            confidence = float(parts[5])
                            
                            if class_id < len(classes):
                                class_name = classes[class_id]
                                detection_stats['class_counts'][class_name] += 1
                                detection_stats['confidence_scores'].append(confidence)
                                
                                image_results['detections'].append({
                                    'class': class_name,
                                    'confidence': confidence
                                })
                    
                    detection_stats['results_per_image'].append(image_results)
                else:
                    detection_stats['results_per_image'].append({
                        'image_name': image_name,
                        'num_detections': 0,
                        'detections': []
                    })
        
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
    
    # Generate report
    generate_detection_report(detection_stats, results_dir)

def generate_detection_report(stats, results_dir):
    """Generate and display detection report"""
    print("\n" + "="*60)
    print("📋 DETECTION REPORT")
    print("="*60)
    
    # Summary statistics
    print(f"🖼️  Total images processed: {stats['total_images']}")
    print(f"🎯 Images with detections: {stats['images_with_detections']}")
    print(f"📊 Detection rate: {stats['images_with_detections']/stats['total_images']*100:.1f}%")
    print(f"🔍 Total detections: {stats['total_detections']}")
    
    if stats['confidence_scores']:
        avg_conf = sum(stats['confidence_scores']) / len(stats['confidence_scores'])
        print(f"📈 Average confidence: {avg_conf:.3f}")
        print(f"📈 Confidence range: {min(stats['confidence_scores']):.3f} - {max(stats['confidence_scores']):.3f}")
    
    # Class distribution
    print(f"\n🏷️  DETECTED CONDITIONS:")
    print("-" * 40)
    total_class_detections = sum(stats['class_counts'].values())
    
    if total_class_detections > 0:
        sorted_classes = sorted(stats['class_counts'].items(), key=lambda x: x[1], reverse=True)
        for class_name, count in sorted_classes:
            if count > 0:
                percentage = count / total_class_detections * 100
                print(f"  {class_name:<25} {count:3d} ({percentage:5.1f}%)")
    else:
        print("  No detections found")
    
    # Top detections per image
    print(f"\n📷 TOP DETECTION RESULTS:")
    print("-" * 40)
    images_with_detections = [img for img in stats['results_per_image'] if img['num_detections'] > 0]
    images_with_detections.sort(key=lambda x: x['num_detections'], reverse=True)
    
    for i, img_result in enumerate(images_with_detections[:5]):  # Top 5
        print(f"  {i+1}. {img_result['image_name']}: {img_result['num_detections']} detection(s)")
        for det in img_result['detections'][:3]:  # Top 3 detections per image
            print(f"     - {det['class']}: {det['confidence']:.3f}")
    
    # Save detailed report
    report_path = os.path.join(results_dir, 'detection_report.json')
    try:
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n💾 Detailed report saved to: {report_path}")
    except Exception as e:
        print(f"❌ Could not save report: {e}")
    
    # Create visualization
    try:
        create_detection_visualizations(stats, results_dir)
    except Exception as e:
        print(f"❌ Could not create visualizations: {e}")

def create_detection_visualizations(stats, results_dir):
    """Create visualization charts for detection results"""
    print("\n📊 Creating visualizations...")
    
    # Class distribution chart
    class_counts = {k: v for k, v in stats['class_counts'].items() if v > 0}
    
    if class_counts:
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Class distribution
        plt.subplot(2, 2, 1)
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        plt.bar(range(len(classes)), counts)
        plt.title('Detection Count by Medical Condition')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        plt.ylabel('Count')
        
        # Subplot 2: Confidence distribution
        if stats['confidence_scores']:
            plt.subplot(2, 2, 2)
            plt.hist(stats['confidence_scores'], bins=20, alpha=0.7)
            plt.title('Confidence Score Distribution')
            plt.xlabel('Confidence')
            plt.ylabel('Frequency')
        
        # Subplot 3: Detection rate
        plt.subplot(2, 2, 3)
        labels = ['With Detections', 'No Detections']
        sizes = [stats['images_with_detections'], 
                stats['total_images'] - stats['images_with_detections']]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.title('Detection Rate')
        
        # Subplot 4: Top conditions
        plt.subplot(2, 2, 4)
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_classes = [item[0] for item in sorted_classes]
        top_counts = [item[1] for item in sorted_classes]
        plt.barh(range(len(top_classes)), top_counts)
        plt.title('Top 5 Detected Conditions')
        plt.yticks(range(len(top_classes)), top_classes)
        plt.xlabel('Count')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(results_dir, 'detection_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Visualization saved to: {plot_path}")
        plt.show()

if __name__ == "__main__":
    batch_detection_test()
