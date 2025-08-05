#!/usr/bin/env python3
"""
Test the simple YOLOv5 detector
"""

import os
import sys
from pathlib import Path

# Test image path
test_image = r"D:\AITest\AIXRAY\data\images\test\0a1aef5326b7b24378c6692f7a454e52.jpg"

def test_simple_detector():
    """Test the simple detector"""
    try:
        from simple_yolo_detector import SimpleYOLODetector
        
        print("🚀 Testing Simple YOLOv5 Detector...")
        
        # Check if test image exists
        if not os.path.exists(test_image):
            print(f"❌ Test image not found: {test_image}")
            return False
        
        # Initialize detector (will auto-find weights)
        print("📦 Loading detector...")
        detector = SimpleYOLODetector(conf_thres=0.25)
        
        # Run detection
        print(f"🔍 Running detection on: {test_image}")
        result = detector.detect_single_image(test_image)
        
        # Print results
        print(f"\n✅ Detection Results:")
        print(f"Image: {result.image_name}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Detections found: {result.detection_count}")
        
        for i, det in enumerate(result.detections, 1):
            print(f"  {i}. {det.class_name}: {det.confidence_percent}")
            print(f"     BBox: {det.bbox}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_detector()
    print(f"\n{'✅ Test PASSED' if success else '❌ Test FAILED'}")
