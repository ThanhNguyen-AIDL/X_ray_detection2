#!/usr/bin/env python3
"""
Simple test script for the refactored detection system
"""

from yolo_detector_refactored import DetectionConfig, YOLOv5Detector, ResultsAnalyzer

def main():
    print("Testing Refactored YOLOv5 Detection System")
    print("=" * 50)
    
    # Initialize
    config = DetectionConfig()
    config.conf_threshold = 0.25
    
    detector = YOLOv5Detector(config)
    analyzer = ResultsAnalyzer(config)
    
    # Test single image
    test_image = "d:/AITest/AIXRAY/data/images/test/0391d2388a2442f14d055d5089a747c6.jpg"
    
    print(f"Processing: {test_image}")
    
    try:
        result = detector.detect_single_image(test_image)
        
        print(f"\nResults:")
        print(f"- Image: {result.image_name}")
        print(f"- Processing time: {result.processing_time:.2f}s")
        print(f"- Detections: {result.detection_count}")
        
        if result.has_detections:
            print(f"\nDetected conditions:")
            for i, detection in enumerate(result.detections, 1):
                print(f"  {i}. {detection.class_name}: {detection.confidence_percent}")
        else:
            print("No medical conditions detected")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
