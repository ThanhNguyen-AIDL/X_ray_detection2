#!/usr/bin/env python3
"""
YOLOv5 Medical Detection CLI
Simplified command-line interface for the refactored detection system
"""

import argparse
import sys
import logging
from pathlib import Path
from yolo_detector_refactored import (
    DetectionConfig, YOLOv5Detector, ResultsAnalyzer, 
    DetectionError, logger
)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser"""
    parser = argparse.ArgumentParser(
        description="YOLOv5 Medical X-ray Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image detection
  python detection_cli.py single path/to/image.jpg
  
  # Batch detection on directory
  python detection_cli.py batch path/to/images/
  
  # Batch with custom confidence threshold
  python detection_cli.py batch path/to/images/ --conf 0.5
  
  # Generate report without visualization
  python detection_cli.py batch path/to/images/ --no-viz
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Detection mode')
    
    # Single image detection
    single_parser = subparsers.add_parser('single', help='Detect on single image')
    single_parser.add_argument('image', help='Path to input image')
    single_parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold (default: 0.25)')
    single_parser.add_argument('--save', action='store_true', help='Save detection results')
    
    # Batch detection
    batch_parser = subparsers.add_parser('batch', help='Detect on multiple images')
    batch_parser.add_argument('directory', help='Path to directory containing images')
    batch_parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold (default: 0.25)')
    batch_parser.add_argument('--no-viz', action='store_true', help='Skip visualization generation')
    batch_parser.add_argument('--output', help='Output directory for results (default: current directory)')
    
    # Global options
    parser.add_argument('--weights', help='Path to model weights file')
    parser.add_argument('--config', help='Path to dataset config file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    return parser


def handle_single_detection(args, detector: YOLOv5Detector) -> None:
    """Handle single image detection"""
    logger.info(f"Processing single image: {args.image}")
    
    try:
        result = detector.detect_single_image(args.image, save_results=args.save)
        
        # Display results
        print(f"\n{'='*50}")
        print(f"🔍 DETECTION RESULTS")
        print(f"{'='*50}")
        print(f"Image: {result.image_name}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Detections found: {result.detection_count}")
        
        if result.has_detections:
            print(f"\n📋 DETECTED CONDITIONS:")
            print(f"{'-'*30}")
            for i, detection in enumerate(result.detections, 1):
                print(f"{i:2d}. {detection.class_name}")
                print(f"    Confidence: {detection.confidence_percent}")
                print(f"    Location: x={detection.bbox[0]:.3f}, y={detection.bbox[1]:.3f}")
                print(f"    Size: w={detection.bbox[2]:.3f}, h={detection.bbox[3]:.3f}")
        else:
            print(f"\n🔍 No medical conditions detected")
            
    except DetectionError as e:
        logger.error(f"Detection failed: {e}")
        sys.exit(1)


def handle_batch_detection(args, detector: YOLOv5Detector, analyzer: ResultsAnalyzer) -> None:
    """Handle batch detection"""
    logger.info(f"Processing batch from directory: {args.directory}")
    
    try:
        results = detector.detect_batch(args.directory)
        
        # Set output directory
        output_dir = args.output or "."
        Path(output_dir).mkdir(exist_ok=True)
        
        # Generate and display report
        report_path = Path(output_dir) / "detection_report.txt"
        report = analyzer.generate_report(results, str(report_path))
        print(report)
        
        # Create visualizations unless disabled
        if not args.no_viz:
            viz_dir = Path(output_dir) / "visualizations"
            analyzer.create_visualizations(results, str(viz_dir))
        
        # Save detailed results as JSON
        json_path = Path(output_dir) / "detection_results.json"
        save_results_json(results, str(json_path))
        
        logger.info(f"Results saved to: {output_dir}")
        
    except DetectionError as e:
        logger.error(f"Batch detection failed: {e}")
        sys.exit(1)


def save_results_json(results, output_path: str) -> None:
    """Save detailed results to JSON file"""
    import json
    
    json_data = {
        "timestamp": str(datetime.now()),
        "total_images": len(results),
        "results": []
    }
    
    for result in results:
        image_data = {
            "image_name": result.image_name,
            "image_path": result.image_path,
            "processing_time": result.processing_time,
            "detection_count": result.detection_count,
            "detections": []
        }
        
        for detection in result.detections:
            det_data = {
                "class_id": detection.class_id,
                "class_name": detection.class_name,
                "confidence": detection.confidence,
                "bbox": {
                    "x": detection.bbox[0],
                    "y": detection.bbox[1],
                    "width": detection.bbox[2],
                    "height": detection.bbox[3]
                }
            }
            image_data["detections"].append(det_data)
        
        json_data["results"].append(image_data)
    
    try:
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"Detailed results saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving JSON results: {e}")


def main():
    """Main CLI function"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Initialize configuration
        config = DetectionConfig()
        
        # Override config with command line arguments
        if args.weights:
            config.weights_path = args.weights
        if args.config:
            config.config_path = args.config
        if hasattr(args, 'conf'):
            config.conf_threshold = args.conf
        
        # Initialize detector and analyzer
        detector = YOLOv5Detector(config)
        analyzer = ResultsAnalyzer(config)
        
        # Handle commands
        if args.command == 'single':
            handle_single_detection(args, detector)
        elif args.command == 'batch':
            handle_batch_detection(args, detector, analyzer)
        
        logger.info("Detection completed successfully! ✨")
        
    except KeyboardInterrupt:
        logger.info("Detection interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    from datetime import datetime
    main()
