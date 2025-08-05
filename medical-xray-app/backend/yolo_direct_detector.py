#!/usr/bin/env python3
"""
YOLOv5 Detector that calls the detect.py run() function directly
This is much more efficient than subprocess calls
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import tempfile
import shutil

# Import cv2 only when needed to avoid import errors at module level
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("opencv-python not available - image processing features limited")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Detection result for a single object"""
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x, y, w, h
    
    @property
    def confidence_percent(self) -> str:
        return f"{self.confidence * 100:.1f}%"

@dataclass
class ImageResult:
    """Result for a single image detection"""
    image_name: str
    image_path: str
    detections: List[Detection]
    processing_time: float
    output_image_path: Optional[str] = None  # Path to the result image with bounding boxes
    
    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0
    
    @property
    def detection_count(self) -> int:
        return len(self.detections)

class YOLOv5DirectDetector:
    """YOLOv5 detector that calls detect.py run() function directly"""
    
    def __init__(self, weights_path: str = None, conf_thres: float = 0.25):
        # Find weights file and convert to absolute path
        if weights_path is None:
            weights_path = self._find_weights()
        
        self.weights_path = os.path.abspath(weights_path)
        self.conf_thres = conf_thres
        self.yolov5_dir = self._find_yolov5_dir()
        self.names = self._load_class_names()
        
        # Add yolov5 to path so we can import detect
        if self.yolov5_dir not in sys.path:
            sys.path.insert(0, self.yolov5_dir)
        
        logger.info(f"Using weights: {self.weights_path}")
        logger.info(f"YOLOv5 directory: {self.yolov5_dir}")
    
    def _find_weights(self) -> str:
        """Find the best weights file and return absolute path"""
        possible_paths = [
            "./runs/train/exp/weights/best.pt",
            "yolov5/runs/train/exp/weights/best.pt", 
            "./yolov5/runs/train/exp/weights/best.pt",
            "runs/train/exp/weights/best.pt"
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
        
        raise FileNotFoundError(f"Could not find weights file. Tried: {possible_paths}")
    
    def _find_yolov5_dir(self) -> str:
        """Find YOLOv5 directory"""
        possible_paths = [
            "yolov5",
            "./yolov5",
            os.path.join(os.path.dirname(__file__), "yolov5")
        ]
        
        for path in possible_paths:
            if os.path.exists(os.path.join(path, "detect.py")):
                return os.path.abspath(path)
        
        raise FileNotFoundError("Could not find YOLOv5 directory with detect.py")
    
    def _load_class_names(self) -> Dict[int, str]:
        """Load class names"""
        # Default medical classes (customize as needed)
        return {
            0: 'Aortic enlargement', 1: 'Atelectasis', 2: 'Calcification',
            3: 'Cardiomegaly', 4: 'Consolidation', 5: 'ILD',
            6: 'Infiltration', 7: 'Lung Opacity', 8: 'Nodule/Mass',
            9: 'Other lesion', 10: 'Pleural effusion', 11: 'Pleural thickening',
            12: 'Pneumothorax', 13: 'Pulmonary fibrosis'
        }
    
    def detect_single_image(self, image_path: str) -> ImageResult:
        """Run detection using YOLOv5's run() function directly"""
        start_time = datetime.now()
        
        # Convert to absolute path
        image_path = os.path.abspath(image_path)
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.info(f"Processing image: {image_path}")
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "detect_output")
            
            # Change to YOLOv5 directory
            original_cwd = os.getcwd()
            try:
                os.chdir(self.yolov5_dir)
                
                # Import the run function from detect
                from detect import run
                
                # Call run() function directly with parameters
                run(
                    weights=self.weights_path,
                    source=image_path,
                    data=None,  # Will use default
                    imgsz=(640, 640),
                    conf_thres=self.conf_thres,
                    iou_thres=0.45,
                    max_det=1000,
                    device='',  # Auto-select
                    view_img=False,
                    save_txt=True,
                    save_conf=True,
                    save_crop=False,
                    nosave=False,  # We want to save the image
                    classes=None,
                    agnostic_nms=False,
                    augment=False,
                    visualize=False,
                    update=False,
                    project=output_dir,
                    name="exp",
                    exist_ok=True,
                    line_thickness=3,
                    hide_labels=False,
                    hide_conf=False,
                    half=False,
                    dnn=False,
                    vid_stride=1
                )
                
                logger.info("Detection completed successfully")
                
            finally:
                os.chdir(original_cwd)
            
            # Parse results
            detections = self._parse_detection_results(
                os.path.join(output_dir, "exp"), 
                image_path
            )
            
            # Copy output image if it exists
            output_image_path = None
            result_image = os.path.join(output_dir, "exp", os.path.basename(image_path))
            if os.path.exists(result_image):
                # Copy to a permanent location
                results_dir = "results"
                os.makedirs(results_dir, exist_ok=True)
                output_image_path = os.path.join(results_dir, f"result_{os.path.basename(image_path)}")
                shutil.copy2(result_image, output_image_path)
                logger.info(f"Result image saved to: {output_image_path}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ImageResult(
            image_name=os.path.basename(image_path),
            image_path=image_path,
            detections=detections,
            processing_time=processing_time,
            output_image_path=output_image_path
        )
    
    def _parse_detection_results(self, results_dir: str, image_path: str) -> List[Detection]:
        """Parse detection results from label files"""
        image_name = Path(image_path).stem
        label_file = os.path.join(results_dir, 'labels', f"{image_name}.txt")
        
        detections = []
        
        if not os.path.exists(label_file):
            logger.warning(f"No detection results found for {image_name}")
            return detections
        
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        confidence = float(parts[5])
                        
                        # Convert from normalized center coordinates to pixel coordinates
                        # Load image to get dimensions (use PIL if cv2 not available)
                        if CV2_AVAILABLE:
                            img = cv2.imread(image_path)
                            img_h, img_w = img.shape[:2]
                        else:
                            # Fallback to default image size or use PIL
                            try:
                                from PIL import Image
                                with Image.open(image_path) as img:
                                    img_w, img_h = img.size
                            except ImportError:
                                # Default fallback size
                                img_w, img_h = 640, 640
                        
                        # Convert to pixel coordinates
                        x_center_px = x_center * img_w
                        y_center_px = y_center * img_h
                        width_px = width * img_w
                        height_px = height * img_h
                        
                        # Convert to top-left corner coordinates
                        x = x_center_px - width_px / 2
                        y = y_center_px - height_px / 2
                        
                        # Get class name
                        class_name = self.names.get(class_id, f"Class_{class_id}")
                        
                        detections.append(Detection(
                            class_name=class_name,
                            confidence=confidence,
                            bbox=(x, y, width_px, height_px)
                        ))
                        
        except Exception as e:
            logger.error(f"Error parsing detection results for {image_name}: {e}")
        
        return detections
    
    def detect_batch(self, source_path: str) -> List[ImageResult]:
        """Run detection on multiple images in a directory"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(source_path).glob(f"*{ext}"))
            image_files.extend(Path(source_path).glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No image files found in {source_path}")
        
        logger.info(f"Processing {len(image_files)} images")
        
        results = []
        for image_file in image_files:
            try:
                result = self.detect_single_image(str(image_file))
                results.append(result)
                logger.info(f"Processed {result.image_name}: {result.detection_count} detections")
            except Exception as e:
                logger.error(f"Failed to process {image_file}: {e}")
                continue
        
        return results
