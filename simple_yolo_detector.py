#!/usr/bin/env python3
"""
Simple YOLOv5 Direct Detector
Uses YOLOv5's built-in functions directly without subprocess calls
"""

import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

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
    
    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0
    
    @property
    def detection_count(self) -> int:
        return len(self.detections)

class SimpleYOLODetector:
    """Simple YOLOv5 detector using YOLOv5's own functions"""
    
    def __init__(self, weights_path: str = None, conf_thres: float = 0.25):
        # Find weights file
        if weights_path is None:
            weights_path = self._find_weights()
        
        self.weights_path = weights_path
        self.conf_thres = conf_thres
        self.model = None
        self.device = None
        self.names = None
        self._setup_yolov5()
        self._load_model()
    
    def _find_weights(self) -> str:
        """Find the best weights file"""
        possible_paths = [
            "./runs/train/exp/weights/best.pt",
            "yolov5/runs/train/exp/weights/best.pt", 
            "./yolov5/runs/train/exp/weights/best.pt",
            "runs/train/exp/weights/best.pt"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(f"Could not find weights file. Tried: {possible_paths}")
    
    def _setup_yolov5(self):
        """Setup YOLOv5 environment"""
        # Add YOLOv5 to path
        yolov5_path = os.path.join(os.path.dirname(__file__), 'yolov5')
        if os.path.exists(yolov5_path) and yolov5_path not in sys.path:
            sys.path.insert(0, yolov5_path)
            logger.info(f"Added YOLOv5 path: {yolov5_path}")
        
        # Select device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def _load_model(self):
        """Load the YOLOv5 model using the same method as detect.py"""
        try:
            logger.info(f"Loading model from {self.weights_path}")
            
            # Change to yolov5 directory like the original detect.py
            original_cwd = os.getcwd()
            yolov5_dir = os.path.join(os.path.dirname(__file__), 'yolov5')
            if os.path.exists(yolov5_dir):
                os.chdir(yolov5_dir)
                logger.info(f"Changed to YOLOv5 directory: {yolov5_dir}")
            
            try:
                # Use YOLOv5's DetectMultiBackend (same as detect.py)
                from models.common import DetectMultiBackend
                from utils.general import check_img_size
                from utils.torch_utils import select_device
                
                # Select device (same as detect.py)
                device = select_device('')
                self.device = device
                
                # Load model using DetectMultiBackend (exact same as detect.py)
                weights_path = os.path.relpath(os.path.join(original_cwd, self.weights_path))
                self.model = DetectMultiBackend(weights_path, device=device, dnn=False, data=None, fp16=False)
                self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
                
                # Check image size (same as detect.py)
                self.imgsz = check_img_size((640, 640), s=self.stride)
                
                # Model warmup (same as detect.py)
                self.model.warmup(imgsz=(1 if self.pt or self.model.triton else 1, 3, *self.imgsz))
                
                logger.info(f"Model loaded successfully using DetectMultiBackend")
                logger.info(f"Model names: {self.names}")
                
            finally:
                # Change back to original directory
                os.chdir(original_cwd)
                
        except Exception as e:
            # Change back to original directory on error
            try:
                os.chdir(original_cwd)
            except:
                pass
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect_single_image(self, image_path: str) -> ImageResult:
        """Run detection on a single image using YOLOv5's exact method"""
        start_time = datetime.now()
        
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            logger.info(f"Processing image: {image_path}")
            
            # Change to yolov5 directory for imports
            original_cwd = os.getcwd()
            yolov5_dir = os.path.join(os.path.dirname(__file__), 'yolov5')
            if os.path.exists(yolov5_dir):
                os.chdir(yolov5_dir)
            
            try:
                # Import YOLOv5 modules (same as detect.py)
                from utils.dataloaders import LoadImages
                from utils.general import non_max_suppression, scale_boxes
                from utils.torch_utils import smart_inference_mode
                import torch
                
                # Load images using YOLOv5's dataloader (same as detect.py)
                dataset = LoadImages(image_path, img_size=self.imgsz, stride=self.stride, auto=self.pt)
                
                detections = []
                
                # Process each image (same logic as detect.py)
                for path, im, im0s, vid_cap, s in dataset:
                    # Preprocess (same as detect.py)
                    im = torch.from_numpy(im).to(self.model.device)
                    im = im.half() if self.model.fp16 else im.float()
                    im /= 255
                    if len(im.shape) == 3:
                        im = im[None]
                    
                    # Inference (same as detect.py)
                    pred = self.model(im, augment=False, visualize=False)
                    
                    # NMS (same as detect.py)
                    pred = non_max_suppression(pred, self.conf_thres, 0.45, None, False, max_det=1000)
                    
                    # Process detections (same as detect.py)
                    for i, det in enumerate(pred):
                        if len(det):
                            # Rescale boxes from img_size to original image size
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
                            
                            # Process each detection
                            for *xyxy, conf, cls in reversed(det):
                                if conf >= self.conf_thres:
                                    # Convert to xywh format
                                    x1, y1, x2, y2 = xyxy
                                    x, y, w, h = float(x1), float(y1), float(x2 - x1), float(y2 - y1)
                                    
                                    # Get class name
                                    class_id = int(cls)
                                    class_name = self.names.get(class_id, f"Class_{class_id}")
                                    
                                    detections.append(Detection(
                                        class_name=class_name,
                                        confidence=float(conf),
                                        bbox=(x, y, w, h)
                                    ))
                
            finally:
                # Change back to original directory
                os.chdir(original_cwd)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ImageResult(
                image_name=os.path.basename(image_path),
                image_path=image_path,
                detections=detections,
                processing_time=processing_time
            )
            
        except Exception as e:
            try:
                os.chdir(original_cwd)
            except:
                pass
            logger.error(f"Detection failed for {image_path}: {e}")
            raise
    
    def _parse_yolov5_results(self, results) -> List[Detection]:
        """Parse YOLOv5 results into Detection objects"""
        detections = []
        
        try:
            # YOLOv5 results parsing
            for result in results.xyxy[0]:  # results for first image
                x1, y1, x2, y2, conf, cls = result.tolist()
                
                if conf >= self.conf_thres:
                    # Ensure coordinates are in correct order
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    
                    # Convert to xywh format
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                    
                    # Ensure positive width and height
                    w = abs(w)
                    h = abs(h)
                    
                    # Get class name
                    class_id = int(cls)
                    class_name = self.names.get(class_id, f"Class_{class_id}")
                    
                    detections.append(Detection(
                        class_name=class_name,
                        confidence=float(conf),
                        bbox=(float(x), float(y), float(w), float(h))
                    ))
        except Exception as e:
            logger.warning(f"Failed to parse YOLOv5 results: {e}")
        
        return detections
    
    def _detect_with_model(self, image_path: str) -> List[Detection]:
        """Direct model inference"""
        detections = []
        
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Simple preprocessing
            img = cv2.resize(img, (640, 640))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).to(self.device)
            img = img.permute(2, 0, 1).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                pred = self.model(img)
            
            # Basic NMS and filtering
            if pred is not None and len(pred) > 0:
                pred = pred[0]  # First image
                
                # Filter by confidence
                mask = pred[:, 4] >= self.conf_thres
                pred = pred[mask]
                
                for detection in pred:
                    x1, y1, x2, y2, conf, cls = detection[:6].tolist()
                    
                    # Ensure coordinates are in correct order
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    
                    # Convert to xywh
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                    
                    # Ensure positive width and height
                    w = abs(w)
                    h = abs(h)
                    
                    # Get class name
                    class_id = int(cls) if len(detection) > 5 else 0
                    class_name = self.names.get(class_id, f"Class_{class_id}")
                    
                    detections.append(Detection(
                        class_name=class_name,
                        confidence=float(conf),
                        bbox=(float(x), float(y), float(w), float(h))
                    ))
        
        except Exception as e:
            logger.warning(f"Direct model inference failed: {e}")
        
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
