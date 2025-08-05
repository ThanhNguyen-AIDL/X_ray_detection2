#!/usr/bin/env python3
"""
Direct YOLOv5 Medical X-ray Detection
Simple implementation that calls YOLOv5 directly without subprocess
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

class DirectYOLODetector:
    """Direct YOLOv5 detector without subprocess calls"""
    
    def __init__(self, weights_path: str = './runs/train/exp/weights/best.pt', conf_thres: float = 0.25):
        # Try to find the weights file in different possible locations
        possible_paths = [
            weights_path,
            f"yolov5/{weights_path}",
            f"yolov5/runs/train/exp/weights/best.pt",
            "./yolov5/runs/train/exp/weights/best.pt",
            "runs/train/exp/weights/best.pt"
        ]
        
        self.weights_path = None
        for path in possible_paths:
            if os.path.exists(path):
                self.weights_path = path
                break
        
        if self.weights_path is None:
            raise FileNotFoundError(f"Could not find weights file. Tried: {possible_paths}")
        
        self.conf_thres = conf_thres
        self.model = None
        self.device = None
        self.names = None
        self._setup_model()
    
    def _setup_model(self):
        """Initialize the YOLOv5 model"""
        try:
            # Add YOLOv5 directory to path
            yolov5_path = os.path.join(os.path.dirname(__file__), 'yolov5')
            if yolov5_path not in sys.path:
                sys.path.append(yolov5_path)
            
            # Select device
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Load model using standard YOLOv5 approach
            logger.info(f"Loading model from {self.weights_path}")
            
            # Use YOLOv5's attempt_load function if available
            try:
                from models.experimental import attempt_load
                self.model = attempt_load(self.weights_path, device=self.device)
                logger.info("Model loaded using YOLOv5's attempt_load")
            except ImportError:
                # Fallback to direct torch.load
                ckpt = torch.load(self.weights_path, map_location=self.device)
                self.model = ckpt['ema'] if 'ema' in ckpt else ckpt['model']
                logger.info("Model loaded using direct torch.load")
            
            # Set model to eval mode
            self.model.eval()
            self.model.to(self.device)
            
            # Get class names
            if hasattr(self.model, 'names'):
                self.names = self.model.names
            elif hasattr(self.model, 'module') and hasattr(self.model.module, 'names'):
                self.names = self.model.module.names
            else:
                # Default medical classes
                self.names = {
                    0: 'Aortic enlargement',
                    1: 'Atelectasis', 
                    2: 'Calcification',
                    3: 'Cardiomegaly',
                    4: 'Consolidation',
                    5: 'ILD',
                    6: 'Infiltration',
                    7: 'Lung Opacity',
                    8: 'Nodule/Mass',
                    9: 'Other lesion',
                    10: 'Pleural effusion',
                    11: 'Pleural thickening',
                    12: 'Pneumothorax',
                    13: 'Pulmonary fibrosis'
                }
            
            logger.info(f"Model loaded successfully with {len(self.names)} classes")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _preprocess_image(self, image_path: str, img_size: int = 640) -> Tuple[torch.Tensor, np.ndarray]:
        """Preprocess image for inference"""
        # Read image
        img0 = cv2.imread(image_path)
        if img0 is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Resize image  
        img = cv2.resize(img0, (img_size, img_size))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        img = torch.from_numpy(img).to(self.device)
        img = img.permute(2, 0, 1)  # HWC to CHW
        img = img.unsqueeze(0)  # Add batch dimension
        
        return img, img0
    
    def _postprocess_detections(self, pred: torch.Tensor, img0_shape: Tuple[int, int]) -> List[Detection]:
        """Convert model predictions to Detection objects"""
        detections = []
        
        if pred is None or len(pred) == 0:
            return detections
        
        # Apply NMS
        pred = self._non_max_suppression(pred, self.conf_thres)
        
        for det in pred:
            if det is not None and len(det):
                # Scale boxes back to original image size
                det[:, :4] = self._scale_boxes((640, 640), det[:, :4], img0_shape)
                
                for *xyxy, conf, cls in det:
                    if conf >= self.conf_thres:
                        # Convert to xywh format
                        x1, y1, x2, y2 = xyxy
                        x, y, w, h = x1, y1, x2 - x1, y2 - y1
                        
                        # Get class name
                        class_id = int(cls)
                        class_name = self.names.get(class_id, f"Class_{class_id}")
                        
                        detections.append(Detection(
                            class_name=class_name,
                            confidence=float(conf),
                            bbox=(float(x), float(y), float(w), float(h))
                        ))
        
        return detections
    
    def _non_max_suppression(self, prediction: torch.Tensor, conf_thres: float = 0.25, iou_thres: float = 0.45):
        """Apply Non-Maximum Suppression"""
        try:
            # Import YOLOv5 NMS function
            from utils.general import non_max_suppression
            return non_max_suppression(prediction, conf_thres, iou_thres)
        except ImportError:
            # Fallback to basic filtering if YOLOv5 utils not available
            return [prediction[0][prediction[0][:, 4] >= conf_thres]] if len(prediction) > 0 else []
    
    def _scale_boxes(self, img1_shape: Tuple[int, int], boxes: torch.Tensor, img0_shape: Tuple[int, int]):
        """Scale boxes from img1_shape to img0_shape"""
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
        
        boxes[:, [0, 2]] -= pad[0]  # x padding
        boxes[:, [1, 3]] -= pad[1]  # y padding
        boxes[:, :4] /= gain
        
        # Clip boxes
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, img0_shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, img0_shape[0])  # y1, y2
        
        return boxes
    
    def detect_single_image(self, image_path: str) -> ImageResult:
        """Run detection on a single image"""
        start_time = datetime.now()
        
        try:
            # Preprocess image
            img, img0 = self._preprocess_image(image_path)
            
            # Run inference
            with torch.no_grad():
                pred = self.model(img)
            
            # Postprocess detections
            detections = self._postprocess_detections(pred, img0.shape[:2])
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ImageResult(
                image_name=os.path.basename(image_path),
                image_path=image_path,
                detections=detections,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Detection failed for {image_path}: {e}")
            raise
    
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
