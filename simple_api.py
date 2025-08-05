#!/usr/bin/env python3
"""
Simple FastAPI service for YOLOv5 Medical X-ray Detection
Uses direct model inference without subprocess calls
"""

import os
import sys
import base64
import json
import uuid
import shutil
from typing import List, Optional
from pathlib import Path

# Import cv2 and numpy only when needed to avoid import errors
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python not available")
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from datetime import datetime

# Import our direct detector that calls detect.py run() function directly
from yolo_direct_detector import YOLOv5DirectDetector, ImageResult, Detection

# Initialize FastAPI
app = FastAPI(
    title="Medical X-ray Detection API",
    description="Simple API for detecting medical conditions in chest X-rays using YOLOv5 directly",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the direct detector once at startup
detector = None

@app.on_event("startup")
async def startup_event():
    """Initialize the detector on startup"""
    global detector
    try:
        # Try different possible weights paths
        possible_weights = [
            "./runs/train/exp/weights/best.pt",
            "yolov5/runs/train/exp/weights/best.pt", 
            "./yolov5/runs/train/exp/weights/best.pt",
            "runs/train/exp/weights/best.pt"
        ]
        
        weights_path = None
        for path in possible_weights:
            if os.path.exists(path):
                weights_path = path
                break
        
        if weights_path is None:
            print(f"❌ Could not find weights file. Tried: {possible_weights}")
            detector = None
            return
        
        print(f"📦 Loading CLI detector...")
        detector = YOLOv5DirectDetector(conf_thres=0.25)  # Auto-find weights
        print("✅ Direct YOLOv5 detector initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        detector = None

# Create directories
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Mount static files for serving processed images
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

def draw_detections_on_image(image_path: str, detections: List[Detection], output_path: str):
    """Draw bounding boxes and labels on the image"""
    try:
        # Read the original image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Define colors for different classes (BGR format)
        colors = {
            'Aortic enlargement': (0, 255, 255),  # Yellow
            'Atelectasis': (255, 0, 0),          # Blue
            'Calcification': (0, 255, 0),        # Green
            'Cardiomegaly': (255, 255, 0),       # Cyan
            'Consolidation': (255, 0, 255),      # Magenta
            'ILD': (128, 0, 128),                # Purple
            'Infiltration': (255, 165, 0),       # Orange
            'Lung Opacity': (0, 0, 255),         # Red
            'Nodule/Mass': (128, 128, 0),        # Olive
            'Other lesion': (128, 128, 128),     # Gray
            'Pleural effusion': (255, 192, 203), # Pink
            'Pleural thickening': (165, 42, 42), # Brown
            'Pneumothorax': (0, 128, 128),       # Teal
            'Pulmonary fibrosis': (128, 0, 0)    # Maroon
        }
        
        # Draw each detection
        for det in detections:
            x, y, w, h = det.bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Get color for this class
            color = colors.get(det.class_name, (0, 255, 0))  # Default to green
            
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label text
            label = f"{det.class_name}: {det.confidence_percent}"
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw label background
            cv2.rectangle(img, (x, y - text_height - 10), (x + text_width, y), color, -1)
            
            # Draw label text
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save the processed image
        cv2.imwrite(output_path, img)
        return True
        
    except Exception as e:
        print(f"Error drawing detections: {e}")
        return False

# Pydantic models
class DetectionResponse(BaseModel):
    image_name: str
    image_url: str  # Added image URL for processed image
    detections: List[dict]
    detection_count: int
    processing_time: float
    success: bool

@app.get("/")
def read_root():
    """API root endpoint"""
    return {
        "message": "Medical X-ray Detection API is running",
        "detector_status": "ready" if detector is not None else "not initialized"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if detector is not None else "unhealthy",
        "detector_loaded": detector is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/detect", response_model=DetectionResponse)
@app.post("/detect/single", response_model=DetectionResponse)
async def detect_image(file: UploadFile = File(...)):
    """Detect conditions in a single X-ray image using direct inference"""
    
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"Processing image: {file_path}")
        
        # Run detection using direct detector
        result = detector.detect_single_image(str(file_path))
        
        # Create processed image URL
        if result.output_image_path and os.path.exists(result.output_image_path):
            # Use the processed image created by YOLOv5 detect.py
            processed_filename = f"yolo_result_{uuid.uuid4()}_{file.filename}"
            processed_path = RESULTS_DIR / processed_filename
            shutil.copy2(result.output_image_path, str(processed_path))
            image_url = f"/results/{processed_filename}"
            print(f"📸 Using YOLOv5 processed image: {image_url}")
        else:
            # Fallback: create processed image with bounding boxes if opencv is available
            processed_filename = f"processed_{uuid.uuid4()}_{file.filename}"
            processed_path = RESULTS_DIR / processed_filename
            image_url = f"/results/{processed_filename}"
            
            if CV2_AVAILABLE and result.detections:
                draw_success = draw_detections_on_image(str(file_path), result.detections, str(processed_path))
                if not draw_success:
                    # If drawing fails, copy original image
                    shutil.copy2(str(file_path), str(processed_path))
                    print(f"📸 Fallback: copied original image as processed")
            else:
                # No opencv or no detections, copy original image
                shutil.copy2(str(file_path), str(processed_path))
                print(f"📸 No opencv or detections: copied original image")
        
        # Convert detections to JSON format
        detections_json = []
        for det in result.detections:
            detections_json.append({
                "class_name": det.class_name,
                "confidence": det.confidence,
                "confidence_percent": det.confidence_percent,
                "confidence_score": det.confidence * 100,
                "bbox": {
                    "x": det.bbox[0],
                    "y": det.bbox[1], 
                    "width": det.bbox[2],
                    "height": det.bbox[3]
                }
            })
        
        # Clean up uploaded file
        try:
            os.unlink(file_path)
        except:
            pass
        
        print(f"✅ Detection completed: {result.detection_count} detections found in {result.processing_time:.2f}s")
        print(f"📸 Processed image saved: {image_url}")
        
        return DetectionResponse(
            image_name=result.image_name,
            image_url=image_url,
            detections=detections_json,
            detection_count=result.detection_count,
            processing_time=result.processing_time,
            success=True
        )
        
    except Exception as e:
        # Clean up uploaded file on error
        try:
            os.unlink(file_path)
        except:
            pass
        
        print(f"❌ Detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect/batch")
async def detect_batch(folder_path: str):
    """Detect conditions in multiple images from a directory"""
    
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")
    
    try:
        print(f"Processing batch from: {folder_path}")
        
        # Run batch detection
        results = detector.detect_batch(folder_path)
        
        # Convert results to JSON format
        batch_results = []
        total_detections = 0
        
        for result in results:
            detections_json = []
            for det in result.detections:
                detections_json.append({
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "confidence_percent": det.confidence_percent,
                    "bbox": {
                        "x": det.bbox[0],
                        "y": det.bbox[1],
                        "width": det.bbox[2], 
                        "height": det.bbox[3]
                    }
                })
            
            batch_results.append({
                "image_name": result.image_name,
                "detections": detections_json,
                "detection_count": result.detection_count,
                "processing_time": result.processing_time,
                "has_detections": result.has_detections
            })
            
            total_detections += result.detection_count
        
        print(f"✅ Batch processing completed: {len(results)} images processed, {total_detections} total detections")
        
        return {
            "success": True,
            "processed_images": len(results),
            "total_detections": total_detections,
            "results": batch_results
        }
        
    except Exception as e:
        print(f"❌ Batch detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Medical X-ray Detection API...")
    print("📍 API will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    
    uvicorn.run("simple_api:app", host="0.0.0.0", port=8000, reload=True)
