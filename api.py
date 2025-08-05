#!/usr/bin/env python3
"""
FastAPI service for YOLOv5 Medical X-ray Detection
Provides API endpoints to integrate with the frontend
"""

import os
import sys
import base64
import json
from typing import List, Optional
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import shutil
import uuid
from datetime import datetime

# Import direct detection system that actually works
from yolo_direct_detector import YOLOv5DirectDetector, ImageResult, Detection
# Import our image processing utilities (for backward compatibility)
try:
    from process_images import find_latest_detection_directory, find_processed_image
except ImportError:
    def find_latest_detection_directory():
        return "results"
    def find_processed_image(result_dir, filename, filepath):
        return f"/results/{filename}", filename

# Initialize FastAPI
app = FastAPI(
    title="Medical X-ray Detection API",
    description="API for detecting medical conditions in chest X-rays using YOLOv5",
    version="1.0.0"
)

# CORS configuration - important for frontend integration
app.add_middleware(
    CORSMiddleware,
    # Allow all origins during development 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize the direct detector once at startup to avoid reloading the model for each request
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
        
        print(f"📦 Loading direct detector...")
        detector = YOLOv5DirectDetector(conf_thres=0.25)  # Auto-find weights
        print("✅ Direct YOLOv5 detector initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        detector = None

# Create directories for uploads and results
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Mount the results directory as static files for image access
# Make sure the directory exists
RESULTS_DIR.mkdir(exist_ok=True)
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# Pydantic models for request/response
class DetectionResponse(BaseModel):
    image_name: str
    image_url: str
    detections: List[dict]
    detection_count: int
    processing_time: float

class BatchResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    results: Optional[List] = None

# Background task storage
background_tasks = {}

@app.get("/")
def read_root():
    """API root endpoint"""
    return {"message": "Medical X-ray Detection API is running"}

@app.post("/detect/single", response_model=DetectionResponse)
async def detect_single_image(file: UploadFile = File(...)):
    """Detect conditions in a single X-ray image using direct inference"""
    
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"Processing image: {file_path}")
        
        # Run direct detection
        result = detector.detect_single_image(str(file_path))
        
        # For compatibility, create result URL (simplified)
        result_url = f"/results/{result.image_name}"
        
        # Create the response
        detections_json = []
        for det in result.detections:
            detections_json.append({
                "class_name": det.class_name,
                "confidence": det.confidence,
                "confidence_percent": det.confidence_percent,
                "confidence_score": det.confidence * 100,  # 0-100 range for UI display
                "bbox": {
                    "x": det.bbox[0],
                    "y": det.bbox[1],
                    "width": det.bbox[2],
                    "height": det.bbox[3]
                },
                "metrics": {
                    "confidence": det.confidence,
                    "confidence_score": det.confidence * 100
                }
            })
        
        print(f"✅ Detection completed: {result.detection_count} detections found in {result.processing_time:.2f}s")
        
        return DetectionResponse(
            image_name=result.image_name,
            image_url=result_url,
            detections=detections_json,
            detection_count=result.detection_count,
            processing_time=result.processing_time
        )
        
    except Exception as e:
        # Clean up uploaded file on error
        try:
            os.unlink(file_path)
        except:
            pass
        
        print(f"❌ Detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
        
        return DetectionResponse(
            image_name=result.image_name,
            image_url=result_url,
            detections=detections_json,
            detection_count=result.detection_count,
            processing_time=result.processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/batch", response_model=BatchResponse)
async def start_batch_detection(background_tasks: BackgroundTasks, folder_path: str = Form(...)):
    """Start a batch detection job on a directory"""
    try:
        # Create job ID
        job_id = str(uuid.uuid4())
        
        # Add task to background tasks
        background_tasks.add_task(process_batch_job, job_id, folder_path)
        
        return BatchResponse(
            job_id=job_id,
            status="started",
            message=f"Batch processing started for {folder_path}"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of a batch detection job"""
    if job_id not in background_tasks:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return background_tasks[job_id]

def process_batch_job(job_id: str, folder_path: str):
    """Background task for processing batch detection"""
    
    if detector is None:
        background_tasks[job_id] = {
            "job_id": job_id,
            "status": "failed", 
            "progress": 0.0,
            "message": "Detector not initialized",
            "results": []
        }
        return
    
    try:
        # Initialize job status
        background_tasks[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "progress": 0.0,
            "message": "Starting batch processing...",
            "results": []
        }
        
        # Process batch using direct detection
        print(f"Processing batch from: {folder_path}")
        results = detector.detect_batch(folder_path)
        
        # Process results for response
        processed_results = []
        for result in results:
            # Simplified result URL
            result_url = f"/results/{result.image_name}"
            
            processed_results.append({
                "image_name": result.image_name,
                "detections_count": result.detection_count,
                "has_detections": result.has_detections,
                "result_url": result_url,
                "processing_time": result.processing_time
            })
        
        # Generate simple report text file
        report_path = RESULTS_DIR / f"report_{job_id}.txt"
        with open(report_path, 'w') as f:
            f.write(f"Batch Detection Report\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Processed: {len(results)} images\n")
            f.write(f"Total detections: {sum(r.detection_count for r in results)}\n\n")
            
            for result in results:
                f.write(f"\nImage: {result.image_name}\n")
                f.write(f"Processing time: {result.processing_time:.2f}s\n")
                f.write(f"Detections: {result.detection_count}\n")
                for det in result.detections:
                    f.write(f"  - {det.class_name}: {det.confidence_percent}\n")
        
        # Update job status
        background_tasks[job_id] = {
            "job_id": job_id,
            "status": "completed",
            "progress": 100.0,
            "message": f"Batch processing completed: {len(results)} images processed",
            "results": processed_results
        }
        
        print(f"✅ Batch processing completed: {len(results)} images, {sum(r.detection_count for r in results)} total detections")
        
    except Exception as e:
        # Update job status on error
        background_tasks[job_id] = {
            "job_id": job_id,
            "status": "failed",
            "progress": 0.0,
            "message": f"Error: {str(e)}",
            "results": []
        }
        print(f"❌ Batch processing failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
