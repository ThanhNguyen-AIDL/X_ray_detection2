# AIXRAY Setup Instructions

## Virtual Environment Setup

**IMPORTANT: Always activate the virtual environment before running any Python scripts**

```powershell
# From the AIXRAY directory
.\yolov5-venv\Scripts\Activate.ps1
```

## Environment Requirements

The project uses a virtual environment located at `yolov5-venv\` which contains:
- PyTorch 2.7+
- YOLOv5 dependencies
- opencv-python (cv2)
- All required packages

## API Usage

After activating the virtual environment:

```powershell
# Start the simple API
python simple_api.py

# Or start the full API
python api.py
```

## Testing Detection

```powershell
# Test direct detector
python -c "from yolo_direct_detector import YOLOv5DirectDetector; detector = YOLOv5DirectDetector(); print('Direct detector works!')"
```

## Key Files

- `yolo_direct_detector.py` - Direct YOLOv5 detection (calls detect.py run() function)
- `simple_api.py` - Simple FastAPI backend 
- `api.py` - Full FastAPI backend
- `yolov5/runs/train/exp/weights/best.pt` - Trained model weights

## Notes

- Always use the virtual environment (`yolov5-venv`)
- The direct detector calls YOLOv5's detect.py run() function directly
- No subprocess calls - direct function calls for better performance
- Medical X-ray detection with 14 classes
