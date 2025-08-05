# Medical X-Ray Application

A complete medical X-ray image analysis application that combines a FastAPI backend with YOLOv5 object detection and a React frontend interface.

## Project Structure

```
medical-xray-app/
├── backend/
│   ├── simple_api.py          # FastAPI server
│   ├── yolo_direct_detector.py # YOLOv5 detection wrapper
│   ├── yolov5/                # YOLOv5 model files
│   └── api_requirements.txt   # Python dependencies
├── frontend/                  # React application
├── start_app.bat             # Single command to start both servers
└── README.md                 # This file
```

## Features

- **Medical X-Ray Detection**: 14-class medical condition detection using trained YOLOv5 model
- **Direct Detection**: No subprocess overhead - direct function calls to YOLOv5
- **Modern Frontend**: React with TypeScript, Vite build system, Tailwind CSS
- **RESTful API**: FastAPI backend with automatic documentation
- **Image Processing**: Supports multiple image formats with bounding box visualization

## Quick Start

### Prerequisites

- Python 3.8+ with virtual environment (yolov5-venv should be activated)
- Node.js 16+ and npm
- Windows OS (for batch file)

### One Command Start

Simply run the startup script:

```batch
start_app.bat
```

This will:
1. Start the FastAPI backend server on http://localhost:8000
2. Start the React frontend development server on http://localhost:5173
3. Open both servers in separate command windows

### Manual Setup

If you need to start servers individually:

#### Backend

```bash
cd backend
# Activate virtual environment
call ../../../yolov5-venv/Scripts/activate
# Install dependencies
pip install -r api_requirements.txt
# Start server
python simple_api.py
```

#### Frontend

```bash
cd frontend
# Install dependencies (if not done already)
npm install
# Start development server
npm run dev
```

## API Endpoints

- `POST /detect/single` - Upload and analyze single X-ray image
- `GET /results/{filename}` - Retrieve processed images with bounding boxes
- `GET /docs` - Interactive API documentation (Swagger UI)

## Model Information

- **Model**: YOLOv5 trained on medical X-ray dataset
- **Classes**: 14 medical conditions including pneumonia, fractures, etc.
- **Weights**: Located at `backend/yolov5/runs/train/exp/weights/best.pt`
- **Input**: X-ray images (JPEG, PNG supported)
- **Output**: Annotated images with bounding boxes and confidence scores

## Development

### Backend Development

The backend uses:
- FastAPI for API framework
- YOLOv5 for object detection
- OpenCV/PIL for image processing
- CORS middleware for frontend communication

### Frontend Development

The frontend features:
- React 18 with TypeScript
- Vite for fast development and building  
- Tailwind CSS for styling
- Component-based architecture
- Image upload and display functionality

## Troubleshooting

1. **Virtual Environment Issues**: Ensure `yolov5-venv` is properly activated
2. **Port Conflicts**: Backend uses 8000, frontend uses 5173
3. **Missing Dependencies**: Run pip/npm install commands in respective directories
4. **CORS Errors**: Backend includes CORS middleware for localhost development

## File Organization

This organized structure separates concerns:
- `/backend` - All Python/API related files
- `/frontend` - All React/Node.js related files  
- Single startup script for easy development
- Clean separation of dependencies and configurations
