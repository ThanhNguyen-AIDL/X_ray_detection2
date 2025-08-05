# YOLOv5 Medical X-ray Detection System 🏥

A refactored and improved YOLOv5-based system for detecting medical conditions in chest X-ray images.

## 🎯 Features

- **14 Medical Condition Detection**: Detects various pathological conditions in chest X-rays
- **Batch Processing**: Process multiple images efficiently
- **Comprehensive Analysis**: Generate detailed reports and visualizations
- **Cross-Platform Compatibility**: Fixed PyTorch 2.7+ compatibility issues
- **CLI Interface**: Easy-to-use command-line interface
- **Configurable**: Customizable through configuration files
- **Logging**: Comprehensive logging for debugging and monitoring

## 📋 Detected Medical Conditions

1. Aortic enlargement
2. Atelectasis
3. Calcification
4. Cardiomegaly
5. Consolidation
6. ILD (Interstitial Lung Disease)
7. Infiltration
8. Lung Opacity
9. Nodule/Mass
10. Other lesion
11. Pleural effusion
12. Pleural thickening
13. Pneumothorax
14. Pulmonary fibrosis

## 🚀 Quick Start

### 1. Setup Environment

```powershell
# Activate your virtual environment
.\.venv\Scripts\Activate.ps1

# Run the setup script
python setup.py
```

### 2. Single Image Detection

```powershell
python detection_cli.py single path/to/xray.jpg
```

### 3. Batch Detection

```powershell
python detection_cli.py batch path/to/images/directory/
```

## 📁 Project Structure

```
AIXRAY/
├── yolov5/                          # YOLOv5 repository
│   ├── models/experimental.py       # Modified for PyTorch 2.7+ compatibility
│   ├── data/vinbigdata.yaml        # Dataset configuration
│   └── runs/train/exp/weights/      # Trained model weights
├── data/                            # Dataset
│   ├── images/                      # X-ray images
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/                      # YOLO format labels
├── yolo_detector_refactored.py      # Main refactored detection system
├── detection_cli.py                # Command-line interface
├── setup.py                        # Environment setup script
├── config.ini                      # Configuration file
├── api.py                          # FastAPI backend service
├── api_requirements.txt            # API dependencies
├── start_api.py                    # Script to start the API server
├── uploads/                        # Directory for uploaded images
├── results/                        # Directory for API results
├── vital-scan-analysis-main/       # Frontend web application
│   └── src/
│       └── integration.js          # Frontend API integration
└── README.md                       # This file
```

## 🔧 Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Step-by-Step Installation

1. **Clone and setup the project**:
   ```powershell
   cd d:\AITest\AIXRAY
   .\.venv\Scripts\Activate.ps1
   ```

2. **Run the setup script**:
   ```powershell
   python setup.py
   ```

3. **Verify installation**:
   ```powershell
   python detection_cli.py --help
   ```

## 💻 Usage

### Command-Line Interface

#### Single Image Detection
```powershell
# Basic detection
python detection_cli.py single image.jpg

# With custom confidence threshold
python detection_cli.py single image.jpg --conf 0.5

# Save detection results
python detection_cli.py single image.jpg --save
```

#### Batch Detection
```powershell
# Process all images in directory
python detection_cli.py batch /path/to/images/

# With custom output directory
python detection_cli.py batch /path/to/images/ --output results/

# Skip visualizations
python detection_cli.py batch /path/to/images/ --no-viz
```

### Python API

```python
from yolo_detector_refactored import DetectionConfig, YOLOv5Detector, ResultsAnalyzer

# Initialize
config = DetectionConfig()
detector = YOLOv5Detector(config)
analyzer = ResultsAnalyzer(config)

# Single image detection
result = detector.detect_single_image("path/to/image.jpg")
print(f"Found {result.detection_count} conditions")

# Batch detection
results = detector.detect_batch("path/to/images/")
report = analyzer.generate_report(results)
print(report)
```

## ⚙️ Configuration

Edit `config.ini` to customize:

```ini
[detection]
conf_threshold = 0.25      # Confidence threshold
iou_threshold = 0.45       # IoU threshold for NMS
img_size = 640            # Input image size

[paths]
yolov5_dir = d:/AITest/AIXRAY/yolov5
weights_path = ./runs/train/exp/weights/best.pt
test_dir = d:/AITest/AIXRAY/data/images/test

[output]
save_visualizations = true
save_json = true
viz_format = png
```

## 📊 Output Formats

### Detection Report
```
📊 SUMMARY STATISTICS
Total images processed: 100
Images with detections: 85
Detection rate: 85.0%
Total detections: 156

🏷️ DETECTED CONDITIONS
Cardiomegaly            42 (26.9%)
Lung Opacity           28 (17.9%)
Pleural effusion       23 (14.7%)
...
```

### JSON Output
```json
{
  "timestamp": "2025-08-04T16:30:00",
  "total_images": 10,
  "results": [
    {
      "image_name": "xray001.jpg",
      "detection_count": 2,
      "detections": [
        {
          "class_name": "Cardiomegaly",
          "confidence": 0.847,
          "bbox": {"x": 0.5, "y": 0.4, "width": 0.3, "height": 0.2}
        }
      ]
    }
  ]
}
```

### Visualizations
- Class distribution charts
- Confidence histograms
- Detection rate analysis
- Top detected conditions

## 🌐 API Integration

The system includes a FastAPI backend for integration with the web interface.

### Starting the API

```powershell
# Activate your virtual environment
.\.venv\Scripts\Activate.ps1

# Install API requirements
pip install -r api_requirements.txt

# Start the API server
python start_api.py
```

### API Endpoints

- `GET /` - API status check
- `POST /detect/single` - Detect conditions in a single image
- `POST /detect/batch` - Process multiple images in a directory
- `GET /jobs/{job_id}` - Check status of batch processing job

### Frontend Integration

The `integration.js` file provides functions for the frontend to communicate with the API:

```javascript
// Example usage
import { DetectionService } from '../integration';

// Detect a single image
const result = await DetectionService.detectSingleImage(imageFile);

// Start batch processing
const job = await DetectionService.startBatchProcessing(folderPath);

// Check job status
const status = await DetectionService.checkJobStatus(job.job_id);
```

## 🔧 Troubleshooting

### Common Issues

#### PyTorch Compatibility Error
```
_pickle.UnpicklingError: Weights only load failed
```
**Solution**: The refactored code includes fixes for PyTorch 2.7+ compatibility.

#### PosixPath Error on Windows
```
UnsupportedOperation: cannot instantiate 'PosixPath' on your system
```
**Solution**: Cross-platform path handling is implemented in the refactored code.

#### Missing Weights File
```
Weights file not found
```
**Solution**: Ensure your model is trained or download pre-trained weights.

### Debug Mode
```powershell
python detection_cli.py single image.jpg --verbose
```

## 🏗️ Architecture

### Key Components

1. **DetectionConfig**: Configuration management
2. **YOLOv5Detector**: Core detection functionality
3. **ResultsAnalyzer**: Analysis and visualization
4. **Detection/ImageResult**: Data classes for results
5. **CLI Interface**: Command-line interaction

### Design Principles

- **Separation of Concerns**: Each class has a specific responsibility
- **Error Handling**: Comprehensive error handling and logging
- **Type Safety**: Type hints throughout the codebase
- **Configurability**: Easy customization through config files
- **Extensibility**: Easy to add new features or modify existing ones

## 📈 Performance

### Typical Processing Times
- Single image: 2-5 seconds (CPU)
- Batch (100 images): 3-8 minutes (CPU)
- GPU acceleration: 3-5x faster

### Memory Requirements
- Minimum: 4GB RAM
- Recommended: 8GB+ RAM
- GPU: 4GB+ VRAM (optional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the AGPL-3.0 License - see the YOLOv5 license for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for the base detection framework
- Medical imaging community for datasets and research
- PyTorch team for the deep learning framework

## 📞 Support

If you encounter any issues:

1. Check the troubleshooting section
2. Run with `--verbose` flag for detailed logs
3. Check the `detection.log` file
4. Create an issue with detailed error information

---

**Happy detecting! 🏥✨**
