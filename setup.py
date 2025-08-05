#!/usr/bin/env python3
"""
YOLOv5 Medical Detection Setup Script
Installs dependencies and validates the environment
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def print_header():
    """Print setup header"""
    print("=" * 60)
    print("🏥 YOLOv5 Medical X-ray Detection Setup")
    print("=" * 60)


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("⚠️  Python 3.8+ is required")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def check_virtual_environment():
    """Check if running in virtual environment"""
    print("\n🔧 Checking virtual environment...")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
        print(f"   Environment: {sys.prefix}")
        return True
    else:
        print("⚠️  No virtual environment detected")
        print("   Recommendation: Use a virtual environment")
        return False


def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    # Core requirements
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0.0",
        "Pillow>=8.3.0"
    ]
    
    failed_packages = []
    
    for package in requirements:
        try:
            print(f"   Installing {package}...")
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', package],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"   ✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}")
            print(f"      Error: {e.stderr.strip()}")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️  Failed to install: {', '.join(failed_packages)}")
        return False
    
    print("\n✅ All packages installed successfully")
    return True


def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    directories = [
        "results",
        "logs",
        "visualizations",
        "exports"
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(exist_ok=True)
            print(f"   ✅ {directory}/")
        except Exception as e:
            print(f"   ❌ Failed to create {directory}/: {e}")
            return False
    
    return True


def validate_yolov5_installation():
    """Validate YOLOv5 installation and fix compatibility issues"""
    print("\n🔍 Validating YOLOv5 installation...")
    
    yolov5_dir = Path("yolov5-bk")
    if not yolov5_dir.exists():
        print("   ❌ YOLOv5 directory not found")
        print("   Please ensure YOLOv5 is properly installed")
        return False
    
    # Check essential files
    essential_files = [
        "detect.py",
        "train.py",
        "val.py",
        "models/experimental.py",
        "models/common.py",
        "utils/general.py"
    ]
    
    missing_files = []
    for file_path in essential_files:
        full_path = yolov5_dir / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"   ❌ Missing YOLOv5 files: {missing_files}")
        return False
    
    print("   ✅ YOLOv5 files validated")
    
    # Check if PyTorch compatibility fixes are applied
    experimental_py = yolov5_dir / "models/experimental.py"
    try:
        with open(experimental_py, 'r') as f:
            content = f.read()
            if "weights_only=False" in content and "platform.system()" in content:
                print("   ✅ PyTorch compatibility fixes applied")
            else:
                print("   ⚠️  PyTorch compatibility fixes may be needed")
    except Exception as e:
        print(f"   ❌ Error checking compatibility fixes: {e}")
        return False
    
    return True


def check_dataset_configuration():
    """Check dataset configuration"""
    print("\n📋 Checking dataset configuration...")
    
    config_file = Path("yolov5-bk/data/vinbigdata.yaml")
    if not config_file.exists():
        print("   ❌ Dataset configuration file not found")
        print("   Expected: yolov5/data/vinbigdata.yaml")
        return False
    
    print("   ✅ Dataset configuration found")
    
    # Check data directories
    data_dirs = [
        "data/images/train",
        "data/images/val", 
        "data/images/test",
        "data/labels/train",
        "data/labels/val",
        "data/labels/test"
    ]
    
    missing_dirs = []
    for data_dir in data_dirs:
        if not Path(data_dir).exists():
            missing_dirs.append(data_dir)
    
    if missing_dirs:
        print(f"   ⚠️  Missing data directories: {missing_dirs}")
        print("   These may be needed for training")
    else:
        print("   ✅ Data directories found")
    
    return True


def check_model_weights():
    """Check if trained model weights exist"""
    print("\n🎯 Checking model weights...")
    
    weights_path = Path("yolov5-bk/runs/train/exp/weights/best.pt")
    if weights_path.exists():
        print("   ✅ Trained model weights found")
        print(f"   Location: {weights_path}")
        return True
    else:
        print("   ⚠️  No trained model weights found")
        print("   You'll need to train a model or download pre-trained weights")
        return False


def run_compatibility_test():
    """Run a quick compatibility test"""
    print("\n🧪 Running compatibility test...")
    
    try:
        # Test imports
        import torch
        import cv2
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        
        print(f"   ✅ PyTorch {torch.__version__}")
        print(f"   ✅ OpenCV {cv2.__version__}")
        print(f"   ✅ NumPy {np.__version__}")
        print(f"   ✅ Pandas {pd.__version__}")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("   ℹ️  CUDA not available (CPU mode)")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False


def main():
    """Main setup function"""
    print_header()
    
    success = True
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Package Installation", install_requirements),
        ("Directory Setup", setup_directories),
        ("YOLOv5 Validation", validate_yolov5_installation),
        ("Dataset Configuration", check_dataset_configuration),
        ("Model Weights", check_model_weights),
        ("Compatibility Test", run_compatibility_test)
    ]
    
    for check_name, check_func in checks:
        try:
            if not check_func():
                success = False
        except Exception as e:
            print(f"   ❌ Error during {check_name}: {e}")
            success = False
    
    # Final status
    print("\n" + "=" * 60)
    if success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Ensure your model is trained or download pre-trained weights")
        print("2. Run: python detection_cli.py single path/to/image.jpg")
        print("3. Or run: python yolo_detector_refactored.py")
    else:
        print("⚠️  Setup completed with warnings")
        print("\nPlease address the issues above before using the detection system")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
