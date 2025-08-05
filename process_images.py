import os
import shutil
from pathlib import Path
from datetime import datetime

def find_latest_detection_directory():
    """Find the most recent detection output directory"""
    detect_dir = Path("runs/detect")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = f"runs/detect/detection_{timestamp}"  # Default fallback
    
    if detect_dir.exists():
        # List all directories starting with "detection_" and sort by creation time
        detection_dirs = [d for d in detect_dir.iterdir() if d.is_dir() and d.name.startswith("detection_")]
        
        if detection_dirs:
            # Get the most recent directory
            latest_dir = max(detection_dirs, key=lambda d: d.stat().st_ctime)
            result_dir = str(latest_dir)
            print(f"Found latest detection directory: {result_dir}")
    else:
        print(f"Detect directory doesn't exist, using: {result_dir}")
            
    return result_dir

def find_processed_image(result_dir, original_filename, file_path=None):
    """
    Find and copy the processed image to the results directory
    
    Args:
        result_dir: The directory where processed images are stored
        original_filename: Original filename of the uploaded image
        file_path: Path to the original file (used as fallback if no processed image found)
        
    Returns:
        tuple: (result_url, result_filename)
    """
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
        
    # Find the processed image
    original_name_base = os.path.splitext(original_filename)[0]
    image_candidates = []
    
    # Check if result directory exists
    if os.path.exists(result_dir):
        # Look for any files with the original name (with any extension)
        for filename in os.listdir(result_dir):
            if original_name_base in filename and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_candidates.append(os.path.join(result_dir, filename))
        
        # Also check if the file exists with exact name
        processed_image_path = os.path.join(result_dir, original_filename)
        if os.path.exists(processed_image_path) and processed_image_path not in image_candidates:
            image_candidates.append(processed_image_path)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_filename = f"{timestamp}_{original_filename}"
    public_result_path = os.path.join("results", result_filename)
    
    # If we found processed images, use the first one
    if image_candidates:
        print(f"Found {len(image_candidates)} processed image candidates for {original_filename}")
        processed_image_path = image_candidates[0]
        
        # Copy the processed image
        print(f"Copying processed image from {processed_image_path} to {public_result_path}")
        shutil.copy(processed_image_path, public_result_path)
    elif file_path and os.path.exists(file_path):
        # If no processed image was found, use the original file as the "processed" image
        print(f"No processed image found, copying original from {file_path} to {public_result_path}")
        shutil.copy(file_path, public_result_path)
    else:
        # If we can't find the processed or original image, use the placeholder
        placeholder_path = os.path.join("results", "no_image_found.jpg")
        if os.path.exists(placeholder_path):
            print(f"No image found, using placeholder from {placeholder_path} to {public_result_path}")
            shutil.copy(placeholder_path, public_result_path)
        else:
            print(f"Warning: No placeholder image found at {placeholder_path}. Creating an empty file.")
            # Create an empty file if even the placeholder is missing
            with open(public_result_path, 'w') as f:
                f.write("")
    
    # Return the URL for the frontend
    result_url = f"http://localhost:8000/results/{result_filename}"
    return result_url, result_filename
