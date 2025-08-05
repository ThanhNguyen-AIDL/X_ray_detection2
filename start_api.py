#!/usr/bin/env python3
"""
Script to start the FastAPI service for the X-ray detection system
"""
import os
import sys
import subprocess
import argparse

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
    except ImportError:
        print("Installing required dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "api_requirements.txt"])

def start_api(host="0.0.0.0", port=8000):
    """Start the FastAPI server"""
    print(f"Starting API server on http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    subprocess.run([sys.executable, "-m", "uvicorn", "api:app", "--host", host, "--port", str(port)])

def main():
    parser = argparse.ArgumentParser(description="Start the X-ray detection API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the API server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the API server")
    args = parser.parse_args()

    check_dependencies()
    start_api(args.host, args.port)

if __name__ == "__main__":
    main()
