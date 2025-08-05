#!/usr/bin/env python3
"""
Start the simple API server
"""

if __name__ == "__main__":
    print("🚀 Starting Simple Medical X-ray Detection API...")
    print("📍 API will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    print("🛑 Press Ctrl+C to stop the server")
    
    try:
        import uvicorn
        uvicorn.run("simple_api:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("❌ uvicorn not installed. Install with: pip install uvicorn")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
