#!/usr/bin/env python3
"""
Test the FastAPI endpoints
"""

import requests
import json

# Test the root endpoint
def test_root():
    try:
        response = requests.get("http://localhost:8000/")
        print(f"Root endpoint status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Root endpoint test failed: {e}")
        return False

# Test health endpoint
def test_health():
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"Health endpoint status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health endpoint test failed: {e}")
        return False

# Test detection endpoint
def test_detection():
    try:
        test_image_path = r"D:\AITest\AIXRAY\data\images\test\0a1aef5326b7b24378c6692f7a454e52.jpg"
        
        with open(test_image_path, 'rb') as f:
            files = {'file': ('test_image.jpg', f, 'image/jpeg')}
            response = requests.post("http://localhost:8000/detect/single", files=files)
        
        print(f"Detection endpoint status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Detection count: {result['detection_count']}")
            print(f"Processing time: {result['processing_time']:.2f}s")
            print("✅ Detection endpoint working!")
            return True
        else:
            print(f"Error response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Detection endpoint test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing API endpoints...")
    
    print("\n1. Testing root endpoint:")
    root_ok = test_root()
    
    print("\n2. Testing health endpoint:")
    health_ok = test_health()
    
    print("\n3. Testing detection endpoint:")
    detection_ok = test_detection()
    
    print(f"\n📊 Test Results:")
    print(f"Root: {'✅' if root_ok else '❌'}")
    print(f"Health: {'✅' if health_ok else '❌'}")
    print(f"Detection: {'✅' if detection_ok else '❌'}")
    
    if all([root_ok, health_ok, detection_ok]):
        print("\n🎉 All tests passed! API is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the API server.")
