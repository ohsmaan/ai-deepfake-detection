#!/usr/bin/env python3
"""
Test script for the Chrome extension functionality
Tests the API endpoints that the extension will use
"""

import requests
import json
from PIL import Image
import io
import base64

# Test configuration
API_BASE = "http://localhost:8001"

def test_health_endpoint():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            print("✅ Health endpoint working")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\n🔍 Testing model info endpoint...")
    try:
        response = requests.get(f"{API_BASE}/models")
        if response.status_code == 200:
            print("✅ Model info endpoint working")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Model info endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info endpoint error: {e}")
        return False

def create_test_image():
    """Create a simple test image"""
    # Create a 384x384 test image with a gradient
    img = Image.new('RGB', (384, 384), color='white')
    pixels = img.load()
    
    # Add a simple gradient
    for i in range(384):
        for j in range(384):
            r = int((i / 384) * 255)
            g = int((j / 384) * 255)
            b = 128
            pixels[i, j] = (r, g, b)
    
    return img

def test_image_upload():
    """Test image upload endpoint"""
    print("\n🔍 Testing image upload endpoint...")
    try:
        # Create test image
        img = create_test_image()
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Prepare file for upload
        files = {'file': ('test_image.jpg', img_bytes, 'image/jpeg')}
        
        # Upload to API
        response = requests.post(f"{API_BASE}/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Image upload working")
            print(f"   Is deepfake: {result.get('is_deepfake')}")
            print(f"   Confidence: {result.get('confidence')}")
            print(f"   Processing time: {result.get('processing_time')}")
            print(f"   Model used: {result.get('model_used')}")
            return True
        else:
            print(f"❌ Image upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Image upload error: {e}")
        return False

def test_cors_headers():
    """Test CORS headers"""
    print("\n🔍 Testing CORS headers...")
    try:
        response = requests.options(f"{API_BASE}/upload")
        cors_headers = response.headers.get('Access-Control-Allow-Origin')
        
        if cors_headers:
            print("✅ CORS headers present")
            print(f"   Allow-Origin: {cors_headers}")
            return True
        else:
            print("❌ CORS headers missing")
            return False
    except Exception as e:
        print(f"❌ CORS test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Chrome Extension API Integration")
    print("=" * 50)
    
    tests = [
        test_health_endpoint,
        test_model_info,
        test_cors_headers,
        test_image_upload
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Extension should work correctly.")
        print("\n📋 Next steps:")
        print("1. Load the extension in Chrome (chrome://extensions/)")
        print("2. Enable Developer mode")
        print("3. Click 'Load unpacked' and select the 'extension' folder")
        print("4. Test on a webpage with images")
    else:
        print("⚠️  Some tests failed. Check the backend configuration.")
    
    return passed == total

if __name__ == "__main__":
    main() 