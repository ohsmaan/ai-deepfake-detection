#!/usr/bin/env python3
"""
Diagnostic script for AI Deepfake Detector
"""

import sys
import os
import requests
import json

def test_backend():
    """Test if backend is running and responding"""
    print("🔍 Testing backend...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running")
            return True
        else:
            print(f"❌ Backend returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not running (connection refused)")
        return False
    except Exception as e:
        print(f"❌ Backend test failed: {e}")
        return False

def test_models():
    """Test if models are loaded"""
    print("\n🤖 Testing models...")
    
    try:
        response = requests.get("http://localhost:8001/models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print("✅ Models endpoint working")
            print(f"📊 Image model: {models.get('image_model', {}).get('name', 'Unknown')}")
            print(f"📊 Audio model: {models.get('audio_model', {}).get('name', 'Unknown')}")
            return True
        else:
            print(f"❌ Models endpoint returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models test failed: {e}")
        return False

def test_image_upload():
    """Test image upload functionality"""
    print("\n🖼️ Testing image upload...")
    
    try:
        # Create a simple test image (1x1 pixel)
        from PIL import Image
        import io
        
        # Create a tiny test image
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Upload to backend
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
        response = requests.post("http://localhost:8001/upload", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Image upload working")
            print(f"📊 Result: {result.get('is_deepfake', 'Unknown')} ({result.get('confidence', 0):.2f})")
            return True
        else:
            print(f"❌ Image upload failed: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Image upload test failed: {e}")
        return False

def check_extension_files():
    """Check if extension files exist"""
    print("\n📁 Checking extension files...")
    
    required_files = [
        'extension/manifest.json',
        'extension/content.js',
        'extension/popup.html',
        'extension/popup.js',
        'extension/content.css',
        'extension/popup.css'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def main():
    print("🔧 AI Deepfake Detector Diagnostic")
    print("=" * 40)
    
    # Check extension files
    extension_ok = check_extension_files()
    
    # Test backend
    backend_ok = test_backend()
    
    if backend_ok:
        models_ok = test_models()
        upload_ok = test_image_upload()
    else:
        models_ok = False
        upload_ok = False
    
    # Summary
    print("\n📋 Summary:")
    print(f"Extension files: {'✅ OK' if extension_ok else '❌ ISSUES'}")
    print(f"Backend running: {'✅ OK' if backend_ok else '❌ NOT RUNNING'}")
    print(f"Models loaded: {'✅ OK' if models_ok else '❌ ISSUES'}")
    print(f"Image upload: {'✅ OK' if upload_ok else '❌ ISSUES'}")
    
    if not backend_ok:
        print("\n🚨 To start backend: cd backend && python app.py")
    
    if not extension_ok:
        print("\n🚨 Extension files missing - check your installation")
    
    if backend_ok and not upload_ok:
        print("\n🚨 Backend is running but upload failing - check logs")

if __name__ == "__main__":
    main() 