#!/usr/bin/env python3
"""
Detailed test script for the improved deepfake detection system
Tests different detection types and analysis
"""

import requests
import json
from PIL import Image
import io
import numpy as np

# Test configuration
API_BASE = "http://localhost:8001"

def create_test_images():
    """Create different types of test images"""
    images = {}
    
    # 1. Simple gradient (should be real)
    img_real = Image.new('RGB', (384, 384), color='white')
    pixels = img_real.load()
    for i in range(384):
        for j in range(384):
            r = int((i / 384) * 255)
            g = int((j / 384) * 255)
            b = 128
            pixels[i, j] = (r, g, b)
    images['real'] = img_real
    
    # 2. High contrast image (might trigger AI detection)
    img_contrast = Image.new('RGB', (384, 384), color='black')
    pixels = img_contrast.load()
    for i in range(384):
        for j in range(384):
            if (i + j) % 50 < 25:
                pixels[i, j] = (255, 255, 255)
            else:
                pixels[i, j] = (0, 0, 0)
    images['contrast'] = img_contrast
    
    # 3. Pattern image (similar to thumbnails)
    img_pattern = Image.new('RGB', (384, 384), color='blue')
    pixels = img_pattern.load()
    for i in range(384):
        for j in range(384):
            if (i // 20 + j // 20) % 2 == 0:
                pixels[i, j] = (255, 255, 255)
            else:
                pixels[i, j] = (0, 100, 200)
    images['pattern'] = img_pattern
    
    return images

def test_detection(image, name):
    """Test detection on a specific image"""
    print(f"\n🔍 Testing {name} image...")
    
    try:
        # Convert to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Prepare file for upload
        files = {'file': (f'{name}_test.jpg', img_bytes, 'image/jpeg')}
        
        # Upload to API
        response = requests.post(f"{API_BASE}/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {name} detection successful")
            print(f"   Is deepfake: {result.get('is_deepfake')}")
            print(f"   Confidence: {result.get('confidence'):.1%}")
            print(f"   Detection type: {result.get('detection_type', 'unknown')}")
            print(f"   Analysis: {result.get('analysis', 'No analysis')}")
            print(f"   Model note: {result.get('model_note', 'No note')}")
            
            # Show debug info if available
            debug_info = result.get('debug_info', {})
            if debug_info:
                print(f"   Debug - Prediction class: {debug_info.get('prediction_class')}")
                print(f"   Debug - All labels: {debug_info.get('all_labels')}")
                thresholds = debug_info.get('thresholds_used', {})
                print(f"   Debug - Thresholds: {thresholds}")
            
            return result
        else:
            print(f"❌ {name} detection failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ {name} detection error: {e}")
        return None

def main():
    """Run comprehensive detection tests"""
    print("🧪 Testing Improved Deepfake Detection System")
    print("=" * 60)
    
    # Check if backend is running
    try:
        response = requests.get(f"{API_BASE}/health")
        if not response.ok:
            print("❌ Backend not running. Please start the backend first.")
            return
        print("✅ Backend is running")
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return
    
    # Create test images
    print("\n📸 Creating test images...")
    images = create_test_images()
    
    # Test each image
    results = {}
    for name, image in images.items():
        result = test_detection(image, name)
        if result:
            results[name] = result
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Detection Summary")
    print("=" * 60)
    
    for name, result in results.items():
        detection_type = result.get('detection_type', 'unknown')
        confidence = result.get('confidence', 0)
        is_deepfake = result.get('is_deepfake', False)
        
        status_icon = "🤖" if is_deepfake else "✅"
        print(f"{status_icon} {name:10} | {detection_type:15} | {confidence:5.1%} | {is_deepfake}")
    
    # Analysis
    print("\n💡 Analysis:")
    print("- Real images should be classified as 'real' with high confidence")
    print("- Pattern/contrast images might be flagged as 'possible_ai'")
    print("- Only high-confidence detections should be marked as 'deepfake'")
    print("- Low confidence results should be marked as 'uncertain'")

if __name__ == "__main__":
    main() 