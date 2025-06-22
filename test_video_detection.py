#!/usr/bin/env python3
"""
Test script for video deepfake detection
Tests with known real and AI videos to understand model behavior
"""

import requests
import time
import json
from PIL import Image
import numpy as np

class VideoDetectionTester:
    def __init__(self, api_url="http://localhost:8001"):
        self.api_url = api_url
        
    def test_api_health(self):
        """Test if the API is running"""
        try:
            response = requests.get(f"{self.api_url}/health")
            if response.status_code == 200:
                print("✅ API is running")
                return True
            else:
                print("❌ API returned error status")
                return False
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            return False
    
    def test_image_detection(self, image_path, description):
        """Test image detection with a single image"""
        try:
            print(f"\n🔍 Testing: {description}")
            
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{self.api_url}/upload", files=files)
            
            if response.status_code == 200:
                result = response.json()
                confidence = result.get('confidence', 0)
                is_deepfake = result.get('is_deepfake', False)
                detection_type = result.get('detection_type', 'unknown')
                
                print(f"   Result: {detection_type}")
                print(f"   Confidence: {confidence:.1%}")
                print(f"   Is Deepfake: {is_deepfake}")
                print(f"   Analysis: {result.get('analysis', 'N/A')}")
                
                return {
                    'description': description,
                    'confidence': confidence,
                    'is_deepfake': is_deepfake,
                    'detection_type': detection_type,
                    'analysis': result.get('analysis', 'N/A')
                }
            else:
                print(f"   ❌ API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            return None
    
    def create_test_images(self):
        """Create simple test images for testing"""
        print("🎨 Creating test images...")
        
        # Create a simple real-looking image (solid color)
        real_img = Image.new('RGB', (224, 224), color='blue')
        real_img.save('test_real.png')
        
        # Create a simple pattern that might trigger AI detection
        pattern_img = Image.new('RGB', (224, 224), color='white')
        pixels = pattern_img.load()
        for i in range(0, 224, 10):
            for j in range(0, 224, 10):
                pixels[i, j] = (255, 0, 0)  # Red dots
        pattern_img.save('test_pattern.png')
        
        print("✅ Test images created: test_real.png, test_pattern.png")
    
    def run_basic_tests(self):
        """Run basic tests with created images"""
        print("\n" + "="*50)
        print("🧪 RUNNING BASIC DETECTION TESTS")
        print("="*50)
        
        # Test 1: Simple blue image (should be real)
        result1 = self.test_image_detection('test_real.png', 'Simple blue image (should be real)')
        
        # Test 2: Pattern image (might trigger AI detection)
        result2 = self.test_image_detection('test_pattern.png', 'Pattern image (might be flagged as AI)')
        
        # Test 3: White image (baseline)
        white_img = Image.new('RGB', (224, 224), color='white')
        white_img.save('test_white.png')
        result3 = self.test_image_detection('test_white.png', 'White image (baseline)')
        
        return [result1, result2, result3]
    
    def analyze_results(self, results):
        """Analyze test results"""
        print("\n" + "="*50)
        print("📊 ANALYSIS OF RESULTS")
        print("="*50)
        
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            print("❌ No valid results to analyze")
            return
        
        print(f"Total tests: {len(valid_results)}")
        
        # Check confidence distribution
        confidences = [r['confidence'] for r in valid_results]
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        
        print(f"Confidence range: {min_confidence:.1%} - {max_confidence:.1%}")
        print(f"Average confidence: {avg_confidence:.1%}")
        
        # Check detection types
        detection_types = {}
        for r in valid_results:
            dt = r['detection_type']
            detection_types[dt] = detection_types.get(dt, 0) + 1
        
        print(f"Detection types: {detection_types}")
        
        # Check if model is being too conservative
        high_confidence_count = sum(1 for c in confidences if c > 0.9)
        print(f"High confidence results (>90%): {high_confidence_count}/{len(valid_results)}")
        
        if high_confidence_count == len(valid_results):
            print("⚠️  WARNING: All results have very high confidence - model might be overconfident")
        
        # Check if all results are the same type
        if len(detection_types) == 1:
            print(f"⚠️  WARNING: All results are {list(detection_types.keys())[0]} - model might not be discriminating")

def main():
    tester = VideoDetectionTester()
    
    # Check API health
    if not tester.test_api_health():
        print("❌ Cannot proceed without API")
        return
    
    # Create test images
    tester.create_test_images()
    
    # Run tests
    results = tester.run_basic_tests()
    
    # Analyze results
    tester.analyze_results(results)
    
    print("\n" + "="*50)
    print("💡 RECOMMENDATIONS")
    print("="*50)
    print("1. If all results have high confidence, the model might be overconfident")
    print("2. If all results are the same type, the model might not be discriminating")
    print("3. Try testing with actual AI-generated images vs real photos")
    print("4. Consider adjusting thresholds in ai_service.py")

if __name__ == "__main__":
    main() 