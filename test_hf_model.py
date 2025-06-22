#!/usr/bin/env python3
"""
Test script to verify Hugging Face Deep-Fake-Detector-v2-Model performance
"""

import requests
import os
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_hf_model():
    """Test the Hugging Face model directly"""
    
    # Model configuration
    model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    api_token = os.getenv("HUGGINGFACE_TOKEN")
    
    print(f"Testing model: {model_name}")
    print(f"API URL: {api_url}")
    print(f"Token available: {bool(api_token)}")
    print("-" * 50)
    
    # Test with some known images
    test_images = [
        {
            "name": "Hugging Face Logo (Real)",
            "url": "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png"
        },
        {
            "name": "Test Image (Real)",
            "url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=224&h=224&fit=crop"
        }
    ]
    
    headers = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    
    for test in test_images:
        print(f"\nTesting: {test['name']}")
        print(f"URL: {test['url']}")
        
        try:
            # Test with URL
            response = requests.post(
                api_url, 
                headers=headers, 
                json={"inputs": test['url']}
            )
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {result}")
                
                if isinstance(result, list) and len(result) > 0:
                    prediction = result[0]
                    label = prediction.get('label', '')
                    score = prediction.get('score', 0.0)
                    print(f"Label: {label}")
                    print(f"Confidence: {score:.3f} ({score:.1%})")
                else:
                    print(f"Unexpected response format: {result}")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Error testing {test['name']}: {e}")
        
        print("-" * 30)

def test_local_model():
    """Test the local model directly"""
    print("\n" + "="*50)
    print("TESTING LOCAL MODEL")
    print("="*50)
    
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        import torch
        from PIL import Image
        import requests
        from io import BytesIO
        
        model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
        
        print(f"Loading local model: {model_name}")
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        
        print(f"Model labels: {model.config.id2label}")
        print(f"Model loaded successfully!")
        
        # Test with multiple images
        test_images = [
            {
                "name": "Real Person (Unsplash)",
                "url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=224&h=224&fit=crop"
            },
            {
                "name": "Real Person 2 (Unsplash)",
                "url": "https://images.unsplash.com/photo-1494790108755-2616b612b786?w=224&h=224&fit=crop"
            },
            {
                "name": "Real Person 3 (Unsplash)",
                "url": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=224&h=224&fit=crop"
            }
        ]
        
        for test in test_images:
            print(f"\nTesting: {test['name']}")
            print(f"URL: {test['url']}")
            
            try:
                # Download and process image
                response = requests.get(test['url'])
                image = Image.open(BytesIO(response.content))
                image = image.convert('RGB')
                
                print(f"Image size: {image.size}")
                
                # Process with model
                inputs = processor(images=image, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    confidence, predicted_class = torch.max(probabilities, 1)
                
                predicted_label = model.config.id2label[predicted_class.item()]
                confidence_score = confidence.item()
                
                print(f"Results:")
                print(f"  Class: {predicted_class.item()}")
                print(f"  Label: {predicted_label}")
                print(f"  Confidence: {confidence_score:.3f} ({confidence_score:.1%})")
                
                # Check if prediction makes sense
                if predicted_label == "Deepfake" and confidence_score > 0.7:
                    print(f"  ⚠️  WARNING: High confidence deepfake detection on real image!")
                elif predicted_label == "Realism" and confidence_score > 0.7:
                    print(f"  ✅ Good: High confidence real detection")
                else:
                    print(f"  ❓ Uncertain: Low confidence or mixed results")
                    
            except Exception as e:
                print(f"Error testing {test['name']}: {e}")
            
            print("-" * 30)
        
    except Exception as e:
        print(f"Error testing local model: {e}")

if __name__ == "__main__":
    test_hf_model()
    test_local_model() 