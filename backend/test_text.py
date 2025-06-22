#!/usr/bin/env python3

import requests
import json

def test_text_detection():
    """Test if text detection is working correctly"""
    print("🧪 Testing text detection with improved 70B model...")
    
    # Better test cases with clearer human vs AI examples
    test_texts = [
        # Clearly human examples
        "Just spilled coffee all over my keyboard. FML. Now I have to clean this mess before my meeting in 10 minutes.",
        "My cat knocked over my plants again. Third time this week. I swear she's doing it on purpose to get attention.",
        "Finally finished that project after pulling an all-nighter. My brain is fried but at least it's done!",
        
        # Clearly AI examples
        "I would be happy to assist you with your inquiry. Please provide additional details so I can better serve your needs.",
        "As an AI language model, I cannot provide personal opinions or make predictions about future events.",
        "Thank you for your message. I understand your concern and will address it promptly. Please allow 24-48 hours for processing.",
        
        # Edge cases
        "The weather is nice today.",
        "I appreciate your patience while we work to resolve this issue. Our team is committed to providing excellent customer service.",
        "OMG that was so funny! 😂😂😂 Can't stop laughing!"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test {i}: {text[:60]}...")
        
        try:
            response = requests.post('http://localhost:8001/detect-text', 
                                   json={'text': text, 'confidence_threshold': 0.5})
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Result: {result['is_bot']} ({result['confidence']:.2f}) - {result['detection_type']}")
                print(f"📊 Analysis: {result['analysis']}")
                if result.get('indicators'):
                    print(f"🔍 Indicators: {', '.join(result['indicators'][:2])}...")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
    
    print("\n🧪 Text detection test complete!")

def test_batch_detection():
    """Test batch text detection"""
    print("\n🧪 Testing batch text detection...")
    
    # Test cases for batch
    batch_texts = [
        "Hey, what's up? Just checking in.",
        "I apologize for the inconvenience. Please allow 2-3 business days for processing your request.",
        "OMG that was so funny! 😂😂😂",
        "Based on the provided information, I can assist you with your inquiry. Please provide additional details.",
        "Just had the best coffee ever at that new place downtown!"
    ]
    
    try:
        response = requests.post('http://localhost:8001/detect-texts', 
                               json=batch_texts)
        
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Batch processed {len(results)} texts:")
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['is_bot']} ({result['confidence']:.2f}) - {result['detection_type']}")
                print(f"     Analysis: {result['analysis'][:100]}...")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Batch request failed: {e}")
    
    print("\n🧪 Batch detection test complete!")

if __name__ == "__main__":
    test_text_detection()
    test_batch_detection() 