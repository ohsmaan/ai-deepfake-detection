#!/usr/bin/env python3
"""
Test script for wav2vec2 audio deepfake detection
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from services.ai_service import AIService

def test_wav2vec2_loading():
    """Test if the wav2vec2 model loads correctly"""
    print("🧪 Testing wav2vec2 model loading...")
    
    try:
        # Initialize AI service
        ai_service = AIService()
        print("✅ AI service initialized successfully")
        
        # Check model info
        model_info = ai_service.get_model_info()
        print(f"📊 Audio model: {model_info['audio_model']['name']}")
        print(f"📊 Model type: {model_info['audio_model']['model_type']}")
        print(f"📊 Accuracy: {model_info['audio_model']['accuracy']}")
        
        print("✅ Wav2Vec2 model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading wav2vec2 model: {e}")
        return False

if __name__ == "__main__":
    success = test_wav2vec2_loading()
    sys.exit(0 if success else 1) 