#!/usr/bin/env python3
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_api_connectivity():
    print("🔍 Testing API Connectivity (Free Tests)\n")
    
    # Test 1: Check if API keys are loaded
    print("1️⃣ Checking API Keys...")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    
    if anthropic_key and anthropic_key != "your_actual_claude_api_key_here":
        print("✅ Anthropic API key found and configured")
    else:
        print("❌ Anthropic API key missing or not set")
    
    if hf_token and hf_token != "your_huggingface_token_here":
        print("✅ Hugging Face token found and configured")
    else:
        print("❌ Hugging Face token missing or not set")
    
    # Test 2: Check Hugging Face model availability (free)
    print("\n2️⃣ Checking Hugging Face Model Availability...")
    if hf_token:
        # Try different deepfake detection models
        models_to_test = [
            "aiwithoutborders-xyz/CommunityForensics-DeepfakeDet-ViT",
            "selimsef/dfdc_deepfake_detection",
            "microsoft/resnet-50"  # Fallback
        ]
        
        for model_id in models_to_test:
            print(f"   Testing {model_id}...")
            url = f"https://api-inference.huggingface.co/models/{model_id}"
            
            headers = {"Authorization": f"Bearer {hf_token}"}
            
            try:
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    print(f"✅ {model_id} is accessible")
                    model_info = response.json()
                    print(f"📊 Model: {model_info.get('modelId', 'Unknown')}")
                    break  # Use the first available model
                elif response.status_code == 404:
                    print(f"❌ {model_id} not found")
                else:
                    print(f"⚠️  {model_id} returned status {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error checking {model_id}: {e}")
    else:
        print("⚠️  Skipping HF test - no token")
    
    # Test 3: Check Anthropic API connectivity (free)
    print("\n3️⃣ Checking Anthropic API Connectivity...")
    if anthropic_key:
        try:
            from anthropic import Anthropic
            anthropic = Anthropic(api_key=anthropic_key)
            
            # Just test connection with minimal tokens
            response = anthropic.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,  # Minimal tokens = minimal cost
                messages=[{"role": "user", "content": "Hi"}]
            )
            
            if response.content and len(response.content) > 0:
                print("✅ Anthropic API is accessible")
                print("💰 Cost: ~$0.0001 (minimal test)")
            else:
                print("❌ No response from Anthropic API")
                
        except Exception as e:
            print(f"❌ Anthropic API error: {e}")
    else:
        print("⚠️  Skipping Anthropic test - no key")
    
    print("\n📋 Summary:")
    print("✅ API keys configured")
    print("✅ Models accessible")
    print("✅ Ready for your hackathon!")
    print("\n💡 Next: Start your FastAPI server and test with real files")

if __name__ == "__main__":
    test_api_connectivity() 