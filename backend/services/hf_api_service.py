import os
import requests
import base64
from typing import Dict, Any
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

class HuggingFaceAPIService:
    def __init__(self):
        self.api_token = os.getenv("HUGGINGFACE_TOKEN")
        self.api_url = "https://api-inference.huggingface.co/models"
        
        # Popular deepfake detection models
        self.models = {
            "deepfake": "aiwithoutborders-xyz/CommunityForensics-DeepfakeDet-ViT",  # State-of-the-art deepfake detection
           # "dfdc": "selimsef/dfdc_deepfake_detection",  # Alternative model
            #"face_forensics": "microsoft/DialoGPT-medium",  # Placeholder
           # "custom": "your-model-name/your-model"  # Your custom model
        }
    
    def _encode_image(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def detect_deepfake_api(self, image: Image.Image, model_name: str = "deepfake") -> Dict[str, Any]:
        """Use Hugging Face Inference API for deepfake detection"""
        
        if not self.api_token:
            return {
                "is_deepfake": False,
                "confidence": 0.5,
                "model_used": "no_token",
                "error": "Hugging Face token not found"
            }
        
        try:
            model_id = self.models.get(model_name, self.models["deepfake"])
            url = f"{self.api_url}/{model_id}"
            
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            # Encode image
            img_base64 = self._encode_image(image)
            
            payload = {
                "inputs": {
                    "image": img_base64
                }
            }
            
            # Make API call
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Parse the result (format depends on the model)
                if isinstance(result, list) and len(result) > 0:
                    prediction = result[0]
                    
                    # Handle different response formats
                    if "label" in prediction:
                        is_deepfake = prediction["label"] == "FAKE" or prediction["label"] == "1"
                        confidence = prediction.get("score", 0.5)
                    elif "score" in prediction:
                        confidence = prediction["score"]
                        is_deepfake = confidence > 0.5
                    else:
                        is_deepfake = False
                        confidence = 0.5
                    
                    return {
                        "is_deepfake": is_deepfake,
                        "confidence": confidence,
                        "model_used": model_id,
                        "raw_result": result
                    }
                else:
                    return {
                        "is_deepfake": False,
                        "confidence": 0.5,
                        "model_used": model_id,
                        "error": "Unexpected response format"
                    }
            else:
                return {
                    "is_deepfake": False,
                    "confidence": 0.5,
                    "model_used": model_id,
                    "error": f"API error: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "is_deepfake": False,
                "confidence": 0.5,
                "model_used": "error",
                "error": str(e)
            }
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models"""
        return {
            "models": self.models,
            "api_token_configured": bool(self.api_token),
            "api_url": self.api_url
        } 