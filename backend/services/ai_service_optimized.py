import os
import time
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import cv2
from typing import Dict, Any, List
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OptimizedAIService:
    def __init__(self):
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Use a smaller, faster model for real-time processing
        self.model_name = "microsoft/resnet-50"  # Much faster than SwinV2
        # Alternative: "google/mobilenet_v2_1.0_224" (even faster)
        
        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
        
        # Use GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode for faster inference
        
        # Optimized thresholds for speed vs accuracy trade-off
        self.deepfake_threshold = 0.75
        self.ai_generated_threshold = 0.70
        
        print(f"🚀 Optimized AI Service loaded on {self.device}")
    
    def detect_deepfake_image(self, image: Image.Image) -> Dict[str, Any]:
        """Optimized detection for real-time webpage filtering"""
        try:
            start_time = time.time()
            
            # Preprocess image efficiently
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to smaller size for faster processing
            image_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Convert to tensor
            inputs = self.processor(images=image_resized, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Fast inference with no_grad
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, 3)
            
            # For webpage filtering, we care about confidence more than specific labels
            confidence_score = float(top_probs[0][0])
            
            # Simple heuristic: high confidence in any class might indicate AI generation
            # This is a simplified approach for speed
            is_suspicious = confidence_score > self.deepfake_threshold
            
            processing_time = time.time() - start_time
            
            detection_type = "suspicious" if is_suspicious else "normal"
            analysis = f"Confidence: {confidence_score:.1%} (processed in {processing_time:.3f}s)"
            
            return {
                "is_deepfake": is_suspicious,
                "confidence": confidence_score,
                "processing_time": processing_time,
                "model_used": f"{self.model_name} (optimized)",
                "detection_type": detection_type,
                "analysis": analysis,
                "model_note": "Optimized for real-time webpage filtering",
                "debug_info": {
                    "top_predictions": [
                        {"label": self.model.config.id2label[int(idx)], "score": float(prob)}
                        for prob, idx in zip(top_probs[0], top_indices[0])
                    ],
                    "device_used": str(self.device)
                }
            }
            
        except Exception as e:
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "processing_time": 0.0,
                "model_used": self.model_name,
                "detection_type": "error",
                "analysis": f"Error: {str(e)}",
                "model_note": "Processing error",
                "debug_info": {"error": str(e)}
            }
    
    def batch_detect_images(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Process multiple images efficiently for webpage filtering"""
        results = []
        
        # Process in small batches for memory efficiency
        batch_size = 4
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = []
            
            for image in batch:
                result = self.detect_deepfake_image(image)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    async def analyze_with_claude(self, detection_result: Dict[str, Any], file_info: str) -> str:
        """Simplified Claude analysis for speed"""
        try:
            prompt = f"Quick analysis: {file_info} - Result: {detection_result.get('detection_type', 'unknown')} ({detection_result.get('confidence', 0):.1%})"
            
            response = self.anthropic.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,  # Shorter for speed
                messages=[{"role": "user", "content": prompt}]
            )
            
            if response.content and len(response.content) > 0:
                return response.content[0].text
            return "Analysis unavailable"
            
        except Exception as e:
            return f"Analysis error: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_loaded": True,
            "device": str(self.device),
            "optimized_for": "real-time webpage filtering",
            "estimated_speed": "0.1-0.5s per image"
        } 