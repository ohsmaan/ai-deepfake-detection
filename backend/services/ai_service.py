import os
import time
from typing import Dict, Any, Optional
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers.pipelines import pipeline
from PIL import Image
import cv2
import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv

# Try different import paths for the HF API service
try:
    from services.hf_api_service import HuggingFaceAPIService
except ImportError:
    try:
        from .hf_api_service import HuggingFaceAPIService
    except ImportError:
        from hf_api_service import HuggingFaceAPIService

# Load environment variables
load_dotenv()

class AIService:
    def __init__(self):
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.hf_api = HuggingFaceAPIService()
        
        # Use pipeline for local image classification
        self.model_name = "aiwithoutborders-xyz/CommunityForensics-DeepfakeDet-ViT"  # Back to your preferred model
        
        # Create pipeline with explicit preprocessing
        processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.pipe = pipeline(
            "image-classification", 
            model=self.model_name,
            feature_extractor=processor
        )
    
    def _load_model(self):
        """Load the deepfake detection model"""
        try:
            print(f"Loading model: {self.model_name}")
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to mock model
            self.model = None
    
    def extract_frames(self, video_path: str, max_frames: int = 10) -> list:
        """Extract frames from video for analysis"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        return frames
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for CommunityForensics model"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to 384x384 (model requirement)
        image_resized = image.resize((384, 384), Image.Resampling.LANCZOS)
        
        return image_resized

    def detect_deepfake_image(self, image: Image.Image) -> Dict[str, Any]:
        """Detect deepfake in a single image using direct model inference"""
        try:
            print(f"Original image size: {image.size}")
            
            # Preprocess image properly
            processed_image = self._preprocess_image(image)
            print(f"Processed image size: {processed_image.size}")
            
            # Use direct model inference instead of pipeline
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            # Load processor and model directly
            processor = AutoImageProcessor.from_pretrained(self.model_name)
            model = AutoModelForImageClassification.from_pretrained(self.model_name)
            
            # Fix processor size to match model expectations
            processor.size = {'height': 384, 'width': 384}
            processor.crop_size = 384
            
            # Process image with our preprocessor
            inputs = processor(images=processed_image, return_tensors="pt")
            
            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence, predicted_class = torch.max(probabilities, 1)
            
            # Get label from model config
            predicted_label = model.config.id2label[predicted_class.item()]
            confidence_score = confidence.item()
            
            print(f"🔍 DEBUG: Raw prediction class: {predicted_class.item()}")
            print(f"🔍 DEBUG: Predicted label: '{predicted_label}'")
            print(f"🔍 DEBUG: Confidence score: {confidence_score}")
            print(f"🔍 DEBUG: All possible labels: {model.config.id2label}")
            
            # More aggressive detection logic - check if it's a binary classification
            if len(model.config.id2label) == 2:
                # Binary classification (0=REAL, 1=FAKE or similar)
                is_deepfake = predicted_class.item() == 1  # Class 1 is usually fake
                print(f"🔍 DEBUG: Binary classification detected, class 1 = fake: {is_deepfake}")
            else:
                # Multi-class or different format
                is_deepfake = (
                    predicted_label.upper() in ["FAKE", "DEEPFAKE", "1", "GENERATED", "AI_GENERATED"] or
                    confidence_score > 0.5  # Any confidence above 50%
                )
                print(f"🔍 DEBUG: Multi-class classification, is_deepfake: {is_deepfake}")
            
            # Add analysis based on confidence
            if confidence_score > 0.8:
                analysis = "High confidence prediction"
            elif confidence_score > 0.6:
                analysis = "Moderate confidence prediction"
            else:
                analysis = "Low confidence prediction - may need additional analysis"
            
            # Add specific analysis for false positives
            if is_deepfake and confidence_score < 0.6:
                analysis += " - Low confidence fake detection, possible false positive"
            elif not is_deepfake and confidence_score > 0.8:
                analysis += " - High confidence real detection"
            
            print(f"Model prediction: {predicted_label}, confidence: {confidence_score}")
            print(f"Analysis: {analysis}")
            
            return {
                "is_deepfake": is_deepfake,
                "confidence": confidence_score,
                "model_used": self.model_name,
                "predicted_label": predicted_label,
                "analysis": analysis,
                "debug_info": {
                    "prediction_class": predicted_class.item(),
                    "all_labels": model.config.id2label,
                    "binary_classification": len(model.config.id2label) == 2
                },
                "note": "This model is specialized for human face deepfakes. For AI-generated animals/objects, consider using a general AI detection model."
            }
            
        except Exception as e:
            print(f"Error in direct model inference: {e}")
            return {
                "is_deepfake": False,
                "confidence": 0.5,
                "model_used": self.model_name,
                "error": str(e)
            }
    
    def detect_deepfake_video(self, video_path: str) -> Dict[str, Any]:
        """Detect deepfake in video by analyzing multiple frames"""
        try:
            frames = self.extract_frames(video_path)
            
            if not frames:
                return {
                    "is_deepfake": False,
                    "confidence": 0.0,
                    "frames_analyzed": 0,
                    "error": "No frames extracted"
                }
            
            # Analyze each frame
            frame_results = []
            for frame in frames:
                image = Image.fromarray(frame)
                result = self.detect_deepfake_image(image)
                frame_results.append(result)
            
            # Aggregate results
            deepfake_count = sum(1 for r in frame_results if r["is_deepfake"])
            avg_confidence = sum(r["confidence"] for r in frame_results) / len(frame_results)
            
            # Determine overall result (simple majority voting)
            is_deepfake = deepfake_count > len(frame_results) / 2
            
            return {
                "is_deepfake": is_deepfake,
                "confidence": avg_confidence,
                "frames_analyzed": len(frame_results),
                "deepfake_frames": deepfake_count,
                "model_used": frame_results[0]["model_used"] if frame_results else "none"
            }
            
        except Exception as e:
            print(f"Error in video analysis: {e}")
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "frames_analyzed": 0,
                "error": str(e)
            }
    
    async def analyze_with_claude(self, detection_result: Dict[str, Any], file_info: str) -> str:
        """Use Claude to provide detailed analysis of detection results"""
        try:
            prompt = f"""
            Analyze this deepfake detection result and provide insights:
            
            File: {file_info}
            Detection Result: {detection_result}
            
            Please provide:
            1. A summary of the detection
            2. Confidence level interpretation
            3. Potential reasons for the result
            4. Recommendations for further analysis if needed
            
            Keep the response concise and professional.
            """
            
            response = self.anthropic.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract text from response
            if response.content and len(response.content) > 0:
                content_block = response.content[0]
                if hasattr(content_block, 'text'):
                    return content_block.text  # type: ignore
                else:
                    return str(content_block)
            else:
                return "No response from Claude API"
            
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            return f"Analysis unavailable: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "pipeline_loaded": self.pipe is not None,
            "anthropic_available": bool(os.getenv("ANTHROPIC_API_KEY")),
            "pipeline_type": "image-classification"
        } 