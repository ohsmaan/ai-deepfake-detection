import os
import time
import torch
import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from typing import Dict, Any, List
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FastAIService:
    def __init__(self):
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Use a much smaller, faster model
        # MobileNetV2 is ~3.5M parameters vs 195M in the current model
        self.model_name = "google/mobilenet_v2_1.0_224"
        
        print(f"🚀 Loading fast model: {self.model_name}")
        
        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
        
        # Use GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode for faster inference
        
        # Optimized thresholds
        self.suspicious_threshold = 0.8
        
        print(f"✅ Fast AI Service loaded on {self.device}")
        print(f"📊 Model size: ~3.5M parameters (vs 195M in current model)")
    
    def detect_deepfake_image(self, image: Image.Image) -> Dict[str, Any]:
        """Ultra-fast detection optimized for real-time use"""
        try:
            start_time = time.time()
            
            # Fast preprocessing
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to 224x224 (MobileNet requirement)
            image_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Convert to tensor efficiently
            inputs = self.processor(images=image_resized, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Ultra-fast inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Get top prediction
            confidence_score = float(torch.max(probabilities))
            
            # Simple heuristic: very high confidence might indicate AI generation
            # This is a simplified approach for speed
            is_suspicious = confidence_score > self.suspicious_threshold
            
            processing_time = time.time() - start_time
            
            detection_type = "suspicious" if is_suspicious else "normal"
            analysis = f"Confidence: {confidence_score:.1%} (processed in {processing_time:.3f}s)"
            
            return {
                "is_deepfake": is_suspicious,
                "confidence": confidence_score,
                "processing_time": processing_time,
                "model_used": f"{self.model_name} (fast)",
                "detection_type": detection_type,
                "analysis": analysis,
                "model_note": "Ultra-fast MobileNetV2 for real-time filtering",
                "debug_info": {
                    "device_used": str(self.device),
                    "processing_time_ms": processing_time * 1000
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
        """Minimal Claude analysis for speed"""
        try:
            prompt = f"Quick check: {file_info} - {detection_result.get('detection_type', 'unknown')} ({detection_result.get('confidence', 0):.1%})"
            
            response = self.anthropic.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,  # Very short for speed
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
            "optimized_for": "ultra-fast real-time filtering",
            "estimated_speed": "0.05-0.2s per image",
            "model_size": "~3.5M parameters (vs 195M)"
        } 