import os
import time
import numpy as np
import torch
import requests
from transformers.pipelines import pipeline
from PIL import Image
import cv2
from typing import Dict, Any, List
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AIService:
    def __init__(self):
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Use the better performing model
        self.model_name = "haywoodsloan/ai-image-detector-deploy"
        
        # Simplified thresholds for better detection
        self.deepfake_threshold = 0.85  # Much higher threshold since model is too aggressive
        self.ai_generated_threshold = 0.80  # Higher threshold for AI detection
        
        # Use the pipeline for inference with safe optimizations
        device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
        print(f"🚀 Loading local model: {self.model_name} on device {device}")
        
        # Safe optimizations that won't crash your system
        self.pipe = pipeline(
            "image-classification", 
            model=self.model_name, 
            device=device,
            model_kwargs={"low_cpu_mem_usage": True}  # Safe memory optimization
        )
        
        # Warm up the model with a dummy inference
        print("🔥 Warming up model...")
        dummy_image = Image.new('RGB', (224, 224), color='white')
        _ = self.pipe(dummy_image)
        print("✅ Model ready!")
    
    def detect_deepfake_image(self, image: Image.Image) -> Dict[str, Any]:
        try:
            total_start = time.time()
            
            # Time image preprocessing
            preprocess_start = time.time()
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
            preprocess_time = time.time() - preprocess_start
            
            # Time model inference
            inference_start = time.time()
            result = self.pipe(image_resized)
            inference_time = time.time() - inference_start
            
            # Time result processing
            processing_start = time.time()
            
            # Handle pipeline result safely
            if isinstance(result, list):
                predictions = result
            elif hasattr(result, '__iter__'):
                predictions = list(result)  # type: ignore
            else:
                predictions = [result]
            
            if not predictions or not isinstance(predictions[0], dict):
                return {
                    "is_deepfake": False,
                    "confidence": 0.0,
                    "model_used": self.model_name,
                    "predicted_label": "",
                    "detection_type": "error",
                    "analysis": "Pipeline returned no result or unexpected format",
                    "model_note": "Pipeline error.",
                    "debug_info": {"pipeline_result": str(result)}
                }

            # Aggregate scores for each label
            label_scores = {}
            for pred in predictions:
                if isinstance(pred, dict):
                    label = str(pred.get('label', '')).lower()
                    score = float(pred.get('score', 0.0))
                    label_scores[label] = score

            # Get the highest scoring label
            if label_scores:
                predicted_label = max(label_scores.keys(), key=lambda k: label_scores[k])
                confidence_score = label_scores[predicted_label]
                predicted_is_fake = predicted_label == 'artificial'
            else:
                predicted_label = 'unknown'
                confidence_score = 0.0
                predicted_is_fake = False

            is_deepfake = False
            detection_type = "real"
            analysis = ""

            if predicted_is_fake and confidence_score > self.deepfake_threshold:
                is_deepfake = True
                detection_type = "deepfake"
                analysis = f"Deepfake detected ({confidence_score:.1%})"
            elif predicted_is_fake and confidence_score > self.ai_generated_threshold:
                is_deepfake = False
                detection_type = "possible_ai"
                analysis = f"Possible AI-generated content ({confidence_score:.1%})"
            elif not predicted_is_fake and confidence_score > 0.6:
                detection_type = "real"
                analysis = f"Real image ({confidence_score:.1%})"
            else:
                detection_type = "uncertain"
                analysis = f"Uncertain ({confidence_score:.1%})"

            if confidence_score < 0.5:
                analysis += " - Very low confidence"
            elif confidence_score > 0.9:
                analysis += " - Very high confidence"

            processing_time = time.time() - processing_start
            total_time = time.time() - total_start
            
            # Print timing breakdown
            print(f"⏱️  Timing breakdown:")
            print(f"   Preprocessing: {preprocess_time:.3f}s")
            print(f"   Model inference: {inference_time:.3f}s")
            print(f"   Result processing: {processing_time:.3f}s")
            print(f"   Total time: {total_time:.3f}s")

            model_note = "This model detects AI-generated images with 97.9% accuracy using local SwinV2 architecture."

            return {
                "is_deepfake": is_deepfake,
                "confidence": confidence_score,
                "processing_time": total_time,
                "model_used": self.model_name,
                "predicted_label": predicted_label,
                "detection_type": detection_type,
                "analysis": analysis,
                "model_note": model_note,
                "debug_info": {
                    "label_scores": label_scores,
                    "thresholds_used": {
                        "deepfake_threshold": self.deepfake_threshold,
                        "ai_generated_threshold": self.ai_generated_threshold
                    },
                    "timing_breakdown": {
                        "preprocessing_ms": preprocess_time * 1000,
                        "inference_ms": inference_time * 1000,
                        "processing_ms": processing_time * 1000,
                        "total_ms": total_time * 1000
                    }
                }
            }
        except Exception as e:
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "processing_time": 0.0,
                "model_used": self.model_name,
                "predicted_label": "",
                "detection_type": "error",
                "analysis": f"Pipeline error: {str(e)}",
                "model_note": "Pipeline error.",
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
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for Deep-Fake-Detector-v2-Model"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to 224x224 (model requirement for ViT-base-patch16-224)
        image_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        return image_resized

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
            "model_loaded": True,
            "processor_loaded": False,
            "anthropic_available": bool(os.getenv("ANTHROPIC_API_KEY"))
        } 