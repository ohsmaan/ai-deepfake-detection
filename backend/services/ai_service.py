import os
import time
import numpy as np
import torch
import requests
from transformers.pipelines import pipeline
from PIL import Image
import cv2
from typing import Dict, Any, List, Optional
from anthropic import Anthropic
from anthropic.types import TextBlock
from dotenv import load_dotenv
import base64
import io
from transformers import AutoImageProcessor, AutoModelForImageClassification, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# Load environment variables
load_dotenv()

class AIService:
    def __init__(self):
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Use the better performing model
        self.model_name = "haywoodsloan/ai-image-detector-deploy"
        
        # More conservative thresholds to reduce false positives
        self.deepfake_threshold = 0.95  # Very high confidence required for deepfake
        self.ai_generated_threshold = 0.90  # High confidence for AI-generated
        self.suspicious_threshold = 0.80  # Medium confidence for suspicious content
        
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
        
        # Initialize wav2vec2 audio detection model
        print(f"🎵 Loading wav2vec2 audio model: Gustking/wav2vec2-large-xlsr-deepfake-audio-classification")
        self.audio_model_name = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
        self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(self.audio_model_name)
        self.audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(self.audio_model_name)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.audio_model = self.audio_model.to('cuda')
        
        self.audio_model.eval()
        
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

            # More conservative detection logic with better categorization
            if predicted_is_fake and confidence_score > self.deepfake_threshold:
                is_deepfake = True
                detection_type = "deepfake"
                analysis = f"Deepfake detected ({confidence_score:.1%}) - Very high confidence"
            elif predicted_is_fake and confidence_score > self.ai_generated_threshold:
                is_deepfake = False
                detection_type = "ai_generated"
                analysis = f"AI-generated content detected ({confidence_score:.1%}) - High confidence"
            elif predicted_is_fake and confidence_score > self.suspicious_threshold:
                is_deepfake = False
                detection_type = "suspicious"
                analysis = f"Suspicious content ({confidence_score:.1%}) - Medium confidence"
            elif not predicted_is_fake and confidence_score > 0.8:
                detection_type = "real"
                analysis = f"Real image ({confidence_score:.1%}) - High confidence"
            elif not predicted_is_fake and confidence_score > 0.6:
                detection_type = "likely_real"
                analysis = f"Likely real image ({confidence_score:.1%}) - Medium confidence"
            else:
                # Default to uncertain for anything below medium confidence
                detection_type = "uncertain"
                analysis = f"Uncertain - Low confidence ({confidence_score:.1%})"

            # Add confidence level indicators
            if confidence_score < 0.6:
                analysis += " - Very low confidence"
            elif confidence_score < 0.8:
                analysis += " - Low confidence"
            elif confidence_score > 0.95:
                analysis += " - Very high confidence"

            processing_time = time.time() - processing_start
            total_time = time.time() - total_start
            
            # Print timing breakdown
            print(f"⏱️  Timing breakdown:")
            print(f"   Preprocessing: {preprocess_time:.3f}s")
            print(f"   Model inference: {inference_time:.3f}s")
            print(f"   Result processing: {processing_time:.3f}s")
            print(f"   Total time: {total_time:.3f}s")

            model_note = "Conservative AI detection model - high confidence thresholds to reduce false positives."

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
            
            # More conservative aggregation logic
            is_deepfake = False
            detection_type = "real"
            
            # Only flag as deepfake if majority of frames are flagged AND high confidence
            if deepfake_count > len(frame_results) * 0.6 and avg_confidence > 0.85:
                is_deepfake = True
                detection_type = "deepfake"
            elif deepfake_count > len(frame_results) * 0.4 and avg_confidence > 0.8:
                is_deepfake = False
                detection_type = "possible_ai"
            elif deepfake_count == 0 and avg_confidence > 0.7:
                detection_type = "real"
            else:
                detection_type = "uncertain"
            
            return {
                "is_deepfake": is_deepfake,
                "confidence": avg_confidence,
                "frames_analyzed": len(frame_results),
                "deepfake_frames": deepfake_count,
                "model_used": frame_results[0]["model_used"] if frame_results else "none",
                "detection_type": detection_type,
                "analysis": f"Analyzed {len(frame_results)} frames ({deepfake_count} flagged as AI)",
                "model_note": "Conservative video analysis - requires high confidence for deepfake detection"
            }
            
        except Exception as e:
            print(f"Error in video analysis: {e}")
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "frames_analyzed": 0,
                "error": str(e)
            }
    
    async def analyze_with_claude(self, detection_result: Dict[str, Any], file_info: str, image_path: Optional[str] = None) -> str:
        """Use Claude to provide detailed analysis of detection results, including image analysis if available."""
        try:
            # If an image path is provided, use multimodal analysis with Claude
            if image_path:
                # 1. Open the image with Pillow to resize it for efficiency and to meet API limits
                with Image.open(image_path) as img:
                    # Standardize to RGB to avoid issues with transparency when saving as JPEG
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    
                    # Resize the image if it's large to prevent API errors and reduce cost
                    max_dimension = 1024  # A reasonable size for analysis without losing detail
                    if img.width > max_dimension or img.height > max_dimension:
                        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                    
                    # 2. Save the (potentially resized) image to an in-memory buffer as a JPEG
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=90) # Use high-quality JPEG
                    image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    media_type = "image/jpeg"

                # 3. Create the multimodal message for Claude
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": f"""
                                You are a senior forensic image analyst working alongside an automated AI detection system. Your job is to provide expert human-like analysis that complements the automated results.

                                **Automated AI Model Results:**
                                {detection_result}

                                **Your Expert Analysis Task:**
                                The automated model above has analyzed this image. Now you need to provide your expert human-like analysis, considering both the automated results AND your own visual examination.

                                1. **Review the Automated Results:**
                                   - Consider the confidence level and detection type
                                   - Note any potential limitations of the automated model

                                2. **Conduct Your Own Visual Analysis:**
                                   - **Contextual Inconsistencies:** Is the subject in a plausible situation? Are there any impossible or unlikely scenarios?
                                   - **Digital Artifacts:** Look for signs of editing, manipulation, or AI generation
                                   - **Lighting & Shadows:** Check for unnatural lighting patterns or impossible shadows
                                   - **Edge Quality:** Look for blurry edges, artifacts, or signs of compositing
                                   - **Pixel-level Analysis:** Check for compression artifacts, cloning artifacts, or AI fingerprints

                                3. **Provide Your Expert Verdict:**
                                   - **Primary Verdict:** REAL, DIGITALLY ALTERED, or AI-GENERATED
                                   - **Confidence Level:** Low, Medium, High, or Very High
                                   - **Key Evidence:** List 2-3 specific visual indicators that support your verdict
                                   - **Agreement with Automated Model:** Do you agree or disagree with the automated results? Why?

                                4. **Final Recommendation:**
                                   - Should this content be flagged for further review?
                                   - Any specific concerns or red flags?

                                File Info: {file_info}
                                
                                Provide a clear, structured analysis that a non-expert can understand.
                                """
                            },
                        ],
                    }
                ]

                # Use the more powerful Sonnet 4 model for its superior reasoning capabilities.
                model_to_use = "claude-3-5-sonnet-20241022"

                response = self.anthropic.messages.create(
                    model=model_to_use,
                    max_tokens=1024,
                    messages=messages # type: ignore
                )

            else:  # Original text-only analysis for videos
                prompt = f"""
                Analyze this deepfake detection result for a video and provide insights:
                
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
                    model="claude-3-5-haiku-20241022",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
            
            # Extract text from response
            if response.content and len(response.content) > 0:
                content_block = response.content[0]
                if isinstance(content_block, TextBlock):
                    return content_block.text
                else:
                    return str(content_block)
            else:
                return "No response from Claude API"
            
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            return f"Analysis unavailable: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded models"""
        return {
            "image_model": {
                "name": self.model_name,
                "loaded": True,
                "type": "image_classification"
            },
            "audio_model": {
                "name": self.audio_model_name,
                "loaded": True,
                "type": "audio_classification",
                "accuracy": "95% (ASVspoof2019)",
                "model_type": "wav2vec2-large-xlsr",
                "performance": {
                    "f1_score": "0.95",
                    "accuracy": "0.9286",
                    "eer": "0.0401"
                }
            },
            "anthropic_available": bool(os.getenv("ANTHROPIC_API_KEY"))
        }

    def detect_deepfake_audio(self, audio_path: str) -> Dict[str, Any]:
        """Detect deepfake in audio using wav2vec2 model"""
        try:
            import numpy as np
            
            total_start = time.time()
            
            # Load audio file using soundfile (already available)
            try:
                import soundfile as sf
                audio, sr = sf.read(audio_path)
                print(f"🎵 Loaded audio: {len(audio)} samples, {sr}Hz sample rate")
            except Exception as sf_error:
                print(f"🎵 Soundfile failed: {sf_error}")
                # Fallback to pydub
                try:
                    from pydub import AudioSegment
                    audio_segment = AudioSegment.from_file(audio_path)
                    audio = np.array(audio_segment.get_array_of_samples())
                    sr = audio_segment.frame_rate
                    print(f"🎵 Loaded with pydub: {len(audio)} samples, {sr}Hz")
                except Exception as pydub_error:
                    raise Exception(f"Could not load audio file: {sf_error} | {pydub_error}")
            
            preprocess_start = time.time()
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
                print(f"🎵 Converted stereo to mono: {len(audio)} samples")
            
            # Resample to 16kHz if needed using scipy
            if sr != 16000:
                from scipy import signal
                target_length = int(len(audio) * 16000 / sr)
                audio = signal.resample(audio, target_length)
                sr = 16000
                print(f"🎵 Resampled to 16kHz: {len(audio)} samples")
            
            # Normalize audio
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Split audio into segments (optimized for shorter audio from extension)
            segment_length = 5 * sr  # 5 seconds for faster analysis
            segments = []
            
            if len(audio) < segment_length:
                # Pad short audio
                audio = np.pad(audio, (0, segment_length - len(audio)), 'constant')
                segments.append(audio)
                print(f"🎵 Padded short audio to 5 seconds")
            else:
                # Extract overlapping segments for better coverage
                hop_length = 2 * sr  # 2 second overlap
                for i in range(0, len(audio) - segment_length + 1, hop_length):
                    segment = audio[i:i + segment_length]
                    segments.append(segment)
                
                print(f"🎵 Extracted {len(segments)} segments")
            
            if not segments:
                return {
                    "is_deepfake": False,
                    "confidence": 0.0,
                    "segments_analyzed": 0,
                    "error": "No valid audio segments found"
                }
            
            # Analyze each segment with wav2vec2
            segment_results = []
            for i, segment in enumerate(segments):
                try:
                    # Prepare input for wav2vec2
                    inputs = self.audio_processor(
                        segment, 
                        sampling_rate=16000, 
                        return_tensors="pt",
                        padding=True
                    )
                    
                    # Move to GPU if available
                    if torch.cuda.is_available():
                        inputs = {k: v.to('cuda') for k, v in inputs.items()}
                    
                    # Run inference
                    with torch.no_grad():
                        outputs = self.audio_model(**inputs)
                        logits = outputs.logits
                        probabilities = torch.softmax(logits, dim=-1)
                    
                    # Get prediction
                    predicted_class_idx = logits.argmax(-1).item()
                    predicted_label = self.audio_model.config.id2label[predicted_class_idx]
                    confidence_score = probabilities[0][predicted_class_idx].item()
                    
                    # Get all class probabilities for debugging
                    all_probs = probabilities[0].tolist()
                    class_labels = [self.audio_model.config.id2label[i] for i in range(len(all_probs))]
                    
                    print(f"🎵 Segment {i+1}: Predicted: {predicted_label}, Confidence: {confidence_score:.3f}")
                    print(f"🎵 Segment {i+1}: All probabilities: {dict(zip(class_labels, [f'{p:.3f}' for p in all_probs]))}")
                    
                    # Convert label to boolean (fake = True, real = False)
                    # The model uses 'fake' and 'real' labels
                    is_fake = predicted_label.lower() in ['fake', 'deepfake', 'synthetic']
                    
                    segment_results.append({
                        "segment": i + 1,
                        "is_deepfake": is_fake,
                        "confidence": confidence_score,
                        "predicted_label": predicted_label,
                        "all_probabilities": dict(zip(class_labels, all_probs))
                    })
                    
                except Exception as segment_error:
                    print(f"🎵 Error processing segment {i+1}: {segment_error}")
                    segment_results.append({
                        "segment": i + 1,
                        "is_deepfake": False,
                        "confidence": 0.0,
                        "predicted_label": "error",
                        "error": str(segment_error)
                    })
            
            preprocess_time = time.time() - preprocess_start
            
            # Aggregate results
            fake_segments = sum(1 for r in segment_results if r["is_deepfake"])
            avg_confidence = sum(r["confidence"] for r in segment_results) / len(segment_results)
            
            print(f"🎵 Final results: {fake_segments}/{len(segment_results)} fake segments, avg confidence: {avg_confidence:.3f}")
            
            # Get detailed segment info for better analysis
            real_segments = sum(1 for r in segment_results if not r["is_deepfake"])
            fake_confidence = sum(r["confidence"] for r in segment_results if r["is_deepfake"]) / max(fake_segments, 1)
            real_confidence = sum(r["confidence"] for r in segment_results if not r["is_deepfake"]) / max(real_segments, 1)
            
            print(f"🎵 Detailed analysis: {fake_segments} fake ({fake_confidence:.3f}), {real_segments} real ({real_confidence:.3f})")
            
            # Improved detection logic for AI voices
            is_deepfake = False
            detection_type = "uncertain_audio"
            analysis = ""
            
            # More sensitive detection for political/AI voices
            if fake_segments > 0:
                # If any segment is flagged as fake, be more suspicious
                if fake_segments >= len(segment_results) * 0.5:
                    is_deepfake = True
                    if avg_confidence > 0.8:
                        detection_type = "deepfake_audio"
                        analysis = f"Deepfake audio detected ({avg_confidence:.1%}) - multiple segments flagged"
                    else:
                        detection_type = "suspicious_audio"
                        analysis = f"Suspicious audio detected ({avg_confidence:.1%}) - possible AI voice"
                else:
                    # Some segments flagged but not majority
                    detection_type = "suspicious_audio"
                    analysis = f"Suspicious audio ({avg_confidence:.1%}) - mixed signals detected"
            else:
                # No fake segments - but check confidence levels
                if avg_confidence < 0.6:
                    detection_type = "uncertain_audio"
                    analysis = f"Uncertain audio ({avg_confidence:.1%}) - low confidence in real classification"
                elif avg_confidence < 0.8:
                    # Medium confidence "real" - could be sophisticated AI
                    detection_type = "suspicious_audio"
                    analysis = f"Suspicious audio ({avg_confidence:.1%}) - medium confidence suggests possible AI voice"
                else:
                    detection_type = "real_audio"
                    analysis = f"Real audio detected ({avg_confidence:.1%})"
            
            # Add specific warnings for political content
            if detection_type in ["suspicious_audio", "deepfake_audio"]:
                analysis += " - Consider fact-checking political claims"
            
            total_time = time.time() - total_start
            
            return {
                "is_deepfake": is_deepfake,
                "confidence": avg_confidence,
                "processing_time": total_time,
                "model_used": self.audio_model_name,
                "segments_analyzed": len(segment_results),
                "fake_segments": fake_segments,
                "detection_type": detection_type,
                "analysis": analysis,
                "model_note": "Wav2Vec2 audio deepfake detection - state-of-the-art audio analysis",
                "segment_details": segment_results,
                "debug_info": {
                    "preprocessing_time_ms": preprocess_time * 1000,
                    "total_time_ms": total_time * 1000,
                    "model_accuracy": "95% (ASVspoof2019)"
                }
            }
            
        except Exception as e:
            print(f"🎵 Audio analysis error: {e}")
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "processing_time": 0.0,
                "model_used": self.audio_model_name,
                "segments_analyzed": 0,
                "error": str(e),
                "detection_type": "error",
                "analysis": f"Audio analysis error: {str(e)}",
                "model_note": "Wav2Vec2 audio analysis failed"
            } 