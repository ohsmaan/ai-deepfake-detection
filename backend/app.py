from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import uvicorn
from typing import Optional, List
import time
import os
import aiofiles
from PIL import Image
import pillow_avif  # Ensure AVIF plugin is loaded
from services.ai_service import AIService
from services.text_service import TextService  # RESTORED - Groq enabled
import io
import uuid
import requests
from transformers.pipelines import pipeline
import cv2
from anthropic import Anthropic
from anthropic.types import TextBlock
from dotenv import load_dotenv
import base64
from transformers import AutoImageProcessor, AutoModelForImageClassification, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# Initialize AI service with accurate model
ai_service = AIService()
text_service = TextService()  # RESTORED - Groq enabled

# Pydantic models for request/response
class DetectionRequest(BaseModel):
    confidence_threshold: float = 0.5

class DetectionResponse(BaseModel):
    is_deepfake: bool
    confidence: float
    processing_time: float
    message: str
    model_used: str
    claude_analysis: str
    detection_type: str = "unknown"
    analysis: str = ""
    model_note: str = ""

# RESTORED - Groq text detection enabled
class TextDetectionRequest(BaseModel):
    text: str
    confidence_threshold: float = 0.5

class TextDetectionResponse(BaseModel):
    is_bot: bool
    confidence: float
    processing_time: float
    message: str
    model_used: str
    detection_type: str = "human"
    analysis: str = ""
    indicators: List[str] = []
    text_length: int = 0

# FastAPI app configuration
app = FastAPI(
    title="AI Deepfake Detector",
    description="API for detecting deepfake videos and images",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add OPTIONS endpoint for CORS preflight
@app.options("/upload")
async def options_upload():
    """Handle CORS preflight for upload endpoint"""
    return {"message": "OK"}

# Global storage for demo purposes (use database in production)
detection_results = {}
active_requests = set()

@app.get("/")
def root():
    return {"message": "AI Deepfake Detector API is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/proxy-image")
async def proxy_image(url: str):
    """Proxy endpoint to fetch images from external URLs, bypassing CORS"""
    try:
        print(f"🔄 Proxying image: {url}")
        
        # Fetch the image from the external URL
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Get the content type
        content_type = response.headers.get('content-type', 'image/jpeg')
        
        print(f"✅ Image proxied successfully: {len(response.content)} bytes, {content_type}")
        
        # Return the image data
        return Response(
            content=response.content,
            media_type=content_type,
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
        )
        
    except Exception as e:
        print(f"❌ Proxy error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to proxy image: {str(e)}")

@app.post("/upload", response_model=DetectionResponse)
async def upload_file(
    file: UploadFile = File(...)
):
    """Upload a video or image file for deepfake detection"""
    
    print(f"📥 Received upload request: {file.filename}, content_type: {file.content_type}, size: {file.size}")
    
    # Validate file type
    allowed_types = ["video/", "image/"]
    if not file.content_type or not any(file.content_type.startswith(t) for t in allowed_types):
        print(f"❌ Invalid file type: {file.content_type}")
        raise HTTPException(
            status_code=400, 
            detail="File must be a video or image"
        )
    
    # Validate file size (50MB limit for demo)
    if file.size and file.size > 50 * 1024 * 1024:
        print(f"❌ File too large: {file.size} bytes")
        raise HTTPException(
            status_code=400, 
            detail="File size must be less than 50MB"
        )
    
    # Generate unique ID for this detection
    detection_id = f"detection_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    # Track active requests
    active_requests.add(detection_id)
    print(f"🚀 Starting request {detection_id} (active: {len(active_requests)})")
    
    # Save uploaded file temporarily
    temp_file_path = f"/tmp/{detection_id}_{file.filename}"
    
    print(f"📁 Processing file: {file.filename} ({file.content_type})")
    
    # Save file
    async with aiofiles.open(temp_file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    print(f"💾 File saved: {temp_file_path} ({len(content)} bytes)")
    
    # Check if file is actually AVIF regardless of content_type or filename
    def is_avif_file(file_path):
        try:
            with open(file_path, 'rb') as f:
                # Check for AVIF signature (first 12 bytes)
                header = f.read(12)
                # AVIF files start with specific bytes
                return header.startswith(b'\x00\x00\x00\x20ftyp') and b'avif' in header
        except:
            return False
    
    # Check if file is actually a GIF
    def is_gif_file(file_path):
        try:
            with open(file_path, 'rb') as f:
                # Check for GIF signature (first 6 bytes)
                header = f.read(6)
                # GIF files start with GIF87a or GIF89a
                return header.startswith(b'GIF87a') or header.startswith(b'GIF89a')
        except:
            return False
    
    # Detect if file is actually AVIF or GIF
    actual_is_avif = is_avif_file(temp_file_path)
    actual_is_gif = is_gif_file(temp_file_path)
    print(f"🔍 File analysis: content_type={file.content_type}, actual_avif={actual_is_avif}, actual_gif={actual_is_gif}")
    
    # Use actual detection if content_type is unreliable
    should_process_as_avif = file.content_type == "image/avif" or actual_is_avif
    should_process_as_gif = file.content_type == "image/gif" or actual_is_gif
    
    # Verify file exists and has correct size
    if os.path.exists(temp_file_path):
        file_size = os.path.getsize(temp_file_path)
        print(f"✅ File verified: {temp_file_path} exists, size: {file_size} bytes")
    else:
        print(f"❌ File not found after saving: {temp_file_path}")
        active_requests.remove(detection_id)
        raise HTTPException(status_code=500, detail="File was not saved properly")
    
    # Start AI processing
    start_time = time.time()
    
    # Detect deepfake based on file type
    if file.content_type.startswith("video/"):
        print("🎥 Processing as video")
        detection_result = ai_service.detect_deepfake_video(temp_file_path)
    else:
        print("🖼️ Processing as image")
        try:
            # Handle AVIF images by converting to JPEG
            if should_process_as_avif:
                print("🔄 Processing AVIF image...")
                try:
                    # Read the file content into memory first
                    with open(temp_file_path, 'rb') as f:
                        image_data = f.read()
                    
                    print(f"📄 Read {len(image_data)} bytes from AVIF file")
                    
                    # First try: Use BytesIO to load the image from memory
                    try:
                        image_buffer = io.BytesIO(image_data)
                        print(f"🔄 Created BytesIO buffer")
                        
                        # Load image directly from BytesIO
                        image = Image.open(image_buffer)
                        print(f"🖼️ AVIF image loaded: {image.size} {image.mode}")
                        
                        # Convert to RGB if needed
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                            print(f"🔄 Converted to RGB: {image.size} {image.mode}")
                        
                        detection_result = ai_service.detect_deepfake_image(image)
                        
                    except Exception as bytesio_error:
                        print(f"❌ BytesIO approach failed: {bytesio_error}")
                        
                        # Second try: Save as proper AVIF file with correct extension
                        try:
                            avif_temp_path = temp_file_path.replace('.jpg', '.avif')
                            if not avif_temp_path.endswith('.avif'):
                                avif_temp_path = temp_file_path + '.avif'
                            
                            with open(avif_temp_path, 'wb') as f:
                                f.write(image_data)
                            print(f"🔄 Saved as proper AVIF file: {avif_temp_path}")
                            
                            image = Image.open(avif_temp_path)
                            print(f"🖼️ AVIF image loaded from file: {image.size} {image.mode}")
                            
                            # Convert to RGB if needed
                            if image.mode != 'RGB':
                                image = image.convert('RGB')
                                print(f"🔄 Converted to RGB: {image.size} {image.mode}")
                            
                            detection_result = ai_service.detect_deepfake_image(image)
                            
                            # Clean up AVIF temp file
                            if os.path.exists(avif_temp_path):
                                os.remove(avif_temp_path)
                                print(f"🗑️ Cleaned up AVIF temp file: {avif_temp_path}")
                                
                        except Exception as avif_file_error:
                            print(f"❌ AVIF file approach failed: {avif_file_error}")
                            
                            # Third try: Try to convert AVIF to JPEG using pillow_avif
                            try:
                                print("🔄 Attempting AVIF to JPEG conversion...")
                                image_buffer = io.BytesIO(image_data)
                                image = Image.open(image_buffer)
                                
                                # Convert to RGB
                                if image.mode != 'RGB':
                                    image = image.convert('RGB')
                                
                                # Save as JPEG to a new buffer
                                jpeg_buffer = io.BytesIO()
                                image.save(jpeg_buffer, format='JPEG', quality=90)
                                jpeg_buffer.seek(0)
                                
                                # Load the JPEG data
                                jpeg_image = Image.open(jpeg_buffer)
                                print(f"🖼️ Converted AVIF to JPEG: {jpeg_image.size} {jpeg_image.mode}")
                                
                                detection_result = ai_service.detect_deepfake_image(jpeg_image)
                                
                            except Exception as conversion_error:
                                print(f"❌ AVIF conversion failed: {conversion_error}")
                                raise Exception(f"All AVIF processing methods failed. Original error: {conversion_error}")
                
                except Exception as avif_error:
                    print(f"❌ AVIF processing error: {avif_error}")
                    print(f"❌ Error type: {type(avif_error)}")
                    
                    # Final fallback: Try to process as regular image (might work if it's actually JPEG)
                    try:
                        print("🔄 Fallback: Trying to process as regular image...")
                        image = Image.open(temp_file_path)
                        print(f"🖼️ Fallback image loaded: {image.size} {image.mode}")
                        detection_result = ai_service.detect_deepfake_image(image)
                    except Exception as fallback_error:
                        print(f"❌ Fallback also failed: {fallback_error}")
                        raise Exception(f"All AVIF processing methods failed. Original error: {avif_error}")
            
            # Handle GIF images by extracting and analyzing frames
            elif should_process_as_gif:
                print("🎬 Processing GIF image...")
                try:
                    # Load the GIF
                    gif_image = Image.open(temp_file_path)
                    print(f"🎬 GIF loaded: {gif_image.size} {gif_image.mode}, frames: {getattr(gif_image, 'n_frames', 1)}")
                    
                    # Extract frames from GIF
                    frames = []
                    frame_count = getattr(gif_image, 'n_frames', 1)
                    max_frames = min(frame_count, 10)  # Limit to 10 frames for performance
                    
                    print(f"🎬 Extracting {max_frames} frames from {frame_count} total frames")
                    
                    for i in range(max_frames):
                        try:
                            gif_image.seek(i)
                            frame = gif_image.copy()
                            
                            # Convert to RGB if needed
                            if frame.mode != 'RGB':
                                frame = frame.convert('RGB')
                            
                            frames.append(frame)
                            print(f"🎬 Extracted frame {i+1}/{max_frames}: {frame.size} {frame.mode}")
                            
                        except Exception as frame_error:
                            print(f"❌ Error extracting frame {i+1}: {frame_error}")
                            continue
                    
                    if not frames:
                        raise Exception("No frames could be extracted from GIF")
                    
                    print(f"🎬 Successfully extracted {len(frames)} frames")
                    
                    # Analyze each frame
                    frame_results = []
                    for i, frame in enumerate(frames):
                        try:
                            result = ai_service.detect_deepfake_image(frame)
                            frame_results.append(result)
                            print(f"🎬 Frame {i+1} analysis: {result.get('is_deepfake', False)} ({result.get('confidence', 0.0):.2f})")
                        except Exception as frame_analysis_error:
                            print(f"❌ Error analyzing frame {i+1}: {frame_analysis_error}")
                            # Add a default result for failed frames
                            frame_results.append({
                                "is_deepfake": False,
                                "confidence": 0.0,
                                "error": str(frame_analysis_error)
                            })
                    
                    # Aggregate results (similar to video processing)
                    deepfake_count = sum(1 for r in frame_results if r.get("is_deepfake", False))
                    avg_confidence = sum(r.get("confidence", 0.0) for r in frame_results) / len(frame_results)
                    
                    print(f"🎬 Frame analysis complete: {deepfake_count}/{len(frame_results)} frames flagged as deepfake")
                    
                    # Determine overall result
                    is_deepfake = False
                    detection_type = "real"
                    
                    # Conservative aggregation logic for GIFs
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
                    
                    detection_result = {
                        "is_deepfake": is_deepfake,
                        "confidence": avg_confidence,
                        "frames_analyzed": len(frame_results),
                        "deepfake_frames": deepfake_count,
                        "model_used": frame_results[0].get("model_used", "unknown") if frame_results else "unknown",
                        "detection_type": detection_type,
                        "analysis": f"Analyzed {len(frame_results)} GIF frames ({deepfake_count} flagged as AI)",
                        "model_note": "GIF analysis - frame-by-frame deepfake detection",
                        "frame_details": frame_results
                    }
                    
                except Exception as gif_error:
                    print(f"❌ GIF processing error: {gif_error}")
                    # Fallback to single frame analysis
                    try:
                        print("🔄 GIF fallback: Trying single frame analysis...")
                        gif_image = Image.open(temp_file_path)
                        gif_image.seek(0)  # Get first frame
                        
                        if gif_image.mode != 'RGB':
                            gif_image = gif_image.convert('RGB')
                        
                        detection_result = ai_service.detect_deepfake_image(gif_image)
                        detection_result["analysis"] = "GIF analysis (fallback to first frame only)"
                        detection_result["model_note"] = "GIF fallback - single frame analysis"
                        
                    except Exception as fallback_error:
                        print(f"❌ GIF fallback also failed: {fallback_error}")
                        raise gif_error
            else:
                # Double-check file exists before opening
                if not os.path.exists(temp_file_path):
                    print(f"❌ File disappeared before Image.open(): {temp_file_path}")
                    active_requests.remove(detection_id)
                    raise HTTPException(status_code=500, detail="File disappeared before processing")
                
                print(f"🖼️ About to open image: {temp_file_path}")
                # For regular images, use the already saved file
                image = Image.open(temp_file_path)
                print(f"🖼️ Image loaded: {image.size} {image.mode}")
                detection_result = ai_service.detect_deepfake_image(image)
        except Exception as img_error:
            print(f"❌ Image processing error: {img_error}")
            # Clean up file before raising exception
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"🗑️ Cleaned up after error: {temp_file_path}")
            active_requests.remove(detection_id)
            raise HTTPException(status_code=500, detail=f"Image processing error: {str(img_error)}")
    
    processing_time = time.time() - start_time
    print(f"⏱️ Processing time: {processing_time:.2f}s")
    print(f"📊 Detection result: {detection_result}")
    
    # Get Claude analysis, sending the image path for multimodal analysis
    file_info = f"Filename: {file.filename}, Size: {file.size} bytes, Type: {file.content_type}"
    image_path_for_claude = temp_file_path if file.content_type.startswith("image/") else None
    
    # Disable Claude analysis for faster processing
    claude_analysis = "Claude analysis disabled for performance"
    
    # try:
    #     claude_analysis = await ai_service.analyze_with_claude(
    #         detection_result, 
    #         file_info, 
    #         image_path=image_path_for_claude
    #     )
    # except Exception as claude_error:
    #     print(f"⚠️ Claude analysis failed: {claude_error}")
    #     claude_analysis = "Analysis unavailable"
    
    # Create response with local model as primary
    result = DetectionResponse(
        is_deepfake=detection_result.get("is_deepfake", False),
        confidence=detection_result.get("confidence", 0.0),
        processing_time=processing_time,
        message=f"Analysis complete for {file.filename}",
        model_used=detection_result.get("model_used", "unknown"),
        claude_analysis=claude_analysis,
        detection_type=detection_result.get("detection_type", "unknown"),
        analysis=detection_result.get("analysis", ""),
        model_note=detection_result.get("model_note", "")
    )
    
    detection_results[detection_id] = result.model_dump()
    print(f"✅ Analysis complete: {result.is_deepfake} ({result.confidence:.2f})")
    
    # Clean up temporary file AFTER successful processing
    try:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"🗑️ Cleaned up: {temp_file_path}")
    except Exception as cleanup_error:
        print(f"⚠️ Cleanup error: {cleanup_error}")
    
    # Remove from active requests
    active_requests.remove(detection_id)
    print(f"🏁 Completed request {detection_id} (active: {len(active_requests)})")
    
    return result

@app.get("/results/{detection_id}")
def get_result(detection_id: str):
    """Get detection result by ID"""
    if detection_id not in detection_results:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    return detection_results[detection_id]

@app.get("/models")
def get_model_info():
    """Get information about available AI models"""
    ai_info = ai_service.get_model_info()
    text_info = text_service.get_model_info()
    
    return {
        "image_model": ai_info.get("image_model", {}),
        "audio_model": ai_info.get("audio_model", {}),
        "text_model": text_info,
        "anthropic_available": ai_info.get("anthropic_available", False),
        "groq_available": bool(os.getenv("GROQ_API_KEY"))
    }

@app.get("/hf-models")
def get_hf_models():
    """Get information about Hugging Face models"""
    return {
        "models": {
            "deepfake": "prithivMLmods/Deep-Fake-Detector-v2-Model"
        },
        "api_token_configured": False,
        "api_url": "Local model"
    }

@app.get("/stats")
def get_stats():
    """Get API statistics"""
    return {
        "total_detections": len(detection_results),
        "api_version": "1.0.0",
        "uptime": "running"
    }

@app.post("/batch-upload")
async def batch_upload(files: List[UploadFile] = File(...)):
    """Upload multiple files for batch detection"""

@app.get("/status/{detection_id}")
async def get_processing_status(detection_id: str):
    """Get real-time processing status"""

@app.post("/detect/{model_name}")
async def detect_with_model(
    model_name: str,
    file: UploadFile = File(...)
):
    """Detect deepfake using a specific model"""
    
    # Validate file type
    allowed_types = ["video/", "image/"]
    if not file.content_type or not any(file.content_type.startswith(t) for t in allowed_types):
        raise HTTPException(
            status_code=400, 
            detail="File must be a video or image"
        )
    
    # Generate unique ID
    detection_id = f"detection_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    temp_file_path = f"/tmp/{detection_id}_{file.filename}"
    
    # Save file
    async with aiofiles.open(temp_file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    start_time = time.time()
    
    # Use specific model for detection
    if file.content_type.startswith("video/"):
        detection_result = ai_service.detect_deepfake_video(temp_file_path)
    else:
        # For images, use the already saved file
        try:
            image = Image.open(temp_file_path)
            detection_result = ai_service.detect_deepfake_image(image)
        except Exception as img_error:
            # Clean up file before raising exception
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise HTTPException(status_code=500, detail=f"Image processing error: {str(img_error)}")
    
    processing_time = time.time() - start_time
    
    # Get Claude analysis, sending the image path for multimodal analysis
    file_info = f"Filename: {file.filename}, Size: {file.size} bytes, Type: {file.content_type}"
    image_path_for_claude = temp_file_path if file.content_type.startswith("image/") else None
    
    # Disable Claude analysis for faster processing
    claude_analysis = "Claude analysis disabled for performance"
    
    # try:
    #     claude_analysis = await ai_service.analyze_with_claude(
    #         detection_result, 
    #         file_info, 
    #         image_path=image_path_for_claude
    #     )
    # except Exception as claude_error:
    #     print(f"⚠️ Claude analysis failed: {claude_error}")
    #     claude_analysis = "Analysis unavailable"
    
    result = DetectionResponse(
        is_deepfake=detection_result.get("is_deepfake", False),
        confidence=detection_result.get("confidence", 0.0),
        processing_time=processing_time,
        message=f"Analysis complete using {model_name}",
        model_used=detection_result.get("model_used", model_name),
        claude_analysis=claude_analysis,
        detection_type=detection_result.get("detection_type", "unknown"),
        analysis=detection_result.get("analysis", ""),
        model_note=detection_result.get("model_note", "")
    )
    
    detection_results[detection_id] = result.model_dump()
    
    # Clean up temporary file AFTER successful processing
    try:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    except Exception as cleanup_error:
        print(f"⚠️ Cleanup error: {cleanup_error}")
    
    return result

@app.post("/upload-audio", response_model=DetectionResponse)
async def upload_audio_file(
    file: UploadFile = File(...)
):
    """Upload an audio file for deepfake detection"""
    
    # Validate file type
    allowed_types = ["audio/"]
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400, 
            detail="File must be an audio file"
        )
    
    # Validate file size (50MB limit for demo)
    if file.size and file.size > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=400, 
            detail="File size must be less than 50MB"
        )
    
    # Generate unique ID for this detection
    detection_id = f"detection_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    # Save uploaded file temporarily
    temp_file_path = f"/tmp/{detection_id}_{file.filename}"
    
    # Save file
    async with aiofiles.open(temp_file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Start AI processing
    start_time = time.time()
    
    # Detect deepfake in audio
    detection_result = ai_service.detect_deepfake_audio(temp_file_path)
    
    processing_time = time.time() - start_time
    
    # Create response
    result = DetectionResponse(
        is_deepfake=detection_result.get("is_deepfake", False),
        confidence=detection_result.get("confidence", 0.0),
        processing_time=processing_time,
        message=f"Audio analysis complete for {file.filename}",
        model_used=detection_result.get("model_used", "unknown"),
        claude_analysis="Audio analysis - no Claude integration yet",
        detection_type=detection_result.get("detection_type", "unknown"),
        analysis=detection_result.get("analysis", ""),
        model_note=detection_result.get("model_note", "")
    )
    
    detection_results[detection_id] = result.model_dump()
    
    # Clean up temporary file AFTER successful processing
    try:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    except Exception as cleanup_error:
        print(f"⚠️ Cleanup error: {cleanup_error}")
    
    return result

# RESTORED - Groq text detection endpoints enabled
@app.post("/detect-text", response_model=TextDetectionResponse)
async def detect_text(request: TextDetectionRequest):
    """Detect if text is likely from a bot or LLM"""
    
    print(f"📝 Received text detection request: {len(request.text)} characters")
    
    # Validate text length
    if len(request.text.strip()) < 5:
        raise HTTPException(
            status_code=400, 
            detail="Text must be at least 5 characters long"
        )
    
    if len(request.text) > 10000:
        raise HTTPException(
            status_code=400, 
            detail="Text must be less than 10,000 characters"
        )
    
    # Generate unique ID for this detection
    detection_id = f"text_detection_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    # Track active requests
    active_requests.add(detection_id)
    print(f"🚀 Starting text detection request {detection_id} (active: {len(active_requests)})")
    
    try:
        # Detect bot/LLM in text
        detection_result = text_service.detect_bot_llm_text(request.text)
        
        # Create response
        result = TextDetectionResponse(
            is_bot=detection_result.get("is_bot", False),
            confidence=detection_result.get("confidence", 0.0),
            processing_time=detection_result.get("processing_time", 0.0),
            message=f"Text analysis complete ({detection_result.get('text_length', 0)} characters)",
            model_used=detection_result.get("model_used", "unknown"),
            detection_type=detection_result.get("detection_type", "human"),
            analysis=detection_result.get("analysis", ""),
            indicators=detection_result.get("indicators", []),
            text_length=detection_result.get("text_length", 0)
        )
        
        print(f"✅ Text analysis complete: {result.is_bot} ({result.confidence:.2f}) - {result.detection_type}")
        
    except Exception as e:
        print(f"❌ Text detection error: {e}")
        active_requests.remove(detection_id)
        raise HTTPException(status_code=500, detail=f"Text detection error: {str(e)}")
    
    # Remove from active requests
    active_requests.remove(detection_id)
    print(f"🏁 Completed text detection request {detection_id} (active: {len(active_requests)})")
    
    return result

@app.post("/detect-texts", response_model=List[TextDetectionResponse])
async def detect_multiple_texts(texts: List[str]):
    """Detect bot/LLM in multiple texts efficiently"""
    
    print(f"📝 Received batch text detection request: {len(texts)} texts")
    
    # Validate input
    if len(texts) == 0:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    if len(texts) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 texts per batch")
    
    # Generate unique ID for this detection
    detection_id = f"batch_text_detection_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    # Track active requests
    active_requests.add(detection_id)
    print(f"🚀 Starting batch text detection request {detection_id} (active: {len(active_requests)})")
    
    try:
        # Detect bot/LLM in all texts
        detection_results = text_service.batch_detect_texts(texts)
        
        # Convert to response format
        results = []
        for detection_result in detection_results:
            result = TextDetectionResponse(
                is_bot=detection_result.get("is_bot", False),
                confidence=detection_result.get("confidence", 0.0),
                processing_time=detection_result.get("processing_time", 0.0),
                message=f"Text analysis complete ({detection_result.get('text_length', 0)} characters)",
                model_used=detection_result.get("model_used", "unknown"),
                detection_type=detection_result.get("detection_type", "human"),
                analysis=detection_result.get("analysis", ""),
                indicators=detection_result.get("indicators", []),
                text_length=detection_result.get("text_length", 0)
            )
            results.append(result)
        
        print(f"✅ Batch text analysis complete: {len(results)} results")
        
    except Exception as e:
        print(f"❌ Batch text detection error: {e}")
        active_requests.remove(detection_id)
        raise HTTPException(status_code=500, detail=f"Batch text detection error: {str(e)}")
    
    # Remove from active requests
    active_requests.remove(detection_id)
    print(f"🏁 Completed batch text detection request {detection_id} (active: {len(active_requests)})")
    
    return results

# Error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)