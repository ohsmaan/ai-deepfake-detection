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
            if file.content_type == "image/avif":
                print("🔄 Processing AVIF image...")
                try:
                    # Read the file content into memory first
                    with open(temp_file_path, 'rb') as f:
                        image_data = f.read()
                    
                    print(f"📄 Read {len(image_data)} bytes from AVIF file")
                    
                    # Use BytesIO to load the image from memory
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
                except Exception as avif_error:
                    print(f"❌ AVIF processing error: {avif_error}")
                    print(f"❌ Error type: {type(avif_error)}")
                    # Try alternative approach - save as proper AVIF file
                    try:
                        avif_temp_path = temp_file_path.replace('.jpg', '.avif')
                        with open(avif_temp_path, 'wb') as f:
                            f.write(image_data)
                        print(f"🔄 Saved as proper AVIF file: {avif_temp_path}")
                        image = Image.open(avif_temp_path)
                        detection_result = ai_service.detect_deepfake_image(image)
                        # Clean up AVIF temp file
                        if os.path.exists(avif_temp_path):
                            os.remove(avif_temp_path)
                    except Exception as alt_error:
                        print(f"❌ Alternative AVIF approach failed: {alt_error}")
                        raise avif_error
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
    return ai_service.get_model_info()

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

# Error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)