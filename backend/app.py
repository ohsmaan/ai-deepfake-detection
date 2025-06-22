from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from typing import Optional, List
import time
import os
import aiofiles
from PIL import Image
from services.ai_service import AIService

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

@app.get("/")
def root():
    return {"message": "AI Deepfake Detector API is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/upload", response_model=DetectionResponse)
async def upload_file(
    file: UploadFile = File(...)
):
    """Upload a video or image file for deepfake detection"""
    
    # Validate file type
    allowed_types = ["video/", "image/"]
    if not file.content_type or not any(file.content_type.startswith(t) for t in allowed_types):
        raise HTTPException(
            status_code=400, 
            detail="File must be a video or image"
        )
    
    # Validate file size (50MB limit for demo)
    if file.size and file.size > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=400, 
            detail="File size must be less than 50MB"
        )
    
    # Generate unique ID for this detection
    detection_id = f"detection_{int(time.time())}"
    
    # Save uploaded file temporarily
    temp_file_path = f"/tmp/{detection_id}_{file.filename}"
    
    try:
        # Save file
        async with aiofiles.open(temp_file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Start AI processing
        start_time = time.time()
        
        # Detect deepfake based on file type
        if file.content_type.startswith("video/"):
            detection_result = ai_service.detect_deepfake_video(temp_file_path)
        else:
            # For images, convert to PIL Image
            image = Image.open(temp_file_path)
            detection_result = ai_service.detect_deepfake_image(image)
        
        processing_time = time.time() - start_time
        
        # Get Claude analysis
        file_info = f"Filename: {file.filename}, Size: {file.size} bytes, Type: {file.content_type}"
        claude_analysis = await ai_service.analyze_with_claude(detection_result, file_info)
        
        # Create response
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
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

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
    detection_id = f"detection_{int(time.time())}"
    temp_file_path = f"/tmp/{detection_id}_{file.filename}"
    
    try:
        # Save file
        async with aiofiles.open(temp_file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        start_time = time.time()
        
        # Use specific model for detection
        if file.content_type.startswith("video/"):
            detection_result = ai_service.detect_deepfake_video(temp_file_path)
        else:
            image = Image.open(temp_file_path)
            # Use local model for detection
            detection_result = ai_service.detect_deepfake_image(image)
        
        processing_time = time.time() - start_time
        
        # Get Claude analysis
        file_info = f"Filename: {file.filename}, Model: {model_name}"
        claude_analysis = await ai_service.analyze_with_claude(detection_result, file_info)
        
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
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)