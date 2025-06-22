import os
import time
import groq  # RESTORED - Groq enabled
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TextService:
    def __init__(self):
        self.client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))  # RESTORED
        self.model = "llama-3.1-8b-instant"  # Fast and reliable 8B model with large context
        
        # More conservative thresholds to reduce false positives
        self.bot_threshold = 0.8  # Higher confidence required for bot detection
        self.llm_threshold = 0.9  # Very high confidence for LLM detection
        self.suspicious_threshold = 0.75  # Higher threshold for suspicious content
        
        print(f"🤖 Text detection service initialized with Groq AI ({self.model})")
    
    def detect_bot_llm_text(self, text: str) -> Dict[str, Any]:
        """Detect if text is likely from a bot or LLM"""
        try:
            start_time = time.time()
            
            # Improved prompt for better detection accuracy
            prompt = f"""
            Analyze this text and determine if it was written by a human or AI/bot:

            TEXT: "{text}"

            Look for:
            - Personal details, emotions, imperfections (human)
            - Generic responses, perfect grammar, formal tone (AI/bot)
            - Specific experiences vs generic advice
            - Natural language vs robotic patterns

            Respond ONLY with this JSON format:
            {{
                "is_bot": true/false,
                "confidence": 0.0-1.0,
                "detection_type": "human/bot/llm/suspicious",
                "analysis": "brief explanation",
                "indicators": ["reason1", "reason2"]
            }}

            Be conservative - only flag as AI if you have strong evidence.
            """
            
            # Call Groq AI with better parameters
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a forensic linguist expert. Provide only valid JSON responses. Be conservative in AI detection - prefer false negatives over false positives."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Zero temperature for maximum consistency
                max_tokens=400,
                top_p=0.1  # Low top_p for focused responses
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            if response_text is None:
                response_text = ""
            response_text = response_text.strip()
            
            # Try to extract JSON from response
            try:
                import json
                # Clean up the response to extract JSON
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
                result = json.loads(response_text.strip())
                
                # Validate the result structure
                is_bot = result.get('is_bot', False)
                confidence = float(result.get('confidence', 0.0))
                detection_type = result.get('detection_type', 'human')
                analysis = result.get('analysis', '')
                indicators = result.get('indicators', [])
                
                # Apply conservative thresholds for final classification
                final_detection_type = 'human'
                if is_bot and confidence > self.llm_threshold:
                    final_detection_type = 'llm'
                elif is_bot and confidence > self.bot_threshold:
                    final_detection_type = 'bot'
                elif is_bot and confidence > self.suspicious_threshold:
                    final_detection_type = 'suspicious'
                else:
                    final_detection_type = 'human'  # Default to human if below thresholds
                
                processing_time = time.time() - start_time
                
                return {
                    "is_bot": is_bot,
                    "confidence": confidence,
                    "detection_type": final_detection_type,
                    "analysis": analysis,
                    "indicators": indicators,
                    "processing_time": processing_time,
                    "model_used": f"groq-{self.model}",
                    "text_length": len(text),
                    "raw_response": response_text
                }
                
            except json.JSONDecodeError as e:
                # Fallback parsing if JSON parsing fails
                print(f"⚠️ JSON parsing failed: {e}")
                print(f"Raw response: {response_text}")
                
                # Conservative fallback analysis
                is_bot = 'bot' in response_text.lower() or 'ai' in response_text.lower() or 'llm' in response_text.lower()
                confidence = 0.3 if is_bot else 0.1  # Lower confidence for fallback
                
                return {
                    "is_bot": is_bot,
                    "confidence": confidence,
                    "detection_type": "suspicious" if is_bot else "human",
                    "analysis": "Fallback analysis - JSON parsing failed",
                    "indicators": ["Response parsing error"],
                    "processing_time": time.time() - start_time,
                    "model_used": f"groq-{self.model}",
                    "text_length": len(text),
                    "raw_response": response_text,
                    "error": "JSON parsing failed"
                }
                
        except Exception as e:
            print(f"❌ Text detection error: {e}")
            return {
                "is_bot": False,
                "confidence": 0.0,
                "detection_type": "error",
                "analysis": f"Detection failed: {str(e)}",
                "indicators": [],
                "processing_time": time.time() - start_time,
                "model_used": f"groq-{self.model}",
                "text_length": len(text),
                "error": str(e)
            }
    
    def batch_detect_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Detect bot/LLM in multiple texts efficiently"""
        results = []
        
        for text in texts:
            if len(text.strip()) < 10:  # Skip very short texts
                results.append({
                    "is_bot": False,
                    "confidence": 0.0,
                    "detection_type": "human",
                    "analysis": "Text too short for analysis",
                    "indicators": ["Text length < 10 characters"],
                    "processing_time": 0.0,
                    "model_used": f"groq-{self.model}",
                    "text_length": len(text)
                })
            else:
                result = self.detect_bot_llm_text(text)
                results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the text detection model"""
        return {
            "model": self.model,
            "provider": "Groq AI",
            "capabilities": ["bot_detection", "llm_detection", "text_analysis"],
            "speed": "sub-100ms typical",
            "model_size": "8B parameters (llama-3.1-8b-instant)",
            "context_window": "131,072 tokens",
            "accuracy": "Fast and reliable with improved prompt",
            "thresholds": {
                "bot_threshold": self.bot_threshold,
                "llm_threshold": self.llm_threshold,
                "suspicious_threshold": self.suspicious_threshold
            }
        } 