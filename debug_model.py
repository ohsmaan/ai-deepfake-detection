#!/usr/bin/env python3
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

def debug_model():
    model_name = "aiwithoutborders-xyz/CommunityForensics-DeepfakeDet-ViT"
    
    print(f"🔍 Debugging model: {model_name}")
    
    try:
        # Load processor and model
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Model config: {model.config}")
        print(f"📊 Processor config: {processor.config if hasattr(processor, 'config') else 'No config'}")
        
        # Check what size the processor expects
        if hasattr(processor, 'size'):
            print(f"📏 Processor size: {processor.size}")
        if hasattr(processor, 'crop_size'):
            print(f"📏 Processor crop_size: {processor.crop_size}")
        if hasattr(processor, 'do_resize'):
            print(f"🔄 Processor do_resize: {processor.do_resize}")
        
        # Create a test image
        test_image = Image.new('RGB', (384, 384), color='red')
        print(f"📸 Test image size: {test_image.size}")
        
        # Try processing
        inputs = processor(images=test_image, return_tensors="pt")
        print(f"📦 Input tensor shape: {inputs['pixel_values'].shape}")
        
        # Try inference
        with torch.no_grad():
            outputs = model(**inputs)
            print(f"✅ Inference successful!")
            print(f"📊 Output shape: {outputs.logits.shape}")
            
            # Get prediction
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, predicted_class = torch.max(probabilities, 1)
            predicted_label = model.config.id2label[predicted_class.item()]
            
            print(f"🎯 Prediction: {predicted_label}")
            print(f"🎯 Confidence: {confidence.item()}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model() 