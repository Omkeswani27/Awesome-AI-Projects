import torch
from PIL import Image, ImageOps, ImageFilter
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import warnings
import os
import time
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import json
from datetime import datetime
import gradio as gr
import cv2
from pathlib import Path

warnings.filterwarnings('ignore')

class EnhancedImageCaptionGenerator:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        """
        Initialize the enhanced BLIP model for image captioning with multiple features
        
        Args:
            model_name (str): Name of the BLIP model to use
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_time = None
        self.caption_history = []
        self.model_loaded = False
        
    def load_model(self):
        """Load the model with progress tracking"""
        if self.model_loaded:
            return True
            
        print("Loading BLIP model...")
        start_time = time.time()
        
        try:
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_name
            ).to(self.device)
            
            self.load_time = time.time() - start_time
            self.model_loaded = True
            print(f"Model loaded on {self.device} in {self.load_time:.2f} seconds")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_image(self, source: str) -> Optional[Image.Image]:
        """
        Load image from various sources (URL, file path, PIL Image)
        
        Args:
            source: Can be URL, file path, or PIL Image
            
        Returns:
            PIL.Image: Loaded image or None if failed
        """
        try:
            if isinstance(source, Image.Image):
                return source.convert('RGB')
            elif source.startswith(('http://', 'https://')):
                response = requests.get(source, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                return image.convert('RGB')
            elif os.path.exists(source):
                image = Image.open(source)
                return image.convert('RGB')
            else:
                print("Invalid image source")
                return None
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def preprocess_image(self, image: Image.Image, size: Tuple[int, int] = (384, 384)) -> Image.Image:
        """
        Preprocess image for better caption generation
        
        Args:
            image: Input image
            size: Target size for resizing
            
        Returns:
            Preprocessed image
        """
        try:
            # Resize maintaining aspect ratio
            image.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Enhance image quality
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            return image
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return image
    
    def generate_caption(
        self, 
        image: Image.Image, 
        max_length: int = 50, 
        num_beams: int = 5,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate caption for the provided image with enhanced options
        
        Args:
            image: Input image
            max_length: Maximum length of the caption
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            repetition_penalty: Penalty for repeated words
            context: Optional context text for conditional generation
            
        Returns:
            Dictionary containing caption and metadata
        """
        if not self.model_loaded:
            if not self.load_model():
                return {"caption": "Error: Model failed to load", "success": False}
        
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Prepare inputs
            if context:
                inputs = self.processor(processed_image, context, return_tensors="pt").to(self.device)
            else:
                inputs = self.processor(processed_image, return_tensors="pt").to(self.device)
            
            # Generate caption
            start_time = time.time()
            with torch.no_grad():
                output = self.model.generate(
                    **inputs, 
                    max_length=max_length, 
                    num_beams=num_beams,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    early_stopping=True
                )
            generation_time = time.time() - start_time
            
            # Decode the generated caption
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Store in history
            result = {
                "caption": caption,
                "success": True,
                "generation_time": generation_time,
                "timestamp": datetime.now().isoformat(),
                "image_size": image.size,
                "parameters": {
                    "max_length": max_length,
                    "num_beams": num_beams,
                    "temperature": temperature,
                    "repetition_penalty": repetition_penalty,
                    "context": context
                }
            }
            
            self.caption_history.append(result)
            return result
            
        except Exception as e:
            error_msg = f"Error generating caption: {e}"
            print(error_msg)
            return {"caption": error_msg, "success": False}
    
    def generate_multiple_captions(
        self, 
        image: Image.Image, 
        num_captions: int = 3,
        max_length: int = 50,
        diversity_penalty: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple diverse captions for the same image
        
        Args:
            image: Input image
            num_captions: Number of diverse captions to generate
            max_length: Maximum length of each caption
            diversity_penalty: Controls diversity among captions
            
        Returns:
            List of caption results
        """
        results = []
        for i in range(num_captions):
            result = self.generate_caption(
                image, 
                max_length=max_length,
                num_beams=5,
                temperature=0.7 + (i * 0.1),  # Vary temperature for diversity
                repetition_penalty=1.0 + (i * 0.1)
            )
            results.append(result)
        return results
    
    def display_image_with_caption(
        self, 
        image: Image.Image, 
        caption: str, 
        save_path: Optional[str] = None
    ):
        """
        Display image with generated caption with enhanced visualization
        
        Args:
            image: Input image
            caption: Generated caption
            save_path: Path to save the visualization
        """
        try:
            plt.figure(figsize=(12, 10))
            plt.imshow(np.array(image))
            
            # Create a semi-transparent background for the caption
            plt.figtext(0.5, 0.01, f"Generated Caption: {caption}", 
                       ha="center", fontsize=16, 
                       bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.5"))
            
            plt.axis('off')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                print(f"Image saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error displaying image: {e}")
    
    def save_caption_history(self, file_path: str = "caption_history.json"):
        """Save caption history to JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.caption_history, f, indent=2)
            print(f"Caption history saved to {file_path}")
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def batch_process_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of image paths or URLs
            
        Returns:
            List of caption results
        """
        results = []
        for i, path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {path}")
            image = self.load_image(path)
            if image:
                result = self.generate_caption(image)
                results.append({"image_path": path, **result})
            else:
                results.append({"image_path": path, "success": False, "error": "Failed to load image"})
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "load_time": self.load_time,
            "captions_generated": len(self.caption_history),
            "model_loaded": self.model_loaded
        }

# Example usage and demonstration
def main():
    """Enhanced main function with multiple demonstration scenarios"""
    
    # Initialize the enhanced caption generator
    caption_generator = EnhancedImageCaptionGenerator()
    
    print("Enhanced Image Caption Generator")
    print("=" * 50)
    print(f"Using device: {caption_generator.device}")
    
    # Example images with different types
    image_sources = [
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg",
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg",
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d",
        "https://images.unsplash.com/photo-1541963463532-d68292c34b19"
    ]
    
    # Load model once
    if caption_generator.load_model():
        # Process each image with different parameters
        for i, source in enumerate(image_sources, 1):
            print(f"\n{'='*20} Processing image {i} {'='*20}")
            
            # Load image
            image = caption_generator.load_image(source)
            if not image:
                continue
            
            # Generate caption with different parameters
            print("1. Standard caption:")
            result1 = caption_generator.generate_caption(image)
            if result1["success"]:
                print(f"Caption: {result1['caption']}")
                print(f"Generation time: {result1['generation_time']:.2f}s")
            
            print("\n2. Creative caption (higher temperature):")
            result2 = caption_generator.generate_caption(image, temperature=1.2)
            if result2["success"]:
                print(f"Caption: {result2['caption']}")
            
            print("\n3. Multiple diverse captions:")
            results = caption_generator.generate_multiple_captions(image, num_captions=2)
            for j, res in enumerate(results, 1):
                if res["success"]:
                    print(f"Caption {j}: {res['caption']}")
            
            # Display the first result
            if result1["success"]:
                caption_generator.display_image_with_caption(image, result1["caption"])
        
        # Show model information
        print(f"\n{'='*20} Model Information {'='*20}")
        model_info = caption_generator.get_model_info()
        for key, value in model_info.items():
            print(f"{key}: {value}")
        
        # Save history
        caption_generator.save_caption_history()
    else:
        print("Failed to load model. Please check your internet connection and try again.")

# Gradio interface for web demo
def create_gradio_interface():
    """Create a Gradio web interface for the caption generator"""
    generator = EnhancedImageCaptionGenerator()
    generator.load_model()
    
    def process_image(image, max_length, num_beams, temperature, context):
        if image is None:
            return "Please upload an image first", None
        
        result = generator.generate_caption(
            image, 
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            context=context if context else None
        )
        
        if result["success"]:
            # Create visualization
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(np.array(image))
            ax.set_title(f"Caption: {result['caption']}", fontsize=12, pad=20)
            ax.axis('off')
            plt.tight_layout()
            
            return result['caption'], fig
        else:
            return result['caption'], None
    
    # Define interface
    with gr.Blocks(title="Enhanced Image Caption Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🖼️ Enhanced Image Caption Generator")
        gr.Markdown("Upload an image and get AI-generated captions with customizable parameters")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload Image", type="pil")
                context_input = gr.Textbox(label="Context (optional)", placeholder="E.g., 'a picture of'")
                
                with gr.Accordion("Advanced Parameters", open=False):
                    max_length = gr.Slider(10, 100, value=50, label="Max Caption Length")
                    num_beams = gr.Slider(1, 10, value=5, step=1, label="Number of Beams")
                    temperature = gr.Slider(0.1, 2.0, value=1.0, label="Temperature")
                
                generate_btn = gr.Button("Generate Caption", variant="primary")
            
            with gr.Column():
                caption_output = gr.Textbox(label="Generated Caption", interactive=False)
                plot_output = gr.Plot(label="Image with Caption")
        
        examples = gr.Examples(
            examples=[
                ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg", ""],
                ["https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d", "a portrait of"],
                ["https://images.unsplash.com/photo-1541963463532-d68292c34b19", "a book titled"]
            ],
            inputs=[image_input, context_input],
            label="Example Images"
        )
        
        generate_btn.click(
            fn=process_image,
            inputs=[image_input, max_length, num_beams, temperature, context_input],
            outputs=[caption_output, plot_output]
        )
    
    return demo

if __name__ == "__main__":
    # Run the enhanced demo
    main()
    
    # Uncomment the next line to launch the Gradio web interface
    # create_gradio_interface().launch(share=True)