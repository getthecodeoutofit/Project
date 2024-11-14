import torch
from PIL import Image
import requests
import pytesseract
import cv2
from transformers import CLIPProcessor, CLIPModel, pipeline

# Load pre-trained CLIP model for image-to-text understanding
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load a text summarization model (e.g., BART or T5)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to perform OCR on image for any textual context in the frame
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text.strip()

# Function to analyze the image using CLIP
def analyze_image(image_path):
    # Load the image
    image = Image.open(image_path)

    # Preprocess the image for CLIP
    inputs = clip_processor(images=image, return_tensors="pt")
    outputs = clip_model(**inputs)

    # We assume the best match of CLIP captions for summarization
    text_description = f"The image contains various elements such as {outputs.logits_per_image.argmax()}. "
    return text_description

# Generate a summary based on extracted elements
def generate_summary(image_path):
    # Extract text from image (OCR)
    image = Image.open(image_path)
    ocr_text = extract_text_from_image(image)

    # Generate description using CLIP
    clip_description = analyze_image(image_path)

    # Combine OCR text and CLIP description for final input to the summarizer
    combined_text = f"{clip_description}. Text found in the image: '{ocr_text}'" if ocr_text else clip_description
    
    # Summarize the combined text using a language model
    summary = summarizer(combined_text, max_length=50, min_length=25, do_sample=False)
    return summary[0]["summary_text"]

# Sample usage
image_path = "path/to/your/image.jpg"  # Replace with your image path
summary = generate_summary(image_path)
print("Generated Summary of the Video Context:")
print(summary)
