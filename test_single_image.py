"""
DermAI Single Image Test Script
Tests model prediction and optional Grad-CAM generation
"""

import os
from inference import (
    load_dermai_model,
    preprocess_image,
    predict_image,
    generate_gradcam,
    save_gradcam
)

# Configuration
MODEL_PATH = "models/final_model_best.keras"
IMAGE_PATH = "samples/ISIC_0033399.jpg"
OUTPUT_DIR = "outputs"

def main():
    """Run single image inference with optional Grad-CAM"""
    
    # Load and predict
    model = load_dermai_model(MODEL_PATH)
    img_array = preprocess_image(IMAGE_PATH)
    label, prob, confidence = predict_image(model, img_array)
    
    # Show results
    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Ask for Grad-CAM
    choice = input("\nShow Grad-CAM? (y/n): ").strip().lower()
    
    if choice == 'y':
        # Create output directory
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        # Generate output path
        image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{image_name}_gradcam.jpg")
        
        # Generate Grad-CAM
        heatmap = generate_gradcam(model, img_array)
        save_gradcam(IMAGE_PATH, heatmap, output_path)
        print(f"Saved: {output_path}")
    else:
        print("Done!")

if __name__ == "__main__":
    main()