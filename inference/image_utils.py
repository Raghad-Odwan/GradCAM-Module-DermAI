import cv2
import numpy as np
import os

IMG_SIZE = (224, 224)

def preprocess_image(img_path):
    """
    Preprocess image for model inference
    
    Args:
        img_path: Path to input image
    
    Returns:
        Preprocessed image array with shape (1, 224, 224, 3)
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    img = cv2.resize(img, IMG_SIZE)
    
    # Normalize to [0, 1]
    img = img / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img