import numpy as np

def predict_image(model, image, threshold=0.5):
    """
    Make prediction on preprocessed image
    
    Args:
        model: Trained Keras model
        image: Preprocessed image array (1, 224, 224, 3)
        threshold: Classification threshold (default: 0.5)
    
    Returns:
        tuple: (label, probability, confidence)
            - label: "Benign" or "Malignant"
            - probability: Raw model output (0-1, probability of Malignant)
            - confidence: Confidence percentage for predicted class
    """
    # Get prediction
    prediction = model.predict(image, verbose=0)
    prob = float(prediction[0][0])
    
    # Determine class and confidence
    if prob >= threshold:
        label = "Malignant"
        confidence = prob * 100
    else:
        label = "Benign"
        confidence = (1 - prob) * 100
    
    return label, prob, confidence