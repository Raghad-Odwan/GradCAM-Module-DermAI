import tensorflow as tf
import os

def load_dermai_model(model_path=None):
    """
    Load the trained DermAI model
    
    Args:
        model_path: Path to model file. If None, uses default path.
    
    Returns:
        Loaded Keras model
    """
    if model_path is None:
        # Default path relative to project root
        model_path = os.path.join("models", "final_model_best.keras")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
    return model