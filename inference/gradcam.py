import tensorflow as tf
import cv2
import numpy as np
import os

def generate_gradcam(model, img_array, layer_name="conv5_block3_out"):
    """
    Generate Grad-CAM heatmap
    
    Args:
        model: Keras model
        img_array: Preprocessed image (1, 224, 224, 3)
        layer_name: Name of last convolutional layer
    
    Returns:
        Heatmap as numpy array
    """
    # Create gradient model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if isinstance(predictions, list):
            predictions = predictions[0]

        if len(predictions.shape) == 1:
            predictions = tf.expand_dims(predictions, axis=0)

        loss = predictions[:, 0]

    # Get gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Pool gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Get feature maps
    conv_outputs = conv_outputs[0]
    
    # Compute weighted feature map
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap + 1e-10)

    return heatmap.numpy()


def save_gradcam(original_path, heatmap, output_path):
    """
    Overlay Grad-CAM heatmap on original image and save
    
    Args:
        original_path: Path to original image
        heatmap: Generated heatmap from generate_gradcam()
        output_path: Where to save the result
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load original image
    img = cv2.imread(original_path)
    if img is None:
        raise ValueError(f"Failed to load image: {original_path}")
    
    # Resize image and heatmap to same size
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (224, 224))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay heatmap on image
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    # Save result
    cv2.imwrite(output_path, overlay)
    print(f"Grad-CAM saved to: {output_path}")