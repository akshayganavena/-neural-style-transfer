import tensorflow as tf
import numpy as np
import PIL.Image
import tensorflow_hub as hub
import os
import random

# Load Pre-trained Model
model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

def load_image(image, max_dim=512):
    """
    Load and preprocess an image.
    """
    img = image.convert("RGB")  
    long_dim = max(img.size)
    scale = max_dim / long_dim
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), PIL.Image.LANCZOS)
    img = np.array(img)[np.newaxis, ...] / 255.0  
    return tf.convert_to_tensor(img, dtype=tf.float32)

def tensor_to_image(tensor):
    """
    Convert a TensorFlow tensor back to a PIL image.
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def style_transfer(content_img, style_img, style_strength=1.0):
    """
    Apply style transfer to a content image using a given style image.
    """
    content_image = load_image(content_img)
    style_image = load_image(style_img)

    stylized_image = model(tf.constant(content_image), tf.constant(style_image) * style_strength)[0]
    return tensor_to_image(stylized_image)
