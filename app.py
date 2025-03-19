import gradio as gr
from model import style_transfer
import os
import random
from PIL import Image

# Ensure output folder exists
os.makedirs("outputs", exist_ok=True)

# Preload style images from the "styles" folder
STYLE_FOLDER = "styles"
style_images = [os.path.join(STYLE_FOLDER, img) for img in os.listdir(STYLE_FOLDER) if img.endswith(('jpg', 'png'))]

def transfer_style(content):
    """
    Apply a random style transfer to the content image.
    """
    if not style_images:
        return "No style images found!"

    style_img_path = random.choice(style_images)
    style_img = Image.open(style_img_path)

    # Generate output path
    output_path = f"outputs/stylized_{random.randint(1,1000)}.jpg"

    # Apply style transfer
    stylized_image = style_transfer(content, style_img, style_strength=random.uniform(0.7, 1.2))
    stylized_image.save(output_path)

    return stylized_image

# Define Gradio Interface
iface = gr.Interface(
    fn=transfer_style,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Neural Style Transfer 🎨",
    description="Upload a content image and a random style will be applied.",
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
