from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import os

def load_trocr_model():
    """Load the TrOCR model and processor."""
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    return model, processor

def ocr_image(image_path, model, processor):
    """Perform OCR on the image using the TrOCR model."""
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = model.generate(pixel_values)
    text = processor.decode(outputs[0], skip_special_tokens=True)
    return text

def save_text_output(image_file, text, output_dir):
    """Save the detected text to a .txt file."""
    base_name = os.path.basename(image_file).replace('.png', '')
    text_file_name = f"{base_name}_{text[:30]}.txt"  # Truncate text for filename
    text_file_path = os.path.join(output_dir, text_file_name)
    with open(text_file_path, 'w') as text_file:
        text_file.write(text)

def process_images_for_text(input_dir, output_dir):
    """Process images to perform OCR and save text files."""
    model, processor = load_trocr_model()
    text_output_dir = create_output_dir(output_dir, "text")

    image_files = get_image_files(input_dir)
    for image_file in image_files:
        text = ocr_image(image_file, model, processor)
        save_text_output(image_file, text, text_output_dir)
        print(f"Text saved for {image_file}: {text[:100]}...")  # Print a snippet of the text

input_directory = "/kaggle/working/Detected"
output_directory = "/kaggle/working/text"
process_images_for_text(input_directory, output_directory)
