import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def display_image(image_path):
    """Display an image using PIL and matplotlib."""
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def create_output_dir(base_output_dir, relative_path):
    """Create output directory structure based on the relative path."""
    output_dir = os.path.join(base_output_dir, relative_path)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_image_files(input_dir):
    """Get all image files from the input directory."""
    return [os.path.join(dp, f) for dp, dn, filenames in os.walk(input_dir) for f in filenames if f.endswith(".png")]

def process_image(image_file, input_dir, output_dir, kernel_size, space_threshold):
    """Process a single image file to extract and save word images."""
    image_path = os.path.join(input_dir, image_file)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones(kernel_size, np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sort_contours(contours)

    combined_contours = combine_contours(contours, space_threshold)

    for i, (x, y, w, h) in enumerate(combined_contours):
        if w > 10 and h > 10:
            word_image = image[y:y+h, x:x+w]
            output_image_path = os.path.join(output_dir, f"{os.path.basename(image_file).replace('.png', '')}.png")
            cv2.imwrite(output_image_path, word_image)
            display_image(output_image_path)
            print(f"Image saved: {output_image_path}")

def sort_contours(contours):
    """Sort contours based on their bounding box position."""
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    (contours, _) = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0], reverse=False))
    return contours

def combine_contours(contours, space_threshold):
    """Combine contours that are close to each other based on space threshold."""
    combined_contours = []
    current_contour = None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if current_contour is None:
            current_contour = [x, y, w, h]
        else:
            prev_x, prev_y, prev_w, prev_h = current_contour
            if x - (prev_x + prev_w) <= space_threshold:
                new_x = prev_x
                new_y = min(prev_y, y)
                new_w = max(prev_x + prev_w, x + w) - new_x
                new_h = max(prev_y + prev_h, y + h) - new_y
                current_contour = [new_x, new_y, new_w, new_h]
            else:
                combined_contours.append(tuple(current_contour))
                current_contour = [x, y, w, h]

    if current_contour is not None:
        combined_contours.append(tuple(current_contour))

    return combined_contours

def determine_group_size(name, groups):
    """Determine the group size based on the name."""
    if name in groups['small']:
        return 'small'
    elif name in groups['large']:
        return 'large'
    elif name in groups['esmall']:
        return 'esmall'
    else:
        return 'normal'

def process_all_folders(input_directory, groups):
    """Process all folders within the input directory."""
    folders = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, f))]

    for folder in folders:
        name = os.path.basename(folder)
        output_directory = os.path.join("Detected", name)
        group_size = determine_group_size(name, groups)

        kernel_size = {
            'small': (8, 8),
            'large': (30, 30),
            'esmall': (4, 4),
            'normal': (15, 15)
        }.get(group_size, (15, 15))

        image_files = get_image_files(folder)
        for image_file in image_files:
            relative_path = os.path.relpath(image_file, folder)
            output_dir = create_output_dir(output_directory, os.path.dirname(relative_path))
            process_image(image_file, folder, output_dir, kernel_size, space_threshold=10)



input_directory = "/kaggle/input/linebyline"

process_all_folders(input_directory, groups=small)
