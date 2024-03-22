from PIL import Image
import numpy as np
import os
from tqdm import tqdm
# Load the original image
image_path = '040324B    0.115 1.JPG'
image = Image.open(image_path)

# Target dimensions
target_width = 720
target_height = 576
original_width, original_height = image.size
aspect_ratio = original_width / original_height


new_width = target_width
new_height = int(new_width / aspect_ratio)
# Resize the image to new dimensions
image_resized = image.resize((new_width, new_height), Image.ANTIALIAS)
print(new_width,new_height)
# Calculate the position to align the bottom left corner of the resized image
bottom_left_corner = (0, target_height - new_height)  # Bottom left corner position
# Create a blank image with the desired dimensions
blank_image = Image.new("RGB", (720, 576), "white")
blank_image.paste(image_resized,bottom_left_corner)

# Save the result
blank_image.save('converted_image3.jpg')
#

def process_images(input_folder, output_folder, target_width=720, target_height=576):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    # Initialize tqdm progress bar
    progress_bar = tqdm(total=len(image_files), desc="Processing Images", unit="image")

    # Iterate over files in the input folder
    for filename in image_files:
        # Construct the full path of the input image
        input_image_path = os.path.join(input_folder, filename)
        
        # Load the original image
        image = Image.open(input_image_path)

        # Calculate aspect ratio
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        # Resize the image to new dimensions
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
        image_resized = image.resize((new_width, new_height), Image.ANTIALIAS)

        # Calculate the position to align the bottom left corner of the resized image
        bottom_left_corner = (0, target_height - new_height)

        # Create a blank image with the desired dimensions
        blank_image = Image.new("RGB", (720, 576), "white")
        blank_image.paste(image_resized, bottom_left_corner)

        # Save the processed image to the output folder
        output_image_path = os.path.join(output_folder, filename)
        blank_image.save(output_image_path)

        # Update progress bar
        progress_bar.update(1)

    # Close progress bar
    progress_bar.close()
    
# Example usage:
input_folder = "/home/pms/script-test/Loop-3/"
output_folder = "/home/pms/script-test/processed-images-Loop-3/"
process_images(input_folder, output_folder)
