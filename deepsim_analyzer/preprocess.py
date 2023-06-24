from PIL import Image
import os

def rescale_image(input_path, output_path, max_dimension):
    image = Image.open(input_path)
    width, height = image.size
    scale = max_dimension / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height))
    image.save(output_path)

def rescale_images(input_folder, output_folder, max_dimension):
    os.makedirs(output_folder, exist_ok=True)
    file_list = os.listdir(input_folder)
    for filename in file_list:
        # Construct the full file paths
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        rescale_image(input_path, output_path, max_dimension)

input_folder = "..."
output_folder = "..."

max_dimension = 512

rescale_images(input_folder, output_folder, max_dimension)
