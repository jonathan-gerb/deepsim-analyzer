from PIL import Image
import os
import concurrent.futures

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
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for filename in file_list:
            # Construct the full file paths
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            futures.append(executor.submit(rescale_image, input_path, output_path, max_dimension))
        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

input_folder = "/mnt/mass_storage/ai_projects/mma/images"
output_folder = "/mnt/mass_storage/ai_projects/mma/images_resized"
max_dimension = 512

rescale_images(input_folder, output_folder, max_dimension)
