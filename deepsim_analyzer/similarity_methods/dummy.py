
import numpy as np
from scipy import spatial
from tqdm import tqdm

def get_similarity_image(image_a, image_b):
    """Get similarity between two images, first calculates image features and then calculates similarity

    Args:
        image_a (np.ndarray): array for first image
        image_b (np.ndarray): array for second image

    Returns:
        float: similarity score
    """
    vector_a = calculate_feature_vector(image_a)
    vector_b = calculate_feature_vector(image_b)
    distance = spatial.distance.cosine(vector_a, vector_b)
    similarity = 1 - distance
    return similarity

def calculate_feature_vector(image):
    """Dummy feature vector.

    Args:
        image (np.ndarray): numpy array of the image 

    Returns:
        np.array: feature vector for image.
    """
    mean = np.mean(image)
    sum = np.sum(image)
    median = np.median(image)
    std = np.std(image)
    feature_vector = np.stack([mean, sum, median, std])

    return feature_vector

def calc_and_save_features(images, datafile_path, overwrite=True):
    """Process list of images. For dummy this doesnt make any difference but with other features you would not 
    want to reload a model for each image processing step, therefore a batch processing function for each feature
    is very necessary. This is an exaple of how to set this up for other features.

    Args:
        images (list): list of filepaths to the images
        datafile_path (str): path to the dataset file to save the outputs to
    """
    # local import inside function to avoid circular import problem
    from deepsim_analyzer.io import save_feature, load_image, get_image_hash, key_in_dataset
    skipped = 0
    calculated = 0
    for image_path in tqdm(images, desc=f"calculating dummy features", total=len(images)):
        image_path = str(image_path)
        image = load_image(image_path)
        hash = get_image_hash(image_path)
        if key_in_dataset(datafile_path, f"{hash}/features/dummy/full") and not overwrite:
            skipped += 1
            continue
        else:
            feature_vector = calculate_feature_vector(image)
            save_feature(datafile_path, hash, feature_vector, 'dummy')
            calculated += 1

    print(f"skipped: {skipped}, caluclated: {calculated}")


def calc_features(image_path, datafile_path):
    """Process list of images. For dummy this doesnt make any difference but with other features you would not 
    want to reload a model for each image processing step, therefore a batch processing function for each feature
    is very necessary. This is an exaple of how to set this up for other features.

    Args:
        images : path to image 
        datafile_path (str): path to the dataset file to save the outputs to
    """
    # local import inside function to avoid circular import problem
    from deepsim_analyzer.io import save_feature, load_image, get_image_hash

    image_path = str(image_path)
    image = load_image(image_path)
    feature_vector = calculate_feature_vector(image)
    return feature_vector