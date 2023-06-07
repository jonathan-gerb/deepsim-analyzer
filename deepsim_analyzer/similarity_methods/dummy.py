
import numpy as np
from scipy import spatial

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