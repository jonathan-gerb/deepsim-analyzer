# similarity based on DINO attention heads, see https://github.com/facebookresearch/dino


def get_similarity_image(image_a, image_b):
    """Get similarity between two images, first calculates image features and then calculates similarity

    Args:
        image_a (np.ndarray): array for first image
        image_b (np.ndarray): array for second image

    Returns:
        float: similarity score
    """
    similarity = 0
    return similarity

def get_feature_vector(image):
    """Get feature vector for given image

    Args:
        image (np.ndarray): numpy array of the image 

    Returns:
        np.array: feature vector for image.
    """
    feature_vector = None
    return feature_vector