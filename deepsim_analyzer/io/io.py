# contains all methods for saving and loading image embeddings
import h5py
from PIL import Image
import hashlib
from pathlib import Path
import numpy as np

def save_feature(datafile_path, img_hash, img_feature, feature_name):
    
    assert img_feature.ndim == 1, f"Image feature should not be multidimensional, found shape {img_feature.shape}"
    with h5py.File(datafile_path, 'r+') as f:
        f.create_dataset(f"{img_hash}/features/{feature_name}", data=img_feature)

def read_feature(datafile_path, img_hash, feature_name):
    with h5py.File(datafile_path, 'r') as f:
        image_feature = f[f'{img_hash}/features/{feature_name}'][:]
    return image_feature

def create_dataset(image_folder, datafile_path="dataset.h5"):
    image_dir = Path(image_folder)
    # check folder for all possible extenstions of images, we don't want to accidentally glob other files
    image_paths_png = list(image_dir.glob("*.png"))
    image_paths_jpg = list(image_dir.glob("*.jpg"))
    image_paths_jpeg = list(image_dir.glob("*.jpeg"))
    image_paths = image_paths_png + image_paths_jpg + image_paths_jpeg

    # create dataset file and add hash for each image
    with h5py.File(datafile_path, "w") as f:
        for image_path in image_paths:
            image_name = str(Path(image_path))
            img_hash = get_image_hash(image_path, is_filepath=True)
            f.create_dataset(f"{img_hash}/filename", data=image_name)
            

def load_image(image_path, return_np=True):
    """load image from filepath, useful for consistant loading in different parts of the repository

    Args:
        image_path (str): path to image file
        return_np (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    # read using pillow, remove alpha channel if its there
    img_array = Image.open(image_path).convert('RGB')
    # sometimes we might not want to conver to a numpy array, for vizualization for example.
    if return_np:
        img_array = np.array(img_array)
    return img_array


def get_image_hash(image, is_filepath=False):
    if is_filepath:
        assert type(image) == str, f"Please pass a string filepath when using the is_filepath argument, got: {type(image)}"
        # return numpy array, always calculate hash from np array to avoid errors
        img_array = load_image(image, return_np=True)
    else:
        assert type(image) == np.ndarray, f"Please pass an numpy array to hash function, got: {type(image)}"
        img_array = image

    md5hash = hashlib.md5(img_array.tobytes())
    return md5hash.hexdigest()
