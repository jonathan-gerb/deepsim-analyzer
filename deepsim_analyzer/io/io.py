# contains all methods for saving and loading image embeddings
import hashlib
import os
import pickle
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from random import shuffle

from ..similarity_methods import dino, dummy, texture, emotion, semantic


def save_feature(dataset_filepath, img_hash, img_feature, feature_name, is_projection=False, overwrite=False):
    # we now save feature maps using this same function so we can have multidim arrays
    # assert (
    #     img_feature.ndim == 1
    # ), f"Image feature should not be multidimensional, found shape {img_feature.shape}"

    if is_projection: 
        with h5py.File(dataset_filepath, "r+") as f:
            key = f"{img_hash}/features/{feature_name}/projection"
            if key in f:
                if overwrite:
                    del f[key]
                    f.create_dataset(f"{img_hash}/features/{feature_name}/projection", data=img_feature, compression="lzf")
                else:
                    # dont overwrite existing feature
                    pass
            else:
                f.create_dataset(f"{img_hash}/features/{feature_name}/projection", data=img_feature, compression="lzf")

    else:
        with h5py.File(dataset_filepath, "r+") as f:
            key = f"{img_hash}/features/{feature_name}/full"
            if key in f:
                if overwrite:
                    del f[key]
                    f.create_dataset(f"{img_hash}/features/{feature_name}/full", data=img_feature, compression="lzf")
                else:
                    # dont overwrite existing feature
                    pass
            else:    
                f.create_dataset(f"{img_hash}/features/{feature_name}/full", data=img_feature, compression="lzf")

def key_in_dataset(dataset_filepath, img_hash):
    with h5py.File(dataset_filepath, "r") as f:
        return img_hash in f.keys()



def read_feature(dataset_filepath, img_hash, feature_name, read_projection=False):
    with h5py.File(dataset_filepath, "r") as f:
        if read_projection:
            image_feature = f[f"{img_hash}/features/{feature_name}/projection"][:]
        else:
            image_feature = f[f"{img_hash}/features/{feature_name}/full"][:]

    return image_feature

def add_metadata_from_pickle(dataset_filepath, pickle_filepath, image_folder):
    # pickle contains a dataframe
    df = pickle.load(open(pickle_filepath, 'rb'))
    metadata_dict = df.to_dict(orient='split', index=True)
    image_folder = Path(image_folder)

    with h5py.File(dataset_filepath, "r+") as f:
        added = 0
        for image_idx in tqdm(range(len(metadata_dict['index'])), desc="processing pickle file ", total=len(metadata_dict['index'])):
            # get image path
            partial_img_path = metadata_dict['data'][image_idx][1]
            full_image_path = image_folder / Path(partial_img_path).name
            if os.path.exists(str(full_image_path)):
                added += 1
                img_hash = get_image_hash(str(full_image_path))
                    
                for column_idx, col in enumerate(metadata_dict['columns']):
                    if f"metadata/{col}" in f[f"{img_hash}"]:
                        print(f"metadata image {str(full_image_path)} with hash {img_hash} already exists!")
                        break
                    else:
                        data = metadata_dict['data'][image_idx]
                        f[f"{img_hash}"].create_dataset(f"metadata/{col}", data=data[column_idx])


def read_metadata(dataset_filepath, img_hash):
    with h5py.File(dataset_filepath, "r") as f:
        metadata_keys = list(f[img_hash]['metadata'].keys())
        return {
            key : f[img_hash]['metadata'][key][()] for key in metadata_keys
        }
    
def read_metadata_batch(dataset_filepath, img_hashes):
    with h5py.File(dataset_filepath, "r") as f:
        metadata_dict = {}
        for img_hash in tqdm(img_hashes, desc="reading metadata", total=len(img_hashes)):
            metadata_keys = list(f[img_hash]['metadata'].keys())
            metadata_for_key =  {
                key : f[img_hash]['metadata'][key][()] for key in metadata_keys
            }
            metadata_dict[img_hash] = metadata_for_key

        # convert all the strings from byte to string
        for hash, metadata in metadata_dict.items():
            for key, value in metadata.items():
                try:
                    metadata_dict[hash][key] = value.decode('UTF-8')
                except:
                    pass
    return metadata_dict


def calculate_features(image_folder, dataset_filepath, target_features=["dummy"]):
    image_dir = Path(image_folder)
    # check folder for all possible extenstions of images, we don't want to accidentally glob other files
    image_paths_png = list(image_dir.glob("*.png"))
    image_paths_jpg = list(image_dir.glob("*.jpg"))
    image_paths_jpeg = list(image_dir.glob("*.jpeg"))
    image_paths = image_paths_png + image_paths_jpg + image_paths_jpeg

    for feature in target_features:
        if feature == "dummy":
            dummy.calc_and_save_features(image_paths, dataset_filepath)
        if feature == "dino":
            dino.calc_and_save_features(image_paths, dataset_filepath)
        if feature == "texture":
            texture.calc_and_save_features(image_paths, dataset_filepath)
        if feature == "emotion":
            emotion.calc_and_save_features(image_paths, dataset_filepath)
        if feature == "semantic":
            semantic.calc_and_save_features(image_paths, dataset_filepath)


def create_dataset(image_folder, dataset_filepath="dataset.h5"):
    image_dir = Path(image_folder)
    # check folder for all possible extenstions of images, we don't want to accidentally glob other files
    image_paths_png = list(image_dir.glob("*.png"))
    image_paths_jpg = list(image_dir.glob("*.jpg"))
    image_paths_jpeg = list(image_dir.glob("*.jpeg"))
    image_paths = image_paths_png + image_paths_jpg + image_paths_jpeg

    shuffle(image_paths) # shuffle for more consistency in time estimate, otherwise it slows up and down

    # create dataset file and add hash for each image.
    # function overwrites existing dataset file on the same location by default
    with h5py.File(dataset_filepath, "w") as f:
        for image_path in tqdm(image_paths, desc="calculating image hashes", total=len(image_paths)):
            img_hash = get_image_hash(str(image_path))
            if f"{img_hash}/filename" in f:
                print(f"file {image_path} with hash {img_hash} already exists!")
            else:
                f.create_dataset(f"{img_hash}/filename", data=str(image_path.name))

    basepath = Path(__file__)
    pickle_path = (
            basepath.parents[2]
            / "data"
            / "raw_immutable"
            / "old_datasets"
            / "similarPaintings.pkl"
        )
    print(pickle_path)

    add_metadata_from_pickle(dataset_filepath, pickle_path, image_folder)


def read_dataset_keys(dataset_filepath):
    with h5py.File(dataset_filepath, "r") as f:
        keys = [key for key in f.keys()]
        key_filename_dict = {key: f[f"{key}/filename"][()].decode() for key in keys}
    return key_filename_dict


def load_image(image_path, return_np=True):
    """load image from filepath, useful for consistant loading in different parts of the repository

    Args:
        image_path (str): path to image file
        return_np (bool, optional): return numpy array instead of pillow image object. Defaults to True.

    Returns:
        PIL image or numpy array: return numpy array by default except when setting return_np to False
    """

    # read using pillow, remove alpha channel if its there
    img = Image.open(image_path).convert("RGB")
    # sometimes we might not want to conver to a numpy array, for vizualization for example.
    if not return_np:
        return img
    else:
        return np.array(img)

def get_image_hash(filename):
    with open(filename, 'rb', buffering=0) as f:
        return hashlib.file_digest(f, 'md5').hexdigest()