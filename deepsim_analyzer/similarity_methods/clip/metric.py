import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import clip


def calc_and_save_features(images, dataset_filepath, save_feature_maps=True, overwrite=True):
    """Process list of images.
    Args:
        images (list): list of filepaths to the images
        datafile_path (str): path to the dataset file to save the outputs to
    """
    # local import inside function to avoid circular import problem
    from deepsim_analyzer.io import get_image_hash, load_image, save_feature, key_in_dataset
    device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    model, preprocess = clip.load("ViT-B/32", device=device)
    model = model.to(device)
    skipped = 0
    calculated = 0 
    
    with torch.no_grad():
        # text_features = model.encode_text(text)

        for image_path in tqdm(
            images, desc=f"calculating CLIP features", total=len(images)
        ):

            img_hash = get_image_hash(image_path)

            if key_in_dataset(dataset_filepath, f"{img_hash}/features/clip/full") and not overwrite:
                skipped += 1
                continue
            calculated += 1

            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            image_features = model.encode_image(image).squeeze().cpu().detach().numpy()
            save_feature(dataset_filepath, img_hash, image_features, "clip")
    
    print(f"skipped: {skipped}, caluclated: {calculated}")

def calc_features(image_path, dataset_filepath, save_feature_maps=True):
    """Process list of images.
    Args:
        image_path: filepath to the image
        datafile_path (str): path to the dataset file to save the outputs to
    """
    # local import inside function to avoid circular import problem
    from deepsim_analyzer.io import get_image_hash, load_image, save_feature
    device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    model, preprocess = clip.load("ViT-B/32", device=device)
    model = model.to(device)
    
    with torch.no_grad():
        # text_features = model.encode_text(text)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_features = model.encode_image(image).squeeze().cpu().detach().numpy()
    return image_features
