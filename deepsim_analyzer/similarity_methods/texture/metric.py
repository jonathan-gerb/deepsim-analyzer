import numpy as np
import torch
from tqdm import tqdm


def calc_and_save_features(images, dataset_filepath, save_feature_maps=False):
    """Process list of images.
    Args:
        images (list): list of filepaths to the images
        datafile_path (str): path to the dataset file to save the outputs to
    """
    # local import inside function to avoid circular import problem
    from deepsim_analyzer.io import get_image_hash, load_image, save_feature
    device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    model = torch.hub.load("pytorch/vision:v0.10.0", "googlenet", pretrained=True)
    model = model.to(device)

    for image_path in tqdm(
        images, desc=f"calculating texture features", total=len(images)
    ):
        image_path = str(image_path)  # in case image_path is a pathlib path


        hash = get_image_hash(image_path)
        img = load_image(image_path, return_np=False)

        # resize all images to the same size to get consistent magnitudes when summing 
        # the activation values. the distortion added by doing so shouldnt be too bad
        img = img.resize((244, 244))

        img_array = np.array(img)[np.newaxis, :]

        img_tensor = torch.tensor(img_array, dtype=torch.float32)
        # change to channel first
        img_tensor = img_tensor.permute(0, 3, 1, 2)
        img_tensor = img_tensor.to(device)

        out = model.conv1(img_tensor)
        out = model.maxpool1(out)
        out = model.conv2(out)
        out = model.conv3(out)
        out = model.maxpool2(out)
        out_3a = model.inception3a(out)
        out_3b = model.inception3b(out_3a).cpu().detach()
        
        # move out_3a result to cpu as well after using it.
        out_3a = out_3a.cpu().detach()

        b_a, c_a, _, _ = out_3a.shape
        b_b, c_b, _, _ = out_3b.shape
        out_3a_summed = torch.sum(out_3a.reshape(b_a, c_a, -1), axis=-1).squeeze()
        out_3b_summed = torch.sum(out_3b.reshape(b_b, c_b, -1), axis=-1).squeeze()

        total_vec = torch.concatenate([out_3a_summed, out_3b_summed])
        feature_vector = total_vec.numpy()

        if save_feature_maps:
            save_feature(dataset_filepath, hash, out_3a.numpy(), "texture_fm_3a")
            save_feature(dataset_filepath, hash, out_3b.numpy(), "texture_fm_3b")

        save_feature(dataset_filepath, hash, feature_vector, "texture")
