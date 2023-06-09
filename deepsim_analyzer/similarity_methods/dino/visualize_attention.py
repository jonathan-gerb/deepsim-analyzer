# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import random
import colorsys
import requests
from io import BytesIO
from pathlib import Path

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
import cv2

from . import vision_transformer as vits

MODEL_PATH_DICT = {
    "vit_small8": "dino_deitsmall8_pretrain_full_checkpoint.pth",
    "vit_small16": "dino_deitsmall16_pretrain_full_checkpoint.pth",
    "vit_base16": "dino_vitbase16_pretrain_full_checkpoint.pth",
    "vit_base8": "dino_vitbase8_pretrain_full_checkpoint.pth",
}


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = (
            image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
        )
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(
    image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5
):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis("off")
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask, (10, 10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect="auto")
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


def load_model(arch, pretrained_model_path, patch_size):
    # get model filename
    model_filename = MODEL_PATH_DICT[f"{arch}{patch_size}"]

    # get path to model relatively
    if pretrained_model_path == "":
        fp = os.path.dirname(os.path.realpath(__file__))
        fp = Path(fp)
        full_path = fp.parents[2] / "data" / "raw_immutable" / "models" / model_filename
    else:
        full_path = pretrained_model_path
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)

    state_dict = torch.load(full_path, map_location="cpu")
    state_dict = state_dict["teacher"]

    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print(
        "Pretrained weights found at {} and loaded with msg: {}".format(
            full_path, msg
        )
    )
    return model


def get_attention_maps(model, image):
    attention_dict = model.get_all_self_attention(image)
    return attention_dict


def get_att_feature_vector(attention_dict):
    nlayers = len(attention_dict)
    nheads = attention_dict[0].shape[0]  # all layers have an equal amount of heads

    feature_vector = np.zeros(nheads * nlayers)

    for layer, attentions in attention_dict.items():
        for head in range(attentions.shape[0]):
            magnitude = np.linalg.norm(attentions[head])

            idx = (layer * nheads) + head
            feature_vector[idx] = magnitude
    return feature_vector


def get_feature_maps(attention_dict, resize_to=False):
    feature_maps = {}
    for layer_idx, attentions in attention_dict.items():
        nh = attentions.shape[0]
        feature_maps[layer_idx] = {}
        for head_idx in range(nh):
            if type(resize_to) != bool:
                feature_maps[layer_idx][head_idx] = cv2.resize(
                    attentions[head_idx], dsize=resize_to, interpolation=cv2.INTER_NEAREST
                )
            else:
                feature_maps[layer_idx][head_idx] = attentions[head_idx]
    return feature_maps


def get_features(
    image,
    patch_size=8,
    arch="vit_base",
    image_size=(480, 480),
    threshold=None,
    pretrained_model_path="",
    return_feature_maps=False,
    resize=True,
):
    model = load_model(arch, pretrained_model_path, patch_size)
    if resize:
        resize = image.size

    transform = pth_transforms.Compose(
        [
            pth_transforms.Resize(image_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    image = transform(image)

    # make the image divisible by the patch size
    w, h = (
        image.shape[1] - image.shape[1] % patch_size,
        image.shape[2] - image.shape[2] % patch_size,
    )
    image = image[:, :w, :h].unsqueeze(0)

    w_featmap = image.shape[-2] // patch_size
    h_featmap = image.shape[-1] // patch_size

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # move model and image to gpu (if available)
    model = model.to(device)
    image = image.to(device)

    attention_dict = get_attention_maps(model, image)

    image = image.to("cpu")

    for key, attentions in attention_dict.items():
        nh = attentions.shape[1]  # number of head

        # we keep only the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
        # filter attention maps
        if threshold is not None:
            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            th_attn = (
                nn.functional.interpolate(
                    th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest"
                )[0]
                .cpu()
                .numpy()
            )

        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = (
            nn.functional.interpolate(
                attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest"
            )[0]
            .cpu()
            .numpy()
        )
        attention_dict[key] = attentions

    feature_vector = get_att_feature_vector(attention_dict)

    # torchvision.utils.make_grid(img, normalize=True, scale_each=True)

    if return_feature_maps:
        feature_maps = get_feature_maps(attention_dict, resize_to=resize)

        return feature_vector, feature_maps
    else:
        return feature_vector


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize Self-Attention maps")
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=["vit_tiny", "vit_small", "vit_base"],
        help="Architecture (support only ViT atm).",
    )
    parser.add_argument(
        "--patch_size", default=8, type=int, help="Patch resolution of the model."
    )
    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="Path to pretrained weights to load.",
    )
    parser.add_argument(
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")',
    )
    parser.add_argument(
        "--image_path", default=None, type=str, help="Path of the image to load."
    )
    parser.add_argument(
        "--image_size", default=(480, 480), type=int, nargs="+", help="Resize image."
    )
    parser.add_argument(
        "--output_dir", default=".", help="Path where to save visualizations."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""",
    )
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(
                args.pretrained_weights, msg
            )
        )
    else:
        print(
            "Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate."
        )
        url = None
        if args.arch == "vit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "vit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print(
                "Since no pretrained weights have been provided, we load the reference pretrained DINO weights."
            )
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/" + url
            )
            model.load_state_dict(state_dict, strict=True)
        else:
            print(
                "There is no reference weights available for this model => We use random weights."
            )

    # open image
    if args.image_path is None:
        # user has not specified any image - we use our own image
        print(
            "Please use the `--image_path` argument to indicate the path of the image you wish to visualize."
        )
        print(
            "Since no image path have been provided, we take the first image in our paper."
        )
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert("RGB")
    elif os.path.isfile(args.image_path):
        with open(args.image_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)
    transform = pth_transforms.Compose(
        [
            pth_transforms.Resize(args.image_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    img = transform(img)

    # make the image divisible by the patch size
    w, h = (
        img.shape[1] - img.shape[1] % args.patch_size,
        img.shape[2] - img.shape[2] % args.patch_size,
    )
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size

    # attentions = model.get_last_selfattention(img.to(device))
    attentions = model.get_n_selfattention(img.to(device), n=1)

    nh = attentions.shape[1]  # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    if args.threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = (
            nn.functional.interpolate(
                th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest"
            )[0]
            .cpu()
            .numpy()
        )

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = (
        nn.functional.interpolate(
            attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest"
        )[0]
        .cpu()
        .numpy()
    )

    # save attentions heatmaps
    os.makedirs(args.output_dir, exist_ok=True)
    torchvision.utils.save_image(
        torchvision.utils.make_grid(img, normalize=True, scale_each=True),
        os.path.join(args.output_dir, "img.png"),
    )
    for j in range(nh):
        fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format="png")
        print(f"{fname} saved.")

    if args.threshold is not None:
        image = skimage.io.imread(os.path.join(args.output_dir, "img.png"))
        for j in range(nh):
            display_instances(
                image,
                th_attn[j],
                fname=os.path.join(
                    args.output_dir,
                    "mask_th" + str(args.threshold) + "_head" + str(j) + ".png",
                ),
                blur=False,
            )
