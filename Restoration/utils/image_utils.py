"""
Created on 2020/9/8

@author: Boyun Li
"""
import os

import cv2
import numpy as np
import torch
import random
import torch.nn as nn
from torch.nn import init
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from collections import Counter


# Function to count occurrences
def count_category_and_dataset(dict_list):
    # Extract the Category and Dataset values
    category_values = [d['category'] for d in dict_list if 'category' in d]
    dataset_values = [d['dataset'] for d in dict_list if 'dataset' in d]

    # Count occurrences of each unique value
    category_count = Counter(category_values)
    dataset_count = Counter(dataset_values)

    # Print the counts for Category
    print("Category Counts:")
    for category, count in category_count.items():
        print(f"{category}: {count}")

    # Print the counts for Dataset
    print("\nDataset Counts:")
    for dataset, count in dataset_count.items():
        print(f"{dataset}: {count}")


# randomly crop a patch from image
def crop_patch(im, pch_size):
    H = im.shape[0]
    W = im.shape[1]
    ind_H = random.randint(0, H - pch_size)
    ind_W = random.randint(0, W - pch_size)
    pch = im[ind_H:ind_H + pch_size, ind_W:ind_W + pch_size]
    return pch


# crop an image to the multiple of base
def crop_img(image, base=64):
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]

def unnormalize_image(normalized_image, means, stds):
    # Ensure the normalized_image has 3 channels (for RGB images)
    normalized_image = normalized_image.transpose(1, 2, 0)
    if normalized_image.shape[2] != 3:
        raise ValueError("Input normalized image should have 3 channels (RGB).")

    # Unnormalize each channel
    unnormalized_image = np.zeros_like(normalized_image, dtype=np.float32)
    for i in range(3):
        unnormalized_image[:, :, i] = (normalized_image[:, :, i] * stds[i]) + means[i]
    unnormalized_image *= 255
    # Clip values to be within valid image range (0 to 255)
    unnormalized_image = np.clip(unnormalized_image, 0, 255).astype(np.uint8)

    return unnormalized_image

def debug_images(img, num, save_dir='results_input/', mode='normal'):
    B, _, _, _ = img.size()
    img = img.detach().cpu().numpy()
    if mode == 'clip':
        means = [0.48145466, 0.4578275, 0.40821073]
        stds = [0.26862954, 0.26130258, 0.27577711]
        for idx in range(B):
            img_norm = unnormalize_image(np.asarray(img[idx]), means, stds)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            cv2.imwrite(save_dir + str(num) + '_' + str(idx) + '_img.png', img_norm)

    elif mode == 'normal':
        for idx in range(B):
            degrad = np.asarray(img[idx]) * 255
            degrad = degrad.transpose(1, 2, 0)

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            degrad = degrad.astype(np.uint8)
            degrad = cv2.cvtColor(degrad, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_dir + str(num) + '_' + str(idx) + '_img.png', degrad)

def unnormalize(x):
    unnorm = []
    for img in x:
        means = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1).cuda()
        stds = torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1).cuda()
        img_norm = (img * stds) + means
        unnorm.append(img_norm)

    return unnorm

def resize_img(image, size=512):
    image = image.resize(size, Image.LANCZOS)
    return image

def resize_img_crop(image, idx, min_size=128):
    h, w, _ = image.shape
    new_h = h
    new_w = w
    if h < min_size:
        new_h = min_size
    if w < min_size:
        new_w = min_size
    # print(new_h, new_w, h, w)
    if new_h != h or new_w != w:
        print("Changed")
        image = np.array(Image.fromarray(image).resize((new_w, new_h)))
        # print("old shape", h, w, idx)
        # print("new shape", image.shape, idx)
    return image


# image (H, W, C) -> patches (B, H, W, C)
def slice_image2patches(image, patch_size=64, overlap=0):
    assert image.shape[0] % patch_size == 0 and image.shape[1] % patch_size == 0
    H = image.shape[0]
    W = image.shape[1]
    patches = []
    image_padding = np.pad(image, ((overlap, overlap), (overlap, overlap), (0, 0)), mode='edge')
    for h in range(H // patch_size):
        for w in range(W // patch_size):
            idx_h = [h * patch_size, (h + 1) * patch_size + overlap]
            idx_w = [w * patch_size, (w + 1) * patch_size + overlap]
            patches.append(np.expand_dims(image_padding[idx_h[0]:idx_h[1], idx_w[0]:idx_w[1], :], axis=0))
    return np.concatenate(patches, axis=0)


def data_augmentation(image, mode):
    if mode == 0:
        # original
        out = image.numpy()
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')
    return out


def random_augmentation(*args):
    out = []
    flag_aug = random.randint(1, 7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out


def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()
    # return img_var.detach().cpu().numpy()[0]


def save_image(name, image_np, output_path="output/normal/"):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    p = np_to_pil(image_np)
    p.save(output_path + "{}.png".format(name))


def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)