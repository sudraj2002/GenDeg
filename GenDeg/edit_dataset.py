from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
from get_samples_mu_sigma_nested import *

class EditDataset(Dataset):
    # For training
    def __init__(
            self,
            path: str,
            split: str = "train",
            splits: tuple[float, float, float] = (0.98, 0.01, 0.01),
            min_resize_res: int = 256,
            max_resize_res: int = 256,
            crop_res: int = 256,
            flip_prob: float = 0.0,
            replace_dir: str = '/home/sambasa2/'
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        self.replace_dir = replace_dir

        with open(Path(self.path, "seeds.json")) as f:
            # The .json files contains paths of degraded and target images
            self.seeds = json.load(f)

        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]

        ## Modified
        self.prompts = json.load(open("train_csv.json"))  # Contains prompts for each clean image generated using BLIP-2

        # Precalculated
        self.mu_min = 0.0
        self.mu_max = 0.7

        self.sigma_min = 0.0
        self.sigma_max = 0.4

        self.num_bins = 128  # Does not include null prompt's bin

        self.mu_bins = self.get_bin(self.mu_min, self.mu_max, self.num_bins)
        self.sigma_bins = self.get_bin(self.sigma_min, self.sigma_max, self.num_bins)

    ## Modified
    def get_prompt_from_image(self, gt_name):
        # Get prompt for a ground truth path
        key = 'filepath'
        result = next((d for d in self.prompts if d.get(key) == gt_name), None)

        return result['title']

    def get_bin(self, start, end, num_bins):
        # Create bin edges
        bin_edges = np.linspace(start, end, num_bins + 1)

        return bin_edges

    def bin_value(self, value, bin_edges):
        """
        value: value to bin
        bin_edges: created bins

        bins the value
        """

        # Find the bin index for the value
        bin_index = np.digitize(value, bin_edges, right=False) - 1  # 0-based indexing

        # Handle edge cases
        if bin_index < 0:
            raise ValueError(f"Value {value} is below the min range.")
        elif bin_index >= self.num_bins:
            bin_index = self.num_bins - 1  # Assign to the last regular bin

        return bin_index

    def one_hot_encode(self, bin_index):
        # Generates a one-hot encoded vector for the given bin index.
        one_hot = np.zeros(self.num_bins + 1, dtype=int)
        one_hot[bin_index] = 1
        return one_hot

    def bin_and_one_hot_encode(self, value, bin_edges):
        # Bins a value
        bin_index = self.bin_value(value, bin_edges)
        one_hot = self.one_hot_encode(bin_index)
        return bin_index, one_hot

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict[str, Any]:

        ## Modified
        line = self.seeds[i]
        degraded_name = line['image_path']  # For image_1
        target_name = line['target_path']  # For image_0

        # Get the prompt, modify it for mixed degradations
        prompt = self.get_prompt_from_image(target_name)
        if line['dataset'] == 'CSD':
            prompt = prompt.replace('in snowy conditions', 'in snowy and hazy conditions')

        if line['dataset'] == 'ORD':
            prompt = prompt.replace('in rainy conditions', 'in rainy and hazy conditions')

        degraded_name = degraded_name.replace('/data/', self.replace_dir)
        target_name = target_name.replace('/data/', self.replace_dir)

        image_0 = Image.open(target_name).convert('RGB')  # clean image--> condition
        image_1 = Image.open(degraded_name).convert('RGB')  # Degraded image--> to predict

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        # Add mu and sigma to condition
        diff_img = np.abs((np.array(image_0) / 255) - (np.array(image_1) / 255))
        diff_img = diff_img.flatten()  # cmap


        # Get mu and sigma
        mu = diff_img.mean()
        sigma = diff_img.std()

        # bin and one-hot encode
        _, one_hot_mu = self.bin_and_one_hot_encode(mu, self.mu_bins)
        _, one_hot_sigma = self.bin_and_one_hot_encode(sigma, self.sigma_bins)

        one_hot_mu = torch.tensor(one_hot_mu).float()
        one_hot_sigma = torch.tensor(one_hot_sigma).float()
        stats = torch.cat([one_hot_mu[None, :], one_hot_sigma[None, :]], dim=0)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt, c_stats=stats)) # Condition image is clean


class EditDatasetParallelMuSigma(Dataset):
    # For generating images batch-wise on a single GPU
    def __init__(
            self,
            path: str,
            deg_type: str,
            split: str = "train",
            splits: tuple[float, float, float] = (0.98, 0.01, 0.01),
            min_resize_res: int = 256,
            max_resize_res: int = 256,
            crop_res: int = 256,
            res: int = 256,
            flip_prob: float = 0.0,
            random_every: int = 20,
            replace_dir: str = '/home/sambasa2/',
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.deg_type = deg_type  # Degradation to generate
        self.res = res
        self.augment = True
        self.replace_dir = replace_dir

        # Automatically add the prompts
        if deg_type == 'rain':
            replace_text = ' in rainy conditions'
        elif deg_type == 'snow':
            replace_text = ' in snowy conditions'
        elif deg_type == 'haze':
            replace_text = ' in hazy conditions'
        elif deg_type == 'motion':
            replace_text = ' in blurry conditions'
            deg_type = 'motion blur'
        elif deg_type == 'low-light':
            replace_text = ' in low-light conditions'  ### CHANGE THIS TO WHICH EVER
        elif deg_type == 'raindrop':
            replace_text = ' with raindrops in it'
        elif 'and' in deg_type:
            replace_text = f" {deg_type}"
        else:
            raise ValueError("invalid degradation")

        self.replace_text = replace_text
        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)
        unique_paths = set()
        self.gt_seeds = []
        for line in self.seeds:
            if line['target_path'] not in unique_paths:
                if self.augment:  # Augment only i.e. don't generate degradation for already existing category of that degradation
                    if line['degradation'].lower() == deg_type:
                        continue
                self.gt_seeds.append(line)
                unique_paths.add(line['target_path'])

        self.seeds = self.gt_seeds

        print(f"Creating dataset for {len(self.seeds)} clean images.")

        ## Modified
        self.prompts = json.load(open("train_csv.json"))

        # Precalculated
        self.mu_min = 0.0
        self.mu_max = 0.7

        self.sigma_min = 0.0
        self.sigma_max = 0.4

        self.num_bins = 128  # Does not include null prompt's bin

        self.mu_bins = self.get_bin(self.mu_min, self.mu_max, self.num_bins)
        self.sigma_bins = self.get_bin(self.sigma_min, self.sigma_max, self.num_bins)

        self.counter = 0  # Counter to check number of traversed samples
        self.random_every = random_every

        self.dataset_stats_mu = load_data(path, deg_type, filter='Mu_')
        self.dataset_stats_sigma = load_data(path, deg_type, filter='Sigma_')
        self.dataset_stats_mu_sigma = load_data(path, deg_type, filter='Mu-Sigma-')
        self.all_datasets = list(self.dataset_stats_mu.keys())
        self.all_datasets.sort()
        self.ds_dict = {ds: 1 / len(self.all_datasets) for ds in self.all_datasets}

        # Some datasets have intense degradations, we don't want these to dominate the generated dataset
        if deg_type.lower() == 'rain':
            self.ds_dict['ORD'] = 0.2 * 1 / len(self.all_datasets)
        elif deg_type.lower() == 'haze':
            self.ds_dict['DenseHaze'] = 0.05 * 1 / len(self.all_datasets)
            self.ds_dict['NH-HAZE'] = 0.1 * 1 / len(self.all_datasets)
            self.ds_dict['I-HAZE'] = 0.25 * 1 / len(self.all_datasets)
        elif deg_type.lower() == 'low-light':
            self.ds_dict['SID'] = 0.2 * 1 / len(self.all_datasets)

        self.all_datasets = list(self.ds_dict.keys())
        self.sample_weights = list(self.ds_dict.values())


    def get_bin(self, start, end, num_bins):
        # Create bin edges
        bin_edges = np.linspace(start, end, num_bins + 1)

        return bin_edges

    def bin_value(self, value, bin_edges):
        # Find the bin index for the value
        bin_index = np.digitize(value, bin_edges, right=False) - 1  # 0-based indexing

        # Handle edge cases
        if bin_index < 0:
            raise ValueError(f"Value {value} is below the min range.")
        elif bin_index >= self.num_bins:
            bin_index = self.num_bins - 1  # Assign to the last regular bin

        return bin_index

    def one_hot_encode(self, bin_index):
        one_hot = np.zeros(self.num_bins + 1, dtype=int)
        one_hot[bin_index] = 1
        return one_hot

    def bin_and_one_hot_encode(self, value, bin_edges):
        bin_index = self.bin_value(value, bin_edges)
        one_hot = self.one_hot_encode(bin_index)
        return bin_index, one_hot

    ## Modified
    def get_prompt_from_image(self, gt_name):
        key = 'filepath'
        result = next((d for d in self.prompts if d.get(key) == gt_name), None)
        result = result['title']
        id_text = result.split(',')[-1]
        assert 'conditions' in id_text or 'raindrops in it' in id_text
        result = result.replace(id_text, self.replace_text)

        return result

    def get_mu_sigma(self):
        # Sample mu and sigma based on strategy in Sec. 3.2 of paper
        if self.counter % self.random_every == 0:
            # Sampler random
            random_sample = True
        else:
            random_sample = False
        dataset = random.choices(self.all_datasets, weights=self.sample_weights, k=1)[0]
        mu, sigma = sample_dataset_stat(dataset=dataset, dataset_dict_mu=self.dataset_stats_mu, random_sample=random_sample,
                                        dataset_dict_sigma=self.dataset_stats_sigma,
                                        dataset_dict_mu_sigma=self.dataset_stats_mu_sigma)

        return mu, sigma, dataset, random_sample


    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict[str, Any]:
        ## Modified
        line = self.seeds[i]
        degraded_name = line['image_path']  # For image_1
        target_name = line['target_path']  # For image_0
        traversed_name = target_name

        prompt = self.get_prompt_from_image(target_name)

        # Mu, sigma sampler
        mu, sigma, sampled_dataset, random_sampled = self.get_mu_sigma()
        _, one_hot_mu = self.bin_and_one_hot_encode(mu, self.mu_bins)
        _, one_hot_sigma = self.bin_and_one_hot_encode(sigma, self.sigma_bins)

        one_hot_mu = torch.tensor(one_hot_mu).float()
        one_hot_sigma = torch.tensor(one_hot_sigma).float()
        stats = torch.cat([one_hot_mu[None, :], one_hot_sigma[None, :]], dim=0)

        # Make sure mixed dataset is correctly handled
        if sampled_dataset == 'ORD':
            prompt = prompt.replace('rainy', 'rainy and hazy')
        if sampled_dataset == 'CSD':
            prompt = prompt.replace('snowy', 'snowy and hazy')

        degraded_name = degraded_name.replace('/data/', self.replace_dir)
        target_name = target_name.replace('/data/', self.replace_dir)
        image_0 = Image.open(target_name).convert('RGB')
        image_1 = Image.open(degraded_name).convert('RGB')

        original_size = image_0.size

        width, height = self.res, self.res

        image_0 = image_0.resize((width, height), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((width, height), Image.Resampling.LANCZOS)


        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        dir_path = line['target_path'].split(f'{self.replace_dir}GenIRData/')[-1]

        name = target_name.split('/')[-1]
        dir_path = dir_path.split(name)[0]

        self.counter += 1

        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt, c_stats=stats), dataset=line['dataset'],
                    degradation=line['degradation'], category=line['category'], name=target_name.split('/')[-1],
                    dir_path=dir_path, split_name=line['split'], orig_size=original_size, mu=mu, sigma=sigma,
                    sampled_dataset=sampled_dataset, random_sampled=random_sampled, traversed_name=traversed_name)


class EditDatasetSingle(Dataset):
    def __init__(
            self,
            # path: str,
            deg_type: str,
            split: str = "train",
            splits: tuple[float, float, float] = (0.98, 0.01, 0.01),
            min_resize_res: int = 256,
            max_resize_res: int = 256,
            crop_res: int = 256,
            res: int = 256,
            flip_prob: float = 0.0,
            random_every: int = 20,
            mu=None,
            sigma=None,
            prompt=None,
            name=None
    ):
        # For editing single image
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        # self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.deg_type = deg_type
        self.res = res
        self.augment = True
        self.degraded_name = self.target_name = name
        self.prompt = prompt

        if mu is not None and sigma is not None:
            self.mu = mu
            self.sigma = sigma

            print(f"Using mu={self.mu}, sigma={self.sigma}")

        else:
            self.mu = None
            self.sigma = None
            print(f"mu and sigma will be sampled from existing data")

        print(f"Creating dataset for 1 image.")

        # Precalculated
        self.mu_min = 0.0
        self.mu_max = 0.7

        self.sigma_min = 0.0
        self.sigma_max = 0.4

        self.num_bins = 128  # Does not include null prompt's bin

        self.mu_bins = self.get_bin(self.mu_min, self.mu_max, self.num_bins)
        self.sigma_bins = self.get_bin(self.sigma_min, self.sigma_max, self.num_bins)

        # If mu and sigma were not provided, sample them using same strategy
        if self.mu is None and self.sigma is None:
            if deg_type == 'rain':
                replace_text = ' in rainy conditions'
            elif deg_type == 'snow':
                replace_text = ' in snowy conditions'
            elif deg_type == 'haze':
                replace_text = ' in hazy conditions'
            elif deg_type == 'motion':
                replace_text = ' in blurry conditions'
                deg_type = 'motion blur'
            elif deg_type == 'low-light':
                replace_text = ' in low-light conditions'
            elif deg_type == 'raindrop':
                replace_text = ' with raindrops in it'
            elif 'and' in deg_type:
                replace_text = f" {deg_type}"
            else:
                raise ValueError("invalid degradation")


            self.random_every = random_every
            self.dataset_stats_mu = load_data("data", deg_type, filter='Mu_')
            self.dataset_stats_sigma = load_data("data", deg_type, filter='Sigma_')
            self.dataset_stats_mu_sigma = load_data("data", deg_type, filter='Mu-Sigma-')
            self.all_datasets = list(self.dataset_stats_mu.keys())
            self.all_datasets.sort()
            self.ds_dict = {ds: 1 / len(self.all_datasets) for ds in self.all_datasets}
            if deg_type.lower() == 'rain':
                self.ds_dict['ORD'] = 0.2 * 1 / len(self.all_datasets)
            elif deg_type.lower() == 'haze':
                self.ds_dict['DenseHaze'] = 0.05 * 1 / len(self.all_datasets)
                self.ds_dict['NH-HAZE'] = 0.1 * 1 / len(self.all_datasets)
                self.ds_dict['I-HAZE'] = 0.25 * 1 / len(self.all_datasets)
            elif deg_type.lower() == 'low-light':
                self.ds_dict['SID'] = 0.2 * 1 / len(self.all_datasets)

            self.all_datasets = list(self.ds_dict.keys())
            self.sample_weights = list(self.ds_dict.values())


    def get_bin(self, start, end, num_bins):
        # Create bin edges
        bin_edges = np.linspace(start, end, num_bins + 1)

        return bin_edges

    def bin_value(self, value, bin_edges):
        """
        value: value to bin
        bin_edges: created bins
        """

        # Find the bin index for the value
        bin_index = np.digitize(value, bin_edges, right=False) - 1  # 0-based indexing

        # Handle edge cases
        if bin_index < 0:
            raise ValueError(f"Value {value} is below the min range.")
        elif bin_index >= self.num_bins:
            bin_index = self.num_bins - 1  # Assign to the last regular bin

        return bin_index

    def one_hot_encode(self, bin_index):
        one_hot = np.zeros(self.num_bins + 1, dtype=int)
        one_hot[bin_index] = 1
        return one_hot

    def bin_and_one_hot_encode(self, value, bin_edges):
        bin_index = self.bin_value(value, bin_edges)
        one_hot = self.one_hot_encode(bin_index)
        return bin_index, one_hot

    ## Modified
    def get_prompt_from_image(self, gt_name):
        key = 'filepath'
        result = next((d for d in self.prompts if d.get(key) == gt_name), None)
        result = result['title']
        id_text = result.split(',')[-1]
        assert 'conditions' in id_text or 'raindrops in it' in id_text
        result = result.replace(id_text, self.replace_text)

        return result

    def get_mu_sigma(self):
        random_sample = False
        dataset = random.choices(self.all_datasets, weights=self.sample_weights, k=1)[0]
        mu, sigma = sample_dataset_stat(dataset=dataset, dataset_dict_mu=self.dataset_stats_mu, random_sample=random_sample,
                                        dataset_dict_sigma=self.dataset_stats_sigma,
                                        dataset_dict_mu_sigma=self.dataset_stats_mu_sigma)

        return mu, sigma, dataset, random_sample


    def __len__(self) -> int:
        return 1

    def __getitem__(self, i: int) -> dict[str, Any]:

        # Mu, sigma sampler
        if self.mu is None and self.sigma is None:
            mu, sigma, sampled_dataset, random_sampled = self.get_mu_sigma()
            _, one_hot_mu = self.bin_and_one_hot_encode(mu, self.mu_bins)
            _, one_hot_sigma = self.bin_and_one_hot_encode(sigma, self.sigma_bins)
        else:
            mu = self.mu
            sigma = self.sigma
            _, one_hot_mu = self.bin_and_one_hot_encode(mu, self.mu_bins)
            _, one_hot_sigma = self.bin_and_one_hot_encode(sigma, self.sigma_bins)

        one_hot_mu = torch.tensor(one_hot_mu).float()
        one_hot_sigma = torch.tensor(one_hot_sigma).float()
        stats = torch.cat([one_hot_mu[None, :], one_hot_sigma[None, :]], dim=0)

        image_0 = Image.open(self.target_name).convert('RGB')
        image_1 = Image.open(self.degraded_name).convert('RGB')

        width, height = self.res, self.res

        image_0 = image_0.resize((width, height), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((width, height), Image.Resampling.LANCZOS)


        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=self.prompt, c_stats=stats),
                    mu=mu, sigma=sigma, name=self.target_name)

