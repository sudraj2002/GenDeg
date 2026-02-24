import os
import random
from copy import deepcopy
from distutils.command.clean import clean

from PIL import Image
import numpy as np
import warnings

from torch.utils.data import Dataset

from utils.image_utils import random_augmentation, crop_img, resize_img, resize_img_crop, count_category_and_dataset, \
    unnormalize_image
import json
import cv2

from torchvision.transforms import Resize

from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from termcolor import colored

DEGRADATIONS = {"Haze": 1, "Rain": 2, "Snow": 3, "Raindrop": 4, "Low-light": 5, "Motion blur": 6, "toled": 7, "poled": 8}

class BaseDataset(Dataset):
    def __init__(self, args):
        super(BaseDataset, self).__init__()

        self.args = args
        self.resize = args.size

        self.imagenet_norm = self.args.imagenet_norm

        self.json_path = self.args.json_path
        self.ids = json.load(open(self.json_path))
        degradation = self.ids[0]['degradation']
        dataset = self.ids[0]['dataset']

        self.degradation = degradation
        self.train = False
        if self.train:
            random.shuffle(self.ids)
            new_ids = self.ids.copy()
            self.ids = []
            for line in new_ids:
                if 'degradation_sub_type' not in line:
                    continue
                if line['degradation_sub_type'] != 'raindrop':
                    continue
                if line['sampled_dataset'] != 'Raindrop':
                    continue
                self.ids.append(line)
        print(colored(f'\nInitializing test dataset for {degradation} and {dataset} of length {len(self.ids)}', 'green'))
        # Supported degradations
        self.valid_degradations = list(DEGRADATIONS.keys())

        # assert degradation in self.valid_degradations, "Invalid degradation encountered"

        if self.imagenet_norm:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transform = transforms.ToTensor()

        if args.size is not None:
            self.size_h = int(args.size[0])
            self.size_w = int(args.size[1])

    def get_new_name(self, name, deg_sub_type, degradation):
        if deg_sub_type == 'Original':
            return name.replace('/data/', self.args.new_dir)
        else:
            old_name = name
            common_dir = name.split('/' + degradation + '/')[0]
            # actual_path = name.split('/' + degradation + '/')[-1]
            actual_path = name.split(os.path.join(common_dir, degradation))[-1]
            actual_path = actual_path.lstrip('/')
            sub_part_dir = common_dir.split('/')[-1]

            file_path = os.path.join(sub_part_dir, degradation, actual_path)
            to_replace = name.split(file_path)[0]
            name = name.replace(to_replace, self.args.new_dir)

            assert os.path.exists(name), f"Path {name} does not exist"
            return name

    def __getitem__(self, idx):
        sample = self.ids[idx]
        degraded_name, clean_name = sample['image_path'], sample['target_path']
        if self.train:
            self.args.new_dir = '/data/GenIRAugmented/'
            if 'degradation_sub_type' not in sample:
                sample['degradation_sub_type'] = 'Original'
            degraded_name = self.get_new_name(degraded_name, sample['degradation_sub_type'],
                                                   sample['degradation'])
            clean_name = self.get_new_name(clean_name, sample['degradation_sub_type'],
                                                    sample['degradation'])
            degradation_sub_type = sample['degradation_sub_type']
        else:
            degradation_sub_type = 'Original'

        # assert sample["split"].lower() == 'test'

        ds = sample["dataset"]
        category = sample["category"] if 'category' in sample else 'Real'

        # print(degraded_name, clean_name)

        if clean_name is None or not os.path.exists(clean_name):
            raise FileNotFoundError(f"Image path {clean_name} for {self.degradation, ds, category} is invalid or does not exist.")

        if degraded_name is None or not os.path.exists(degraded_name):
            raise FileNotFoundError(f"Image path {degraded_name} for {self.degradation, ds, category} is invalid or does not exist.")

        degrad_img = Image.open(degraded_name).convert('RGB')
        clean_img = Image.open(clean_name).convert('RGB')

        if not self.resize:
            degrad_img = crop_img(np.array(degrad_img), base=16)
            clean_img = crop_img(np.array(clean_img), base=16)
        else:
            degrad_img = resize_img(degrad_img, (self.size_h, self.size_w))
            clean_img = resize_img(clean_img, (self.size_h, self.size_w))

        degrad_img = np.array(degrad_img)
        clean_img = np.array(clean_img)

        clean_img = self.transform(clean_img)
        degrad_img = self.transform(degrad_img)

        if self.args.text is not None:
            return [clean_name, degraded_name, ds, category], degrad_img, clean_img, degradation_sub_type, self.args.text
        else:
            return [clean_name, degraded_name, ds, category], degrad_img, clean_img, degradation_sub_type, 'none'

    def __len__(self):
        return len(self.ids)

class BaseDatasetTxt(Dataset):
    def __init__(self, degradation, args):
        super(BaseDatasetTxt, self).__init__()
        self.ALL_SENTENCES = {
            "haze": ["Remove the haze from this image.", "Dehaze this image.",
                     "Improve the visibility of this image. It is very hazy."],
            "rain": ["Remove the rain from this image.", "Derain this image.", "Remove the rain streaks from this image."],
            "rain_and_haze": ["Remove the rain and haze from this image.", "Derain and dehaze this image.",
                              "Remove the rain streaks and haze from this image."],
            "snow": ["Remove the snow from this image.", "Desnow this image.", "Remove the snow flakes from this image."],
            "snow_and_haze": ["Remove the snow and haze from this image.", "Desnow and dehaze this image.",
                              "Remove the snow flakes and haze from this image."],
            "low-light": ["Increase the brightness in this image.", "This image is very dark, make it brighter.",
                          "Make this image brighter."],
            "motion blur": ["Deblur this image.", "This image is very blurry, make it sharp.",
                            "Remove the blur from this image."],
            "motion": ["Deblur this image.", "This image is very blurry, make it sharp.",
                       "Remove the blur from this image."],
            "raindrop": ["Remove the drops from this image.", "This image has raindrops, remove them.",
                         "Remove the raindrops from this image."],
            "all": ["Remove all corruptions from this image.", "Enhance this image by removing all degradations.",
                    "Make this image clear."
                    "Remove all degradations from this image."]
        }

        print(colored(f'\nInitializing Val dataset for {degradation}', 'green'))
        if degradation == 'Motionblur':
            degradation = 'Motion blur'  # Account for space

        self.degradation = degradation
        self.args = args
        self.resize_size = args.resize_size
        self.all_datasets = []

        self.resize = True
        warnings.warn(f"Split is validation and images will be resized to {self.resize_size}")

        self.json_path = self.get_json()
        # Supported degradations
        self.valid_degradations = list(DEGRADATIONS.keys())

        assert degradation in self.valid_degradations, "Invalid degradation encountered"

        self._init_combined_ids()
        self.imagenet_norm = args.imagenet_norm

        if self.imagenet_norm:
            self.transform = self.transform = transforms.Compose([transforms.ToTensor(),
                                                                  transforms.Normalize(IMAGENET_DEFAULT_MEAN,
                                                                                       IMAGENET_DEFAULT_STD)])

            assert args.resize_size == args.patch_size
        else:
            self.transform = transforms.ToTensor()

    def get_json(self):
        json_path = os.path.join(self.args.dir, f"{self.degradation}_combined_val.json")

        return json_path

    def _init_combined_ids(self):
        self.ids = []
        loaded_json = json.load(open(self.json_path))

        # Count occurrences of each category and dataset
        count_category_and_dataset(loaded_json)
        print_degradation_subtypes(loaded_json)

        # For validation split, reduce size if necessary
        for line in loaded_json:
            if self.args.server:
                line['image_path'] = line['image_path'].replace('/data/', '/data/sambasa2/')
                line['target_path'] = line['target_path'].replace('/data/', '/data/sambasa2/')

            self.ids.append(line)

            curr_dataset = line['dataset']
            if curr_dataset not in self.all_datasets:
                self.all_datasets.append(curr_dataset)
        print(colored(f"Initialized {len(self.ids)} Val IDs", "blue"))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def __getitem__(self, idx):
        sample = self.ids[idx]
        degraded_name, clean_name = sample['image_path'], sample['target_path']

        assert sample["split"].lower() == 'val'

        # print(degraded_name, clean_name)
        ds = sample["dataset"]
        category = sample["category"]

        if random.random() < 0.2:
            text_key = "all"
        else:
            text_key = sample["degradation"].lower()

        if ds == 'CSD' and text_key != "all":
            text_key = "snow_and_haze"
        if ds == 'ORD' and text_key != "all":
            text_key = "rain_and_haze"

        # sentence_list = ALL_SENTENCES[text_key]
        # sentence = random.choice(sentence_list)
        sentence = get_random_embedding(text_key)

        if clean_name is None or not os.path.exists(clean_name):
            raise FileNotFoundError(
                f"Image path {clean_name} for {self.degradation, ds, category} is invalid or does not exist.")

        if degraded_name is None or not os.path.exists(degraded_name):
            raise FileNotFoundError(
                f"Image path {degraded_name} for {self.degradation, ds, category} is invalid or does not exist.")

        degrad_img = Image.open(degraded_name).convert('RGB')
        clean_img = Image.open(clean_name).convert('RGB')

        degrad_img = resize_img(degrad_img, self.resize_size)
        clean_img = resize_img(clean_img, self.resize_size)

        degrad_img = np.array(degrad_img)
        clean_img = np.array(clean_img)

        clean_img = self.transform(clean_img)
        degrad_img = self.transform(degrad_img)

        # self.debug(degrad_img, clean_img, idx, self.split)
        return ds, degrad_img, clean_img, sentence

    def __len__(self):
        return len(self.ids)


class BaseDatasetImage(Dataset):
    def __init__(self, args):
        super(BaseDatasetImage, self).__init__()

        self.args = args
        self.resize = args.size

        self.imagenet_norm = self.args.imagenet_norm

        self.image= self.args.image
        self.ids = [{"image_path": self.image, "target_path": self.args.image_clean}]
        print(colored(f'\nInitializing test dataset for {self.image} of length {len(self.ids)}', 'green'))
        # Supported degradations
        self.valid_degradations = list(DEGRADATIONS.keys())
        if self.imagenet_norm:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transform = transforms.ToTensor()

        if args.size is not None:
            self.size_h = int(args.size[0])
            self.size_w = int(args.size[1])

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        # ind_H = 200
        # ind_W = 200

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def __getitem__(self, idx):
        sample = self.ids[idx]
        degraded_name, clean_name = sample['image_path'], sample['target_path']

        ds = "single_img"
        category = "single_img_category"

        if degraded_name is None or not os.path.exists(degraded_name):
            raise FileNotFoundError(f"Image path {degraded_name} is invalid or does not exist.")

        degrad_img = Image.open(degraded_name).convert('RGB')
        clean_img = Image.open(clean_name).convert('RGB')

        if not self.resize:
            degrad_img = crop_img(np.array(degrad_img), base=16)
            clean_img = crop_img(np.array(clean_img), base=16)
        else:
            degrad_img = resize_img(degrad_img, (self.size_h, self.size_w))
            if self.args.model_name[0] == 'swin_vis':
                clean_img_dino = resize_img(clean_img, (224, 224))
            else:
                clean_img_dino = None
            clean_img = resize_img(clean_img, (self.size_h, self.size_w))

        # degrad_img, clean_img = self._crop_patch(degrad_img, clean_img)

        degrad_img = np.array(degrad_img)
        clean_img = np.array(clean_img)
        if self.args.model_name[0] == 'swin_vis':
            clean_img_dino = np.array(clean_img_dino)
            clean_img_dino = self.transform(clean_img_dino)

        clean_img = self.transform(clean_img)
        degrad_img = self.transform(degrad_img)


        # if self.resize:
        #     degrad_img = self.resizefn(degrad_img)
        #     clean_img = self.resizefn(clean_img)

        return [clean_name, degraded_name, ds, category], degrad_img, clean_img, clean_img_dino

    def __len__(self):
        return len(self.ids)