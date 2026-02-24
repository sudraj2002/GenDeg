import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img, resize_img, resize_img_crop
from utils.degradation_utils import Degradation
import json
import cv2

from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask


class SimMIMTransform:
    def __init__(self):
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])

        if config.MODEL.TYPE in ['swin', 'swinv2']:
            model_patch_size = config.MODEL.SWIN.PATCH_SIZE
        else:
            raise NotImplementedError

        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )

    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()

        return img, mask
class PromptTrainDataset(Dataset):
    def __init__(self, args):
        super(PromptTrainDataset, self).__init__()
        self.args = args
        # Supported degradations
        self.degradations = {'brightdown': [], 'jpeg': [], 'defocus': [], 'motion': [], 'brightup': [],
                             'text': [], 'gt': [], 'others': []}

        self.rs_ids = []
        self.hazy_ids = []
        self.de_type = self.args.de_type
        print(self.de_type)

        self.de_dict = {'brightdown': 1, 'jpeg': 2, 'defocus': 3, 'motion': 4, 'brightup': 5,
                        'text': 6, 'gt': 7, 'others': 8}

        self.num_gts = 500
        self.num_others = 0
        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

        self.mask_generator = MaskGenerator(
            input_size=192,
            mask_patch_size=32,
            model_patch_size=4,
            mask_ratio=0.6,
        )

    def _init_ids(self):
        self._init_all_ids()
        random.shuffle(self.de_type)

    def _init_all_ids(self):
        # Give json/text file for sampling input and gt
        file = os.path.join(self.args.wriva_dir, self.args.train_file)
        assert 'json' in self.args.train_file, "Only supports json and txt files"
        self._init_json(file)

    def _init_json(self, file):
        f = json.load(open(file))
        self.read(f)

    def read(self, f):
        for line in f:
            image_path = line['artifact_image_path']
            target_path = line['target_path']
            deg_type = line['artifacts']

            if deg_type:
                if ('brightdown' in deg_type or 'saturation' in deg_type or 'randombrightnesscontrast' in deg_type) and 'brightdown' in self.de_type:
                    self.degradations['brightdown'].append(
                        {'image_path': image_path, 'target_path': target_path, 'de_type': self.de_dict['brightdown']})

                if 'brightup' in deg_type and 'brightup' in self.de_type:
                    self.degradations['brightup'].append(
                        {'image_path': image_path, 'target_path': target_path, 'de_type': self.de_dict['brightup']})

                if ('jpeg' in deg_type or 'compression' in deg_type) and 'jpeg' in self.de_type:
                    self.degradations['jpeg'].append(
                        {'image_path': image_path, 'target_path': target_path, 'de_type': self.de_dict['jpeg']})

                if 'defocus' in deg_type and 'defocus' in self.de_type:
                    self.degradations['defocus'].append(
                        {'image_path': image_path, 'target_path': target_path, 'de_type': self.de_dict['defocus']})

                elif ('motion' in deg_type or 'blur' in deg_type) and 'motion' in self.de_type:
                    self.degradations['motion'].append(
                        {'image_path': image_path, 'target_path': target_path, 'de_type': self.de_dict['motion']})

                if 'text' in deg_type and 'text' in self.de_type:
                    self.degradations['text'].append(
                        {'image_path': image_path, 'target_path': target_path, 'de_type': self.de_dict['text']})

                if 'gt' in deg_type:
                    if len(self.degradations['gt']) < self.num_gts:
                        self.degradations['gt'].append(
                            {'image_path': image_path, 'target_path': target_path, 'de_type': self.de_dict['gt']})

            else:
                if len(self.degradations['others']) < self.num_others:
                    self.degradations['others'].append({'image_path': image_path, 'target_path': target_path,
                                                    'de_type': self.de_dict['others']})

        for key in self.degradations.keys():
            print(f"{key}: {len(self.degradations[key])}")

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_sample_name(self, d):
        im_name = d['image_path']
        gt_name = d['target_path']
        if self.args.server:
            im_name = im_name.replace('/data/wriva_all', '/data/sambasa2/wriva_all')
            gt_name = gt_name.replace('/data/wriva_all', '/data/sambasa2/wriva_all')
        return im_name, gt_name

    def _merge_ids(self):
        self.sample_ids = []

        for key in self.degradations.keys():
            self.sample_ids.extend(self.degradations[key])

        print(f"Total train samples: {len(self.sample_ids)}")

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]
        sample_name, clean_name = self._get_sample_name(sample)

        degrad_img = crop_img(np.array(Image.open(sample_name).convert('RGB')), base=16)
        clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
        degrad_img = resize_img(degrad_img, 192)
        clean_img = resize_img(clean_img, 192)

        degrad_patch, clean_patch = random_augmentation(degrad_img, clean_img)

        clean_patch = self.transform(clean_patch)
        degrad_patch = self.transform(degrad_patch)

        mask = self.mask_generator()

        # self.debug(degrad_patch, clean_patch, idx)

        return [clean_name, de_id], degrad_patch, clean_patch, mask

    def __len__(self):
        return len(self.sample_ids)

    def debug(self, img, gt, num, save_dir='results/'):
        degrad = np.asarray(img) * 255
        clean = np.asarray(gt) * 255
        degrad = degrad.transpose(1, 2, 0)
        clean = clean.transpose(1, 2, 0)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        degrad = degrad.astype(np.uint8)
        clean = clean.astype(np.uint8)

        degrad = cv2.cvtColor(degrad, cv2.COLOR_RGB2BGR)
        clean = cv2.cvtColor(clean, cv2.COLOR_RGB2BGR)

        cv2.imwrite(save_dir + str(num) + '_gt.png', clean)
        cv2.imwrite(save_dir + str(num) + '_degrad.png', degrad)


class PromptValDataset(Dataset):
    def __init__(self, args):
        super(PromptValDataset, self).__init__()
        self.args = args
        # Supported degradations
        self.degradations = {'brightdown': [], 'jpeg': [], 'defocus': [], 'motion': [], 'brightup': [],
                             'text': [], 'others': []}

        self.rs_ids = []
        self.hazy_ids = []
        self.de_type = self.args.val_de_type
        print(self.de_type)

        self.de_dict = {'brightdown': 1, 'jpeg': 2, 'defocus': 3, 'motion': 4, 'brightup': 5,
                        'text': 6, 'others': 7}

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

        self.mask_generator = MaskGenerator(
            input_size=192,
            mask_patch_size=32,
            model_patch_size=4,
            mask_ratio=0.6,
        )

    def _init_ids(self):
        self._init_all_ids()
        random.shuffle(self.de_type)

    def _init_all_ids(self):
        # Give json/text file for sampling input and gt
        file = os.path.join(self.args.wriva_dir, self.args.val_file)
        if 'json' in self.args.val_file:
            self._init_json(file)
        elif 'txt' in self.args.val_file:
            self._init_txt(file)
        else:
            raise ValueError("Only supports json and txt files")

    def _init_json(self, file):
        f = json.load(open(file))
        self.read(f)

    def _init_txt(self, file):
        f = open(file)
        self.read(f.readlines(), True)

    def read(self, f, strip=False):
        for line in f:
            image_path = line['artifact_image_path']
            target_path = line['target_path']
            deg_type = line['artifacts']

            if deg_type:
                if ('brightdown' in deg_type or 'saturation' in deg_type) and 'brightdown' in self.de_type:
                    self.degradations['brightdown'].append(
                        {'image_path': image_path, 'target_path': target_path, 'de_type': self.de_dict['brightdown']})

                if 'brightup' in deg_type and 'brightup' in self.de_type:
                    self.degradations['brightup'].append(
                        {'image_path': image_path, 'target_path': target_path, 'de_type': self.de_dict['brightup']})

                if ('jpeg' in deg_type or 'compression' in deg_type) and 'jpeg' in self.de_type:
                    self.degradations['jpeg'].append(
                        {'image_path': image_path, 'target_path': target_path, 'de_type': self.de_dict['jpeg']})

                if 'defocus' in deg_type and 'defocus' in self.de_type:
                    self.degradations['defocus'].append(
                        {'image_path': image_path, 'target_path': target_path, 'de_type': self.de_dict['defocus']})

                elif ('motion' in deg_type or 'blur' in deg_type) and 'motion' in self.de_type:
                    self.degradations['motion'].append(
                        {'image_path': image_path, 'target_path': target_path, 'de_type': self.de_dict['motion']})

                if 'text' in deg_type and 'text' in self.de_type:
                    self.degradations['text'].append(
                        {'image_path': image_path, 'target_path': target_path, 'de_type': self.de_dict['text']})

                if 'gt' in deg_type:
                    if len(self.degradations['gt']) < self.num_gts:
                        self.degradations['gt'].append(
                            {'image_path': image_path, 'target_path': target_path, 'de_type': self.de_dict['gt']})

            else:
                self.degradations['others'].append({'image_path': image_path, 'target_path': target_path,
                                                    'de_type': self.de_dict['others']})

        for key in self.degradations.keys():
            print(f"{key}: {len(self.degradations[key])}")

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_sample_name(self, d):
        im_name = d['image_path']
        gt_name = d['target_path']
        if self.args.server:
            im_name = im_name.replace('/data/wriva_all', '/data/sambasa2/wriva_all')
            gt_name = gt_name.replace('/data/wriva_all', '/data/sambasa2/wriva_all')
        return im_name, gt_name

    def _merge_ids(self):
        self.sample_ids = []

        for key in self.degradations.keys():
            self.sample_ids.extend(self.degradations[key])

        print(f"Total val samples: {len(self.sample_ids)}")

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]
        sample_name, clean_name = self._get_sample_name(sample)

        degrad_img = crop_img(np.array(Image.open(sample_name).convert('RGB')), base=16)
        clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

        degrad_img = resize_img(degrad_img, 192)
        clean_img = resize_img(clean_img, 192)

        clean_patch = self.transform(clean_img)
        degrad_patch = self.transform(degrad_img)

        # self.debug(degrad_patch, clean_patch, idx)
        mask = self.mask_generator()

        return [clean_name, de_id], degrad_patch, clean_patch, mask

    def __len__(self):
        return len(self.sample_ids)

    def debug(self, img, gt, num, save_dir='results/'):
        degrad = np.asarray(img) * 255
        clean = np.asarray(gt) * 255
        degrad = degrad.transpose(1, 2, 0)
        clean = clean.transpose(1, 2, 0)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        degrad = degrad.astype(np.uint8)
        clean = clean.astype(np.uint8)

        cv2.imwrite(save_dir + str(num) + '_gt.png', clean)
        cv2.imwrite(save_dir + str(num) + '_degrad.png', degrad)
