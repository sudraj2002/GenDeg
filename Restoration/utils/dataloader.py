import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, SequentialSampler
from torchvision import datasets, transforms
import pytorch_lightning as pl
from utils.baseloader import BaseDataset, BaseDatasetImage
from torch.utils.data.distributed import DistributedSampler

def calculate_weights(dataset_lengths):
    # Inverse of the dataset size as weights
    total_length = sum(dataset_lengths)
    weights = [total_length / length for length in dataset_lengths]
    return weights


def oversample_and_concat_datasets(datasets, oversample=True):
    """
    Oversamples smaller datasets to match the size of the largest dataset in the array,
    and then concatenates all datasets into one final trainset.

    Args:
        datasets (list of Dataset): The array of datasets to be processed.

    Returns:
        ConcatDataset: A concatenated dataset where all component datasets have been
                       oversampled to have the same size as the largest dataset.
    """

    # Function to oversample a dataset
    def oversample_dataset(dataset, target_size):

        oversample_factor = target_size // len(dataset) + int(target_size % len(dataset) > 0)
        oversampled_data_indices = (list(range(len(dataset))) * oversample_factor)[:target_size]
        return torch.utils.data.Subset(dataset, oversampled_data_indices)

    # Oversample smaller datasets
    if oversample:
        # Find the size of the largest dataset
        max_size = max(len(ds) for ds in datasets)
        oversampled_datasets = [oversample_dataset(ds, max_size) for ds in datasets]

        # Concatenate all oversampled datasets into one
        final_trainset = torch.utils.data.ConcatDataset(oversampled_datasets)
    else:
        final_trainset = torch.utils.data.ConcatDataset(datasets)

    return final_trainset


def return_loaders(args, split, concatset):
    if split == 'train':
        shuffle = True
        batch_size = args.batch_size
        drop_last = True
    else:
        shuffle = False
        batch_size = 1
        drop_last = False

    print(f"Combined set length for {split}: {len(concatset)}")

    return DataLoader(concatset, batch_size=batch_size, pin_memory=True, num_workers=args.num_workers,
                      drop_last=drop_last, shuffle=shuffle)


def create_datasets(args):
    datasets = []
    dataset = BaseDataset(args=args)

    datasets.append(dataset)

    testmodule = oversample_and_concat_datasets(datasets, oversample=False)

    testloader = return_loaders(args=args, split='test', concatset=testmodule)

    return testloader


def create_datasets_singleimg(args):
    datasets = []
    dataset = BaseDatasetImage(args=args)

    datasets.append(dataset)

    testmodule = oversample_and_concat_datasets(datasets, oversample=False)

    testloader = return_loaders(args=args, split='test', concatset=testmodule)

    return testloader


