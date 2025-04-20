import numpy as np
import json
import os

import random
import bisect


def min_max_key(histogram):
    # Get min and max key
    random_bin = histogram[0]
    min_key = None
    max_key = None
    for key in random_bin:
        if 'min' in key:
            min_key = key

        if 'max' in key:
            max_key = key

    return min_key, max_key

def get_lowest_and_highest_bins(histogram):
    # Filter bins with 'Count' > 0
    non_zero_bins = [bin_dict for bin_dict in histogram if bin_dict.get('Count', 0) > 0]

    if not non_zero_bins:
        # No bins with non-zero counts
        return None, None

    min_key, max_key = min_max_key(histogram)
    # Find the lowest 'min' value
    lowest_min = min(bin_dict[min_key] for bin_dict in non_zero_bins)

    # Find the highest 'max' value
    highest_max = max(bin_dict[max_key] for bin_dict in non_zero_bins)

    return lowest_min, highest_max

def preprocess_histogram(histogram, mode='normal'):
    cumulative_counts = []
    total = 0
    for bin_dict in histogram:
        total += bin_dict['Count']
        cumulative_counts.append(total)
    return cumulative_counts, total


def sample_from_histogram(histogram, cumulative_counts, total):
    # Step 1: Randomly select a count
    rand_count = random.randint(1, total)

    # Step 2: Find the bin where rand_count falls into
    bin_index = bisect.bisect_left(cumulative_counts, rand_count)
    selected_bin = histogram[bin_index]

    min_key, max_key = min_max_key(histogram)
    # Step 3: Sample a value within the selected bin
    min_val = selected_bin[min_key]
    max_val = selected_bin[max_key]

    sampled_value = random.uniform(min_val, max_val)

    return sampled_value, f"{min_val:.3f} - {max_val:.3f}"

def sample_dataset_stat(dataset, dataset_dict_mu, dataset_dict_sigma, dataset_dict_mu_sigma, 
                        random_sample=False):
    if random_sample:
        # Dont sample based on histogram
        histogram_mu = dataset_dict_mu[dataset]
        histogram_sigma = dataset_dict_sigma[dataset]
        min_bin_mu, max_bin_mu = get_lowest_and_highest_bins(histogram_mu)
        min_bin_sigma, max_bin_sigma = get_lowest_and_highest_bins(histogram_sigma)

        return random.uniform(min_bin_mu, max_bin_mu), random.uniform(min_bin_sigma, max_bin_sigma)

    # Sample a value based on histogram of mu or sigma
    histogram_mu = dataset_dict_mu[dataset]
    cumulative_counts, total = preprocess_histogram(histogram_mu)
    
    sampled_mu, key = sample_from_histogram(histogram_mu, cumulative_counts, total)

    dataset_mu_sigma = dataset_dict_mu_sigma[dataset]
    histogram_sigma = get_count_by_mu_range_dict(dataset_mu_sigma, key)
    cumulative_counts, total = preprocess_histogram(histogram_sigma)

    sampled_sigma, _ = sample_from_histogram(histogram_sigma, cumulative_counts, total)


    return sampled_mu, sampled_sigma

# Function to convert list to dictionary
def convert_list_to_dict(data):
    return {entry["Mu range"]: entry["Count"] for entry in data}

# Function to get 'count' using the lookup dictionary
def get_count_by_mu_range_dict(lookup_dict, target_mu_range):
    return lookup_dict.get(target_mu_range, None)

def load_data(root, degradation, filter='Mu'):
    json_dir = os.path.join(root, degradation)
    dataset_dict = {}
    for json_file in os.listdir(json_dir):
        if filter in json_file:
            file_path = os.path.join(json_dir, json_file)
            content = json.load(open(file_path))
            dataset = content[0]['Dataset']
            if dataset not in dataset_dict:
                if 'Mu-Sigma' in filter:
                    # Create the lookup dictionary
                    lookup_dict = convert_list_to_dict(content)
                    dataset_dict[dataset] = lookup_dict
                else:
                    dataset_dict[dataset] = content
    return dataset_dict
