import torch
import matplotlib.pyplot as plt
import os


def save_each_feature(features, save_dir='features'):
    """
    Saves each feature of shape (1, C, H, W) as a separate image file.

    Parameters:
    features (torch.Tensor): Tensor of shape (1, C, H, W).
    save_dir (str): Directory to save the visualized images.
    """
    # Remove the batch dimension (1, C, H, W) -> (C, H, W)
    features = features.squeeze(0)

    # Get the number of channels
    C = features.shape[0]

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save each feature map separately
    for i in range(C):
        plt.figure()
        plt.imshow(features[i].cpu().detach().numpy(), cmap='viridis')
        plt.axis('off')
        plt.title(f'Channel {i}')

        # Save each channel as a separate image
        save_path = os.path.join(save_dir, f'feature_{i}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
