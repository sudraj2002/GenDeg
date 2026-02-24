import torch
import numpy as np
from sklearn.decomposition import PCA


def pca_reduce_channels(features, n_components=3):
    """
    Applies PCA to reduce the channel dimension of a feature map from (1, C, H, W) to (1, 3, H, W).

    Parameters:
    features (torch.Tensor): Tensor of shape (1, C, H, W).
    n_components (int): Number of principal components to keep (default is 3).

    Returns:
    torch.Tensor: Tensor of shape (1, 3, H, W) after PCA.
    """
    # Remove the batch dimension (1, C, H, W) -> (C, H, W)
    features = features.squeeze(0)

    # Get the dimensions
    C, H, W = features.shape

    # Flatten the spatial dimensions (H, W) to apply PCA across channels
    flattened_features = features.view(C, -1).T  # Shape: (H*W, C)

    # Move the tensor to CPU and convert it to a numpy array for PCA
    flattened_features_np = flattened_features.cpu().detach().numpy()

    # Apply PCA to reduce channels from C to n_components
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(flattened_features_np)  # Shape: (H*W, n_components)

    # Reshape back to (n_components, H, W)
    reduced_features = reduced_features.T.reshape(n_components, H, W)

    # Add the batch dimension back to get the shape (1, n_components, H, W)
    reduced_features = torch.tensor(reduced_features).unsqueeze(0)

    return reduced_features