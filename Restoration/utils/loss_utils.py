import torch
import torch.nn as nn
from torch.nn.functional import mse_loss


# Implements consistency loss
class ConLoss(nn.Module):
    def __init__(self, consistency_wt, contrastive_wt, tau=0.07):
        super(ConLoss, self).__init__()

        self.consistency_wt = consistency_wt
        self.contrastive_wt = contrastive_wt
        self.contrastive_loss = ContrastiveLoss(tau)
        self.consistency_loss = nn.L1Loss()

    def filter_tensors_by_mask(self, dict1, dict2, mask_tensor):
        """
        Filters out the tensors from dict1 and dict2 where the mask_tensor has a value of 0.

        Parameters:
        dict1 (dict): Dictionary containing tensors of shape (B, C, H, W).
        dict2 (dict): Another dictionary with the same structure and shape as dict1.
        mask_tensor (torch.Tensor): Tensor of shape (B, 1) with values 0 or 1.

        Returns:
        dict: Two dictionaries containing filtered tensors where mask_tensor has 0.
        """
        # Get the indices where the mask_tensor is 0 (considering B-dimension only)
        zero_indices = (mask_tensor == 0).nonzero(as_tuple=True)[0]

        # Filter the tensors from both dictionaries at the found indices
        filtered_dict1 = {key: value[zero_indices] for key, value in dict1.items()}
        filtered_dict2 = {key: value[zero_indices] for key, value in dict2.items()}

        return filtered_dict1, filtered_dict2, zero_indices

    def compute_l1_loss(self, filtered_dict_deg, filtered_dict_clean):
        """
        Computes L1 loss to ensure:
        1. Consistency between each 1, C, H, W element in filtered_dict_deg.
        2. Each element in filtered_dict_deg matches the corresponding element in filtered_dict_clean.

        Parameters:
        filtered_dict_deg (dict): Dictionary of degenerated tensors of shape (B, C, H, W).
        filtered_dict_clean (dict): Dictionary of clean tensors of the same shape (B, C, H, W).

        Returns:
        torch.Tensor: Total L1 loss (internal + external consistency).
        """
        total_loss = 0.0

        for key in filtered_dict_deg:
            deg_tensor = filtered_dict_deg[key]
            clean_tensor = filtered_dict_clean[key]

            B = deg_tensor.shape[0]
            # Ensure internal consistency for degenerated tensors: L1 loss between all B elements
            if B > 1:
                internal_loss = self.consistency_loss(deg_tensor[:-1], deg_tensor[1:])
                total_loss += internal_loss

            # Ensure external consistency with corresponding clean tensor
            external_loss = self.consistency_loss(deg_tensor, clean_tensor)
            total_loss += external_loss

            print(key, external_loss)
        total_loss /= len(filtered_dict_deg)

        return total_loss

    def forward(self, degraded_dict, clean_dict, scene_ids):
        consistency_loss = self.consistency_wt * self.compute_l1_loss(degraded_dict, clean_dict)
        # contrastive_loss = self.contrastive_wt * self.contrastive_loss(degraded_dict, scene_ids)

        return consistency_loss



class ContrastiveLoss(nn.Module):
    def __init__(self, tau):
        super(ContrastiveLoss, self).__init__()
        self.tau = tau

    def compute_similarity_loss(self, tensor_B_L, binary_tensor_B):
        """
        Computes a loss such that:
        - Elements in tensor_B_L should be close if corresponding values in binary_tensor_B are both 0.
        - Elements should be far apart if at least one of the values in binary_tensor_B is 1.

        Parameters:
        tensor_B_L (torch.Tensor): Tensor of shape (B, L), where B is batch size and L is feature length.
        binary_tensor_B (torch.Tensor): Binary tensor of shape (B,) containing 0s and 1s.

        Returns:
        torch.Tensor: The computed loss.
        """
        # Normalize the B, L tensor for cosine similarity computation (optional but useful)
        tensor_B_L_normalized = torch.nn.functional.normalize(tensor_B_L, p=2, dim=1)

        # print("\n New batch")
        # print("Normalized tensor", tensor_B_L_normalized)

        # Compute similarity matrix (B, B), pairwise similarity using matrix multiplication
        similarity_matrix = torch.matmul(tensor_B_L_normalized, tensor_B_L_normalized.T)
        # print("Similarity matrix", similarity_matrix)
        similarity_matrix_exp = torch.exp(similarity_matrix / self.tau)
        # print("Similarity matrix exp", similarity_matrix_exp)

        # Create masks based on binary_tensor_B
        # Create a mask where both values are 0 (encourage closeness)
        closeness_mask = (binary_tensor_B == 0).unsqueeze(1) * (binary_tensor_B == 0).unsqueeze(0)

        # Ignore diagonal elements (self-similarities)
        diagonal_mask = torch.eye(closeness_mask.size(0), device=closeness_mask.device, dtype=torch.bool)
        closeness_mask = closeness_mask & ~diagonal_mask  # Remove diagonals from the mask

        # Create a mask where at least one value is 1 (encourage distance)
        distance_mask = (binary_tensor_B.unsqueeze(1) + binary_tensor_B.unsqueeze(0)) > 0

        # print("Close mask", closeness_mask)
        # print("Far mask", distance_mask)

        # Calculate the closeness
        closeness = similarity_matrix_exp[closeness_mask].mean()

        # Calculate the loss for distance or farness
        distance = similarity_matrix_exp[distance_mask].mean()
        if torch.isnan(distance):
            distance = 1.0

        total_loss = -torch.log(closeness / (distance + 1e-3))

        return total_loss

    def forward(self, degraded_dict, scene_ids):
        total_loss = 0.0
        for key in degraded_dict:
            deg_tensor = degraded_dict[key]

            B = deg_tensor.shape[0]
            deg_tensor = deg_tensor.mean(dim=(2, 3)).view(B, -1)  # along channels or spatial??

            loss_key = self.compute_similarity_loss(deg_tensor, scene_ids)

            total_loss += loss_key

        total_loss /= len(degraded_dict)

        return total_loss

