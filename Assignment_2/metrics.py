import numpy as np
from skimage.morphology import binary_erosion
from medpy.metric.binary import hd95

def extract_surface_voxels(mask):
    """
    Extract surface voxels from a 3D segmentation mask by subtracting an eroded version of itself.

    Parameters:
        mask (ndarray): 3D segmentation mask (integer values for each class).

    Returns:
        dict: A dictionary where keys are class labels, and values are (x, y, z) coordinates of surface voxels.
    """
    unique_classes = np.unique(mask)  # Get unique class labels
    class_surfaces = {}

    for cls in unique_classes:  # Include background (0)
        binary_mask = (mask == cls)  # Convert to binary mask for the current class
        eroded = binary_erosion(binary_mask)  # Erode to get interior
        surface = binary_mask ^ eroded  # XOR to keep only boundary voxels
        class_surfaces[cls] = surface  # Store binary surface mask

    return class_surfaces

def hausdorff_95_multiclass(target, predict, region_ids):
    """
    Compute the 95th percentile Hausdorff distance for a specific region of interest using MedPy's hd95.
    
    Parameters:
    - target (ndarray): Ground truth segmentation mask (3D).
    - predict (ndarray): Predicted segmentation mask (3D).
    - region_ids (list): List of class IDs corresponding to the region of interest (e.g., [1, 4] for Tumor Core).
    
    Returns:
    - float: The 95th percentile Hausdorff distance for the specified region.
    """
    # Create binary masks for the union of the specified region IDs in both the target and predicted masks

    binary_o = np.isin(target, region_ids).astype(np.uint8)
    binary_t = np.isin(predict, region_ids).astype(np.uint8)

    # Extract the surface voxels for each mask
    surfaces_o = extract_surface_voxels(binary_o)
    surfaces_t = extract_surface_voxels(binary_t)

    # Handle case where no surface is found for a class
    #if np.sum(surface_o_merged) == 0 or np.sum(surface_t_merged) == 0:
    #    return float('inf')  # Return inf if there are no valid surfaces to compare

    # Compute the 95th percentile Hausdorff distance using MedPy's hd95
    hd95_value = hd95(surfaces_o[1], surfaces_t[1])
    
    return hd95_value

def dice_score_multiclass(o, t, class_ids, eps=1e-8):
    """
    Compute the Dice Similarity Coefficient (DSC) for specific classes (union of classes) in two 3D multi-class segmentation masks.
    
    Args:
    - o (np.ndarray): The ground truth 3D multi-class mask.
    - t (np.ndarray): The predicted 3D multi-class mask.
    - class_ids (list or set): List or set of class IDs to consider for the Dice score.
    - eps (float): Small value added to the denominator to avoid division by zero.
    
    Returns:
    - float: The Dice score for the union of the specified class IDs.
    """
    # Create a binary mask for the union of the specified class IDs in the ground truth and predicted masks
    binary_o = np.isin(o, class_ids).astype(np.uint8)
    binary_t = np.isin(t, class_ids).astype(np.uint8)
    
    # Compute the intersection and sum of masks for the union of these classes
    intersection = np.sum(binary_o * binary_t)
    sum_masks = np.sum(binary_o) + np.sum(binary_t)
    
    # Compute the Dice score for the union of the specified classes
    dice = (2.0 * intersection) / (sum_masks + eps)  # Add eps to avoid division by zero
    
    return dice
