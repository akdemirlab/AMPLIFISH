"""
Shannon entropy measurement for FISH target-spot spatial distribution.

Provides spatial binning, per-nucleus entropy computation, and heatmap
visualization for target-spot distribution analysis.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
from skimage.measure import regionprops

from amplifish.utils.image import split_color_channels

# Default bin diameter (pixels) for spatial entropy calculation
BIN_DIAMETER = 50


def calculate_target_bin_matrix(nucleus_mask: np.ndarray,
                                 target_mask: np.ndarray,
                                 bin_diameter: int = BIN_DIAMETER):
    """
    Count target pixels per spatial bin within a nucleus bounding box.

    Parameters
    ----------
    nucleus_mask : np.ndarray
        Boolean mask for a single nucleus.
    target_mask : np.ndarray
        Boolean mask of all target (FISH spot) pixels, same shape.
    bin_diameter : int
        Side length (pixels) of each square bin.

    Returns
    -------
    tuple : (bin_matrix, num_bins_row, num_bins_col, min_row, min_col)
        bin_matrix   – 2D int array of target pixel counts per bin.
        num_bins_row – number of bin rows.
        num_bins_col – number of bin columns.
        min_row, min_col – top-left corner of the nucleus bounding box.
    """
    target_in_nucleus = nucleus_mask & target_mask
    rows, cols = np.where(nucleus_mask)

    if rows.size == 0:
        return np.array([0]), 0, 0, 0, 0

    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()

    row_extent = max_row - min_row + 1
    col_extent = max_col - min_col + 1

    num_bins_row = int(np.ceil(row_extent / bin_diameter))
    num_bins_col = int(np.ceil(col_extent / bin_diameter))

    bin_matrix = np.zeros((num_bins_row, num_bins_col), dtype=int)
    target_rows, target_cols = np.where(target_in_nucleus)

    for r, c in zip(target_rows, target_cols):
        bin_r = (r - min_row) // bin_diameter
        bin_c = (c - min_col) // bin_diameter
        bin_matrix[bin_r, bin_c] += 1

    return bin_matrix, num_bins_row, num_bins_col, min_row, min_col


def calculate_global_bin_matrix(target_mask: np.ndarray,
                                 image_shape: tuple,
                                 bin_diameter: int = BIN_DIAMETER) -> np.ndarray:
    """
    Count target pixels per spatial bin across the full image.

    Parameters
    ----------
    target_mask : np.ndarray
        Boolean mask of all target (FISH spot) pixels.
    image_shape : tuple
        (rows, cols) of the image.
    bin_diameter : int
        Side length (pixels) of each square bin.

    Returns
    -------
    np.ndarray
        2D int array of target pixel counts per bin.
    """
    rows, cols = image_shape
    num_bins_row = int(np.ceil(rows / bin_diameter))
    num_bins_col = int(np.ceil(cols / bin_diameter))

    bin_matrix = np.zeros((num_bins_row, num_bins_col), dtype=int)
    target_rows, target_cols = np.where(target_mask)

    for r, c in zip(target_rows, target_cols):
        bin_matrix[r // bin_diameter, c // bin_diameter] += 1

    return bin_matrix


def save_bin_matrix_heatmap(bin_matrix: np.ndarray, nucleus_label: int,
                             image_name: str, out_dir: str) -> None:
    """
    Save a per-nucleus bin matrix as a heatmap image (debug use).

    Parameters
    ----------
    bin_matrix : np.ndarray
        2D array of target pixel counts per bin.
    nucleus_label : int
        Nucleus identifier (used in the filename).
    image_name : str
        Base image name (used in the filename).
    out_dir : str
        Directory where the heatmap PNG is saved.
    """
    if bin_matrix.size == 0 or not np.any(bin_matrix > 0):
        return

    import os
    fig, ax = plt.subplots()
    im = ax.imshow(bin_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Target Pixel Count per Bin')
    ax.set_title(f"Bin Matrix (Nucleus {nucleus_label})")
    ax.set_xlabel("Bin Column Index")
    ax.set_ylabel("Bin Row Index")
    ax.tick_params(axis='both', which='major', labelsize=6)

    output_path = os.path.join(out_dir, f"{image_name}_nucleus_{nucleus_label}_bin_matrix_DEBUG.png")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def create_global_bin_heatmap(bin_matrix: np.ndarray, out_dir: str,
                               image_name: str) -> None:
    """
    Save the global target-density bin matrix as a heatmap image.

    Parameters
    ----------
    bin_matrix : np.ndarray
        2D array of target pixel counts per bin (full image).
    out_dir : str
        Output directory.
    image_name : str
        Base image name used in the filename.
    """
    if bin_matrix.size == 0 or not np.any(bin_matrix > 0):
        return

    import os
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(bin_matrix, cmap='magma', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Total Target Pixel Count per Bin')
    ax.set_title(f"Global Target Bin Density Map for {image_name}")
    ax.set_xlabel(f"Bin Column Index (Bin Size: {BIN_DIAMETER}×{BIN_DIAMETER} pixels)")
    ax.set_ylabel("Bin Row Index")

    output_path = os.path.join(out_dir, f"{image_name}_global_target_bin_heatmap.png")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    logger.info("Saved global target bin heatmap.")


def create_global_entropy_heatmap(image: np.ndarray, nuclei_labels: np.ndarray,
                                   entropy_data_list: list, out_dir: str,
                                   image_name: str) -> None:
    """
    Save a per-nucleus Shannon entropy overlay on the FISH image.

    Each nucleus region is coloured by its computed target-spot entropy value,
    overlaid (alpha=0.6) on the red channel of the FISH image.

    Parameters
    ----------
    image : np.ndarray
        Full RGB FISH image (H, W, 3).
    nuclei_labels : np.ndarray
        Nucleus label image (H, W), same spatial extent as `image`.
    entropy_data_list : list of dict
        Each dict must have keys 'Nucleus_Label' and 'target_entropy'.
    out_dir : str
        Output directory.
    image_name : str
        Base image name used in the filename.
    """
    from csbdeep.utils import normalize
    import os

    entropy_map_canvas = np.full(nuclei_labels.shape, np.nan, dtype=float)
    entropy_lookup = {d['Nucleus_Label']: d['target_entropy'] for d in entropy_data_list}

    for region in regionprops(nuclei_labels):
        lbl = region.label
        entropy_value = entropy_lookup.get(lbl, 0.0)

        if entropy_value > 0:
            nucleus_mask = (nuclei_labels == lbl)
            current_values = entropy_map_canvas[nucleus_mask]
            if np.any(~np.isnan(current_values)):
                update_mask = nucleus_mask & (np.nan_to_num(entropy_map_canvas) < entropy_value)
                entropy_map_canvas[update_mask] = entropy_value
            else:
                entropy_map_canvas[nucleus_mask] = entropy_value

    R_channel, _, _ = split_color_channels(image)
    R_channel_normalized = normalize(R_channel, 1, 99.8, axis=(0, 1))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(R_channel_normalized, cmap='gray')

    max_entropy = np.nanmax(entropy_map_canvas) if not np.all(np.isnan(entropy_map_canvas)) else 1.0
    im = ax.imshow(entropy_map_canvas, cmap='plasma', alpha=0.6, vmin=0.0, vmax=max_entropy)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Target Spot Distribution Entropy (bits)', rotation=270, labelpad=15)
    ax.set_title(f"Target Spot Entropy Heatmap Overlay for {image_name}")
    ax.axis('off')

    output_path = os.path.join(out_dir, f"{image_name}_global_entropy_heatmap.png")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    logger.info("Saved global entropy heatmap overlay.")


def compute_entropy_per_nucleus(nuclei_labels: np.ndarray,
                                 target_mask: np.ndarray,
                                 bin_diameter: int = BIN_DIAMETER) -> list:
    """
    Compute Shannon entropy of target-spot spatial distribution per nucleus.

    Parameters
    ----------
    nuclei_labels : np.ndarray
        Nucleus label image.
    target_mask : np.ndarray
        Boolean mask of target (FISH spot) pixels.
    bin_diameter : int
        Bin side length for spatial binning.

    Returns
    -------
    list of dict
        One dict per nucleus with keys:
        'Nucleus_Label', 'target_entropy', 'bin_matrix_rows', 'bin_matrix_cols'.
    """
    entropy_data = []
    for region in regionprops(nuclei_labels):
        nucleus_label = region.label
        nucleus_mask = (nuclei_labels == nucleus_label)

        bin_matrix, num_bins_row, num_bins_col, _, _ = calculate_target_bin_matrix(
            nucleus_mask, target_mask, bin_diameter
        )
        local_entropy = 0.0

        if np.sum(bin_matrix) > 0:
            total = np.sum(bin_matrix)
            counts = bin_matrix.flatten()
            positive_counts = counts[counts > 0]
            probabilities = positive_counts / total
            local_entropy = -np.sum(probabilities * np.log2(probabilities))

        entropy_data.append({
            'Nucleus_Label': nucleus_label,
            'target_entropy': local_entropy,
            'bin_matrix_rows': num_bins_row,
            'bin_matrix_cols': num_bins_col,
        })

    return entropy_data
