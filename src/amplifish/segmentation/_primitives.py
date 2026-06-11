"""
Low-level segmentation primitives for FISH images.

CellProfiler-inspired building blocks — speckle enhancement and
threshold-based object identification — used internally by the
segmentation backends.
"""

import logging

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
from skimage import morphology, exposure
from skimage.exposure import histogram as hist
from skimage.filters import threshold_multiotsu, threshold_triangle, threshold_li, threshold_yen, gaussian
from skimage.segmentation import clear_border, watershed
from skimage.morphology import local_maxima, remove_small_holes
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt as distance_transform

from amplifish.utils.viz import _generate_random_cmap


def _clean_labeled_image(labeled_image, area_threshold=64):
    """Fill small holes within each distinct label in a labeled image."""
    cleaned_image = np.zeros_like(labeled_image)
    unique_labels = np.unique(labeled_image[labeled_image != 0])
    for lbl in unique_labels:
        binary_mask = (labeled_image == lbl)
        cleaned_mask = remove_small_holes(binary_mask, area_threshold=area_threshold)
        cleaned_image[cleaned_mask] = lbl
    return cleaned_image


def enhance_speckles(img: np.ndarray, disk_radius: int = 10,
                     plot_for_testing: bool = False,
                     plot_gamma: float = 1.0,
                     plot_cmap: str = 'gray') -> np.ndarray:
    """
    Enhance bright speckles via a white top-hat filter.

    White Top-Hat = Original - Morphological Opening. Highlights small bright
    features (speckles) against a suppressed background.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image. Integer dtypes (uint8/uint16) are rescaled to [0, 1].
    disk_radius : int
        Radius of the disk structuring element. Choose slightly larger than the
        expected speckle radius.
    plot_for_testing : bool
        Display a before/after diagnostic plot.
    plot_gamma : float
        Gamma correction applied only to the display plots.
    plot_cmap : str
        Matplotlib colormap for display plots.

    Returns
    -------
    np.ndarray
        Enhanced image with speckles highlighted.
    """
    if img.dtype in (np.uint8, np.uint16):
        img = img / np.iinfo(img.dtype).max

    footprint = morphology.disk(disk_radius)
    opened_img = morphology.opening(img, footprint=footprint)
    enhanced_speckles = img - opened_img

    if plot_for_testing:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(exposure.adjust_gamma(img, gamma=plot_gamma), cmap=plot_cmap)
        axes[0].set_title('Original')
        axes[0].axis('off')
        axes[1].imshow(exposure.adjust_gamma(opened_img, gamma=plot_gamma), cmap=plot_cmap)
        axes[1].set_title('Opening')
        axes[1].axis('off')
        axes[2].imshow(exposure.adjust_gamma(enhanced_speckles, gamma=plot_gamma), cmap=plot_cmap)
        axes[2].set_title(f'Enhanced (R={disk_radius})')
        axes[2].axis('off')

    return enhanced_speckles


def identify_primary_objects(
    grayscale_image: np.ndarray,
    min_diameter: int = 100,
    max_diameter: int = 1000,
    hole_diameter: int = 10,
    threshold_method: str = 'otsu',
    threshold_kwargs: dict = None,
    threshold_smooth_scale: float = 0.0,
    threshold_correction_factor: float = 1.0,
    use_log_preprocessing: bool = True,
    clumping_method: str = 'watershed',
    sigma_for_declumping: float = 2.0,
    clear_border_objects: bool = True,
    marker_source: str = 'shape',
    dividing_source: str = 'intensity',
    plot_for_testing: bool = False,
) -> np.ndarray:
    """
    Segment objects in a grayscale image via thresholding and optional watershed.

    Simulates CellProfiler's IdentifyPrimaryObjects module. Size filtering uses
    object area (diameter² pixels).

    Parameters
    ----------
    grayscale_image : np.ndarray
        2D float array in [0.0, 1.0].
    min_diameter : int
        Minimum object diameter; objects with area < min_diameter² are discarded.
    max_diameter : int
        Maximum object diameter; objects with area > max_diameter² are discarded.
    hole_diameter : int
        Holes smaller than hole_diameter² pixels are filled before labeling.
    threshold_method : str
        Thresholding strategy: ``'otsu'``, ``'triangle'``, ``'li'``, or ``'yen'``.
    threshold_kwargs : dict, optional
        Method-specific keyword arguments forwarded to the threshold function.
        For ``'otsu'``: ``{'classes': 3}`` (default 3).
        For ``'li'``: ``{'tolerance': ...}`` or ``{'initial_guess': ...}``.
        Ignored by ``'triangle'`` and ``'yen'``.
    threshold_smooth_scale : float
        Gaussian sigma applied before threshold calculation (0 = disabled).
    threshold_correction_factor : float
        Multiplier applied to the computed threshold.
    use_log_preprocessing : bool
        Apply log1p transform before thresholding to enhance dim objects.
    clumping_method : str
        'watershed' for watershed-based declumping, 'none' for direct labeling.
    sigma_for_declumping : float
        Gaussian sigma for smoothing during watershed seed detection.
    clear_border_objects : bool
        Remove objects touching the image boundary.
    marker_source : str
        Seed source for watershed: 'shape' (distance transform) or 'intensity'.
    dividing_source : str
        Basin source for watershed: 'shape' or 'intensity'.
    plot_for_testing : bool
        Display diagnostic plots (intended for interactive / notebook use).

    Returns
    -------
    np.ndarray
        2D int32 label array; 0 = background, positive integers = object IDs.

    Raises
    ------
    ValueError
        For unsupported threshold or clumping methods.
    """
    if np.all(grayscale_image == 0) or grayscale_image.min() == grayscale_image.max():
        logger.warning("Input grayscale image is empty or uniform. Returning zero label mask.")
        return np.zeros_like(grayscale_image, dtype=np.int32)

    if use_log_preprocessing:
        grayscale_image = np.log1p(grayscale_image)
        if grayscale_image.max() > 0:
            grayscale_image /= grayscale_image.max()

    if threshold_smooth_scale > 0:
        grayscale_image = gaussian(grayscale_image, sigma=threshold_smooth_scale, preserve_range=True)

    kw = threshold_kwargs or {}
    _THRESHOLD_FNS = {
        'otsu':     lambda img: threshold_multiotsu(img, classes=kw.get('classes', 3))[-1],
        'triangle': lambda img: threshold_triangle(img),
        'li':       lambda img: threshold_li(img, **kw),
        'yen':      lambda img: threshold_yen(img),
    }
    method = threshold_method.lower()
    if method not in _THRESHOLD_FNS:
        raise ValueError(f"Unsupported threshold method: '{threshold_method}'. "
                         f"Choose one of: {list(_THRESHOLD_FNS)}")
    thresholds = [_THRESHOLD_FNS[method](grayscale_image)]
    threshold_to_apply = thresholds[-1] * threshold_correction_factor
    binary_mask = grayscale_image > threshold_to_apply

    if hole_diameter > 0:
        binary_mask = morphology.remove_small_holes(binary_mask, hole_diameter ** 2)

    if clumping_method.lower() == 'watershed':
        distance = distance_transform(binary_mask)
        smoothed_distance = gaussian(distance, sigma=sigma_for_declumping, preserve_range=True)

        if marker_source.lower() == 'intensity':
            smoothed_marker_source = gaussian(grayscale_image, sigma=sigma_for_declumping, preserve_range=True)
            local_maxi_mask = local_maxima(smoothed_marker_source)
        elif marker_source.lower() == 'shape':
            local_maxi_mask = local_maxima(smoothed_distance)
        else:
            raise ValueError(f"Unsupported marker source: '{marker_source}'. Must be 'shape' or 'intensity'.")

        markers = label(local_maxi_mask)

        if dividing_source.lower() == 'shape':
            watershed_input = -distance
        elif dividing_source.lower() == 'intensity':
            watershed_input = -grayscale_image
        else:
            raise ValueError(f"Unsupported dividing source: '{dividing_source}'. Must be 'shape' or 'intensity'.")

        labeled_objects = watershed(watershed_input, markers, mask=binary_mask)

    elif clumping_method.lower() == 'none':
        labeled_objects = label(binary_mask)
        local_maxi_mask = None
    else:
        raise ValueError(f"Unsupported clumping method: '{clumping_method}'. Only 'watershed' and 'none' are supported.")

    if labeled_objects.max() > 0:
        props = regionprops(labeled_objects)
        labels_to_keep = [
            p.label for p in props
            if min_diameter ** 2 <= p.area <= max_diameter ** 2
        ]
        filtered_mask = np.isin(labeled_objects, labels_to_keep)
        labeled_objects = labeled_objects * filtered_mask

    if clear_border_objects:
        labeled_objects = clear_border(labeled_objects)

    if plot_for_testing:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        plt.subplots_adjust(wspace=0.01, hspace=0.05, left=0, right=1, bottom=0, top=0.95)
        axes = axes.flatten()
        img_height, img_width = grayscale_image.shape
        img_aspect_ratio = img_height / img_width

        axes[0].imshow(grayscale_image)
        axes[0].set_title('1. Original Image')
        axes[0].axis('off')

        counts, bin_centers = hist(grayscale_image.ravel(), nbins=256)
        ax_hist = axes[1]
        ax_hist.set_box_aspect(img_aspect_ratio)
        ax_hist.bar(bin_centers, counts, width=bin_centers[1] - bin_centers[0], color='lightgray')
        ax_hist.set_title(f"Threshold Histogram ({threshold_method}, classes: {kw.get('classes', 3)})")
        ax_hist.set_xlabel("Normalized Intensity")
        ax_hist.set_ylabel("Pixel Count")
        ax_hist.set_yscale('log')
        colors_cmap = plt.cm.get_cmap('Paired', max(len(thresholds), 1))
        for i, t in enumerate(thresholds):
            ax_hist.axvline(t, color=colors_cmap(i), linestyle='--', label=f'Threshold {i+1}: {t:.3f}')
        ax_hist.axvline(threshold_to_apply, color='red', linestyle='-', linewidth=2,
                        label=f'Applied Threshold: {threshold_to_apply:.3f}')
        ax_hist.legend()

        axes[2].imshow(binary_mask, cmap='gray')
        axes[2].set_title('Binary Mask (Post-Threshold)')
        axes[2].axis('off')

        if clumping_method.lower() != 'none' and local_maxi_mask is not None:
            marker_coords = np.argwhere(local_maxi_mask)
            axes[3].imshow(markers, cmap='gray')
            axes[3].scatter(marker_coords[:, 1], marker_coords[:, 0],
                            s=30, marker='o', color='red', label='Markers')
            axes[3].set_title('Seeded Markers for Watershed')
            axes[3].axis('off')
        else:
            random_cmap = _generate_random_cmap(
                n_colors=max(labeled_objects.max(), 1),
                background_color='white', base_cmap='nipy_spectral', seed=0
            )
            axes[3].imshow(labeled_objects, cmap=random_cmap, interpolation='nearest')
            n_objects = len(np.unique(labeled_objects[labeled_objects != 0]))
            axes[3].set_title(f'Identified Objects (Count: {n_objects})')
            axes[3].set_xticks([])
            axes[3].set_yticks([])

        plt.suptitle("Image Processing: Object Identification")

    return labeled_objects
