"""
Object-level measurements for FISH segmentation results.

Provides nucleus area filtering, FISH spot-to-nucleus assignment,
regionprops measurement, metadata annotation, and segmentation image output.
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
from skimage.measure import regionprops, regionprops_table
from skimage.segmentation import mark_boundaries

from amplifish.utils.viz import _generate_random_cmap

def _mark_image(mask_source_image: np.ndarray, image_to_mark: np.ndarray) -> np.ndarray:
    """Zero out pixels in `image_to_mark` that are background in `mask_source_image`."""
    binary_mask = (mask_source_image > 0)
    return image_to_mark * binary_mask.astype(image_to_mark.dtype)


def _relate_objects_by_centroid(
    parent_labels: np.ndarray,
    child_labels: np.ndarray,
    parent_col: str = 'Parent_Object_Label',
    child_col: str = 'Child_Object_Label',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Assign each child object to the parent whose region contains the child's centroid.

    Parameters
    ----------
    parent_labels : np.ndarray
        Labeled image of parent objects (e.g., nuclei).
    child_labels : np.ndarray
        Labeled image of child objects (e.g., FISH spots).
    parent_col : str
        Column name for parent IDs in the output DataFrame.
    child_col : str
        Column name for child IDs in the output DataFrame.
    verbose : bool
        When ``True``, display a relationship visualization (children
        coloured by parent ID).

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame: child_col → parent_col.
    """
    child_properties = regionprops(child_labels)
    results = {child_col: [], parent_col: []}

    for props in child_properties:
        child_id = props.label
        centroid_y, centroid_x = props.centroid
        max_y, max_x = parent_labels.shape
        idx_y = int(np.clip(round(centroid_y), 0, max_y - 1))
        idx_x = int(np.clip(round(centroid_x), 0, max_x - 1))
        parent_id = parent_labels[idx_y, idx_x]

        results[child_col].append(child_id)
        if parent_id == 0:
            logger.warning("%s %s centroid (%d, %d) fell on background.", child_col, child_id, idx_x, idx_y)
        results[parent_col].append(parent_id)

    n_total = len(results[child_col])
    n_bg = sum(1 for p in results[parent_col] if p == 0)
    logger.info("%s: %d spots, %d assigned, %d on background", child_col, n_total, n_total - n_bg, n_bg)

    relationship_df = pd.DataFrame(results)

    if verbose:
        related_child_labels = np.zeros_like(child_labels)
        for child_id in relationship_df[child_col]:
            parent_id = relationship_df[relationship_df[child_col] == child_id][parent_col].iloc[0]
            related_child_labels[child_labels == child_id] = parent_id

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        plt.suptitle("RelateObjects: Child-Parent Assignment", fontsize=14)
        random_cmap = _generate_random_cmap(
            n_colors=max(parent_labels.max(), 1), base_cmap='nipy_spectral', seed=0
        )
        axes[0].imshow(parent_labels, cmap=random_cmap, interpolation='nearest')
        axes[0].set_title('Parent Objects')
        axes[0].axis('off')
        axes[1].imshow(child_labels, cmap=random_cmap, interpolation='nearest')
        axes[1].set_title('Child Objects')
        axes[1].axis('off')
        axes[2].imshow(related_child_labels, cmap=random_cmap, interpolation='nearest')
        axes[2].set_title('Children Colored by Parent ID')
        axes[2].axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return relationship_df


# Default nucleus area bounds (pixels²)
MIN_NUCLEUS_AREA = 100 ** 2
MAX_NUCLEUS_AREA = 2000 ** 2


def filter_nuclei_by_area(
    nuclei_labels: np.ndarray,
    min_area: int = MIN_NUCLEUS_AREA,
    max_area: int = MAX_NUCLEUS_AREA,
):
    """
    Keep only nuclei whose area falls within [min_area, max_area].

    Parameters
    ----------
    nuclei_labels : np.ndarray
        Integer label image from a segmentation model.
    min_area : int
        Minimum nucleus area in pixels².
    max_area : int
        Maximum nucleus area in pixels².

    Returns
    -------
    tuple : (nuclei, rejected_nuclei)
        nuclei          – label image with only accepted nuclei.
        rejected_nuclei – label image with only rejected nuclei.
    """
    nuclei = np.zeros_like(nuclei_labels)
    for region in regionprops(nuclei_labels):
        if min_area <= region.area <= max_area:
            nuclei[nuclei_labels == region.label] = region.label

    rejected_nuclei = nuclei_labels.copy()
    rejected_nuclei[nuclei > 0] = 0

    n_accepted = len(np.unique(nuclei[nuclei > 0]))
    n_rejected = len(np.unique(rejected_nuclei[rejected_nuclei > 0]))
    logger.info("Nucleus filter: %d accepted, %d rejected (area [%d\u2013%d] px\u00b2)",
                n_accepted, n_rejected, min_area, max_area)

    return nuclei, rejected_nuclei


def relate_fish_to_nuclei(
    nuclei: np.ndarray,
    target: np.ndarray,
    centromere: np.ndarray,
    verbose: bool = False,
):
    """
    Assign each FISH spot to the nucleus that contains its centroid.

    Parameters
    ----------
    nuclei : np.ndarray
        Accepted nucleus label image.
    target : np.ndarray
        Target spot label image.
    centromere : np.ndarray
        Centromere spot label image.
    verbose : bool
        When ``True``, pass to :func:`_relate_objects_by_centroid` to show
        a parent-child assignment visualization.

    Returns
    -------
    tuple : (target, centromere, target_parents, centromere_parents)
        target, centromere    – label images masked to nuclei interiors.
        target_parents        – DataFrame mapping Target_Label → Nucleus_Label.
        centromere_parents    – DataFrame mapping Centromere_Label → Nucleus_Label.
    """
    target = _mark_image(nuclei, target)
    centromere = _mark_image(nuclei, centromere)
    target_parents = _relate_objects_by_centroid(
        nuclei, target, parent_col='Nucleus_Label', child_col='Target_Label',
        verbose=verbose,
    )
    centromere_parents = _relate_objects_by_centroid(
        nuclei, centromere, parent_col='Nucleus_Label', child_col='Centromere_Label',
        verbose=verbose,
    )
    return target, centromere, target_parents, centromere_parents


def measure_objects(
    nuclei: np.ndarray, target: np.ndarray, centromere: np.ndarray,
    B_channel: np.ndarray, R_channel: np.ndarray, G_channel: np.ndarray,
    target_parents: pd.DataFrame, centromere_parents: pd.DataFrame,
):
    """
    Measure regionprops for nuclei, targets, and centromeres and merge parent labels.

    Parameters
    ----------
    nuclei : np.ndarray
        Accepted nucleus label image.
    target : np.ndarray
        Target spot label image (masked to nuclei).
    centromere : np.ndarray
        Centromere spot label image (masked to nuclei).
    B_channel : np.ndarray
        Blue channel (nucleus intensity reference).
    R_channel : np.ndarray
        Red channel (target intensity reference).
    G_channel : np.ndarray
        Green channel (centromere intensity reference).
    target_parents : pd.DataFrame
        Output of :func:`relate_fish_to_nuclei` for targets.
    centromere_parents : pd.DataFrame
        Output of :func:`relate_fish_to_nuclei` for centromeres.

    Returns
    -------
    tuple : (nuclei_table, target_table, centromere_table)
        Each is a :class:`pandas.DataFrame` with regionprops and parent labels.
    """
    nuclei_table = pd.DataFrame(
        regionprops_table(nuclei, intensity_image=B_channel,
                          properties=['label', 'area', 'centroid', 'intensity_mean'])
    ).rename(columns={'label': 'Nucleus_Label'})

    target_table = pd.DataFrame(
        regionprops_table(target, intensity_image=R_channel,
                          properties=['label', 'area', 'centroid', 'intensity_mean',
                                      'intensity_max', 'eccentricity', 'solidity',
                                      'axis_major_length'])
    )
    centromere_table = pd.DataFrame(
        regionprops_table(centromere, intensity_image=G_channel,
                          properties=['label', 'area', 'centroid', 'intensity_mean',
                                      'intensity_max', 'eccentricity', 'solidity',
                                      'axis_major_length'])
    )

    target_table = target_table.merge(
        target_parents, left_on='label', right_on='Target_Label', how='left'
    ).drop(columns=['label'])
    centromere_table = centromere_table.merge(
        centromere_parents, left_on='label', right_on='Centromere_Label', how='left'
    ).drop(columns=['label'])

    return nuclei_table, target_table, centromere_table


def annotate_tables(
    dfs: list,
    image_name: str,
    patient_id: str,
    sample_id: str,
    type_: str,
) -> None:
    """
    Add image-level metadata columns to each DataFrame in-place.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        Tables to annotate (nuclei, target, centromere, entropy).
    image_name : str
        Image identifier.
    patient_id : str
        Patient identifier.
    sample_id : str
        Sample identifier.
    type_ : str
        Image type label.
    """
    for df in dfs:
        df['image_name'] = image_name
        df['patient_id'] = patient_id
        df['sample_id'] = sample_id
        df['type'] = type_


def save_segmentation_images(
    image: np.ndarray,
    nuclei: np.ndarray,
    rejected_nuclei: np.ndarray,
    target: np.ndarray,
    centromere: np.ndarray,
    out_dir: str,
    image_name: str,
    save_original: bool = True,
    save_segment: bool = True,
    save_debug_tiff: bool = True,
) -> None:
    """
    Save debug, segment, and original TIFF images for one processed image.

    Three optional files can be written (each controlled by a flag), each in
    its own subdirectory under ``out_dir``:

    * ``<out_dir>/debug/<image_name>_debug.tiff``    – all boundaries overlaid,
      rejected nuclei shown in red (useful for QC).
    * ``<out_dir>/segment/<image_name>_segment.tiff`` – accepted nuclei, targets,
      and centromeres only (no rejected nuclei).
    * ``<out_dir>/original/<image_name>_original.tiff`` – raw input image,
      unmodified.

    Parameters
    ----------
    image : np.ndarray
        Full RGB FISH image (H, W, 3).
    nuclei : np.ndarray
        Accepted nucleus label image.
    rejected_nuclei : np.ndarray
        Rejected nucleus label image.
    target : np.ndarray
        Target spot label image.
    centromere : np.ndarray
        Centromere spot label image.
    out_dir : str
        Parent directory; ``debug/``, ``segment/``, and ``original/``
        subdirectories are created automatically as needed.
    image_name : str
        Base name used as the filename prefix.
    save_original : bool
        Save ``<image_name>_original.tiff``.
    save_segment : bool
        Save ``<image_name>_segment.tiff``.
    save_debug_tiff : bool
        Save ``<image_name>_debug.tiff``.
    """
    if save_debug_tiff:
        debug_dir = os.path.join(out_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        debug_image = mark_boundaries(image, nuclei, color=(1, 1, 1), mode='thick')
        debug_image = mark_boundaries(debug_image, rejected_nuclei, color=(1, 0, 0), mode='thick')
        debug_image = mark_boundaries(debug_image, target, color=(1, 0.5, 0.5), mode='thick')
        debug_image = mark_boundaries(debug_image, centromere, color=(0.5, 1, 0.5), mode='thick')
        io.imsave(os.path.join(debug_dir, f"{image_name}_debug.tiff"),
                  img_as_ubyte(debug_image), check_contrast=False)

    if save_segment:
        segment_dir = os.path.join(out_dir, 'segment')
        os.makedirs(segment_dir, exist_ok=True)
        segment_image = mark_boundaries(image, nuclei, color=(1, 1, 1), mode='thick')
        segment_image = mark_boundaries(segment_image, target, color=(1, 0.5, 0.5), mode='thick')
        segment_image = mark_boundaries(segment_image, centromere, color=(0.5, 1, 0.5), mode='thick')
        io.imsave(os.path.join(segment_dir, f"{image_name}_segment.tiff"),
                  img_as_ubyte(segment_image), check_contrast=False)

    if save_original:
        original_dir = os.path.join(out_dir, 'original')
        os.makedirs(original_dir, exist_ok=True)
        io.imsave(os.path.join(original_dir, f"{image_name}_original.tiff"),
                  img_as_ubyte(image), check_contrast=False)
