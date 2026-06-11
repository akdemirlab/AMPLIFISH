"""
Downstream FISH measurement pipeline.

Takes pre-computed label images — nuclei, target spots, and centromere spots,
each a uint16 TIFF — assigns spots to nuclei, measures per-object regionprops,
computes per-nucleus target-spot entropy, and writes the measurement tables and
overlay images for a single image.

Segmentation and spot detection are run upstream by the caller (see
``demo/run_demo.py``: it calls ``segmentation.nuclei.segment_nuclei`` and
``segmentation.spots.detect_*`` directly, saves the label TIFFs, then calls
:func:`process_precomputed_otsu` to assemble the tables).

Example
-------
>>> from amplifish import process_precomputed_otsu, PipelineConfig
>>> cfg = PipelineConfig(verbose=True)
>>> result = process_precomputed_otsu(
...     row, nuclei_dir, target_dir, centromere_dir, out_images_dir, config=cfg)
>>> result.nuclei.head()
"""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from skimage import io

from amplifish.config import PipelineConfig
from amplifish.utils.image import split_color_channels
from amplifish.measurement.objects import (
    filter_nuclei_by_area,
    relate_fish_to_nuclei,
    measure_objects,
    annotate_tables,
    save_segmentation_images,
)
from amplifish.measurement.entropy import (
    calculate_global_bin_matrix,
    create_global_bin_heatmap,
    create_global_entropy_heatmap,
    compute_entropy_per_nucleus,
)


@dataclass
class PipelineResult:
    """Return type of :func:`process_precomputed_otsu`."""
    nuclei:     pd.DataFrame
    target:     pd.DataFrame
    centromere: pd.DataFrame
    entropy:    pd.DataFrame


def process_precomputed_otsu(
    row: dict,
    nuclei_segment_dir: str,
    target_segment_dir: str,
    centromere_labels_dir: str,
    out_images_dir: str,
    config: PipelineConfig = None,
):
    """
    Run downstream FISH measurement from pre-computed label images.

    Loads the nuclei, target-spot, and centromere-spot label TIFFs for one
    image, assigns spots to nuclei, measures per-object regionprops, computes
    per-nucleus target entropy, and writes the overlay images.

    Parameters
    ----------
    row : dict
        Metadata for a single image with keys:
        ``file_path``, ``patient``, ``sample``, ``image_name``, ``type``.
    nuclei_segment_dir : str
        Directory containing ``<image_name>_nuclei_labels.tiff``.
    target_segment_dir : str
        Directory containing ``<image_name>_target_labels.tiff``.
    centromere_labels_dir : str
        Directory containing ``<image_name>_centromere_labels.tiff``.
        Labels are uint16; 0 = background, 1..N = spot IDs.
    out_images_dir : str
        Parent directory for image outputs. Overlay TIFFs are written into
        ``segment/``, ``debug/``, and ``original/`` subdirectories; heatmaps
        go into ``debug/``.
    config : PipelineConfig, optional
        Runtime configuration. Controls which output files are saved and
        whether verbose diagnostic plots are shown. Defaults to
        ``PipelineConfig()`` (all outputs enabled, no verbose plots).

    Returns
    -------
    PipelineResult or None
        Named result with fields ``nuclei``, ``target``, ``centromere``,
        ``entropy`` — each a :class:`pandas.DataFrame`.
        Returns ``None`` on failure or if any required file is missing.
    """
    if config is None:
        config = PipelineConfig()

    image_name = row['image_name']
    image_path = row['file_path']
    patient_id = row['patient']
    sample_id  = row['sample']
    type_      = row['type']

    nuclei_path     = os.path.join(nuclei_segment_dir,    f'{image_name}_nuclei_labels.tiff')
    target_path     = os.path.join(target_segment_dir,    f'{image_name}_target_labels.tiff')
    centromere_path = os.path.join(centromere_labels_dir, f'{image_name}_centromere_labels.tiff')

    for path, label in [
        (image_path,      'image'),
        (nuclei_path,     'nuclei'),
        (target_path,     'target'),
        (centromere_path, 'centromere'),
    ]:
        if not os.path.exists(path):
            logger.warning("%s file not found: %s. Skipping.", label, path)
            return None

    logger.info("Processing precomputed (Otsu centromere) results for %s", image_name)

    try:
        image = io.imread(image_path)
        R_channel, G_channel, B_channel = split_color_channels(image)

        nuclei_labels = io.imread(nuclei_path).astype(np.int32)
        target        = io.imread(target_path).astype(np.int32)
        centromere    = io.imread(centromere_path).astype(np.int32)

        n_target     = int(target.max())
        n_centromere = int(centromere.max())
        logger.info("%d target spots, %d centromere spots loaded", n_target, n_centromere)

        nuclei, rejected_nuclei = filter_nuclei_by_area(nuclei_labels)

        target_mask = target > 0

        # Heatmap QC images (off by default) go under a debug/ subdir, created
        # only when one is actually requested — otherwise no empty dir is left.
        debug_dir = os.path.join(out_images_dir, 'debug')
        if config.save_bin_heatmap:
            os.makedirs(debug_dir, exist_ok=True)
            global_bin_matrix = calculate_global_bin_matrix(target_mask, image.shape[:2])
            create_global_bin_heatmap(global_bin_matrix, debug_dir, image_name)

        target, centromere, target_parents, centromere_parents = relate_fish_to_nuclei(
            nuclei, target, centromere, verbose=config.verbose
        )

        nuclei_table, target_table, centromere_table = measure_objects(
            nuclei, target, centromere,
            B_channel, R_channel, G_channel,
            target_parents, centromere_parents,
        )

        entropy_data  = compute_entropy_per_nucleus(nuclei, target_mask)
        entropy_table = pd.DataFrame(entropy_data)
        if config.save_entropy_heatmap:
            os.makedirs(debug_dir, exist_ok=True)
            create_global_entropy_heatmap(image, nuclei, entropy_data, debug_dir, image_name)

        annotate_tables([nuclei_table, target_table, centromere_table, entropy_table],
                        image_name, patient_id, sample_id, type_)
        save_segmentation_images(
            image, nuclei, rejected_nuclei, target, centromere,
            out_images_dir, image_name,
            save_original=config.save_original,
            save_segment=config.save_segment,
            save_debug_tiff=config.save_debug_tiff,
        )

        logger.info("Completed %s.", image_name)
        return PipelineResult(
            nuclei=nuclei_table,
            target=target_table,
            centromere=centromere_table,
            entropy=entropy_table,
        )

    except Exception as e:
        logger.exception("Error processing %s: %s", image_name, e)
        return None
