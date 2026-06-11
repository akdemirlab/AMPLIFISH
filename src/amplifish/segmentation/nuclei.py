"""
Cellpose-based nucleus segmentation for FISH images.

Exposes two functions:

* ``load_model(**kwargs)`` – load and return the model.
* ``segment_nuclei(B_channel, model, **kwargs) -> np.ndarray`` – run inference
  on the blue (DAPI) channel and return an integer label image.

An alternative backend can be added later as another module exposing the same
two functions — callers segment, save the label TIFFs, then hand them to
:func:`amplifish.pipeline.process_precomputed_otsu` (see ``demo/run_demo.py``).

Cellpose (a core dependency) is installed with the package; no extra is needed.

Typical usage
-------------
>>> from amplifish.segmentation import nuclei as cp_seg
>>> model = cp_seg.load_model(model_path="/path/to/custom_model", gpu=True)
>>> nuclei_labels = cp_seg.segment_nuclei(dapi_channel, model)
"""

import logging
import os

import cv2

logger = logging.getLogger(__name__)
import numpy as np
from skimage.segmentation import clear_border
from csbdeep.utils import normalize

# Default resize ratio: scales images from native 2448×2048 to ~1000px wide,
# matching the resolution used during fine-tuning of the CellPoseSAM models.
DEFAULT_RESIZE_RATIO = 1000 / 2448


def load_model(model_path: str = None, gpu: bool = True):
    """
    Load a Cellpose segmentation model.

    Parameters
    ----------
    model_path : str, optional
        Path to a fine-tuned CellposeModel weights file. If None or the path
        does not exist, cellpose's pretrained CellposeSAM (``'cpsam'``) model is
        used — this is the default for the AMPLIFISH demo, so no fine-tuning is
        required. The ``cpsam`` weights are not bundled here: cellpose downloads
        them (~1.2 GB) on first use and caches them under ``~/.cellpose/models/``.
    gpu : bool
        Request GPU acceleration if available.

    Returns
    -------
    cellpose.models.CellposeModel
        A loaded model, using either custom weights or pretrained CellposeSAM.

    Raises
    ------
    ImportError
        If ``cellpose`` is not installed.
    """
    try:
        from cellpose import models, core
    except ImportError as e:
        raise ImportError(
            "cellpose is required for Cellpose segmentation. "
            "Reinstall the package with: pip install amplifish"
        ) from e

    use_gpu = gpu and core.use_gpu()
    logger.info("Cellpose: using GPU=%s", use_gpu)

    if model_path and os.path.exists(model_path):
        logger.info("Loading custom Cellpose model: %s", model_path)
        return models.CellposeModel(gpu=use_gpu, pretrained_model=model_path)

    logger.info("Loading pretrained CellposeSAM ('cpsam') model.")
    return models.CellposeModel(gpu=use_gpu)


def segment_nuclei(
    B_channel: np.ndarray,
    model,
    *,
    niter: int = 250,
    flow_threshold: float = 0.55,
    cellprob_threshold: float = 0.0,
    diameter=None,
    resize_ratio: float = DEFAULT_RESIZE_RATIO,
) -> np.ndarray:
    """
    Segment nuclei in the blue channel using Cellpose.

    Parameters
    ----------
    B_channel : np.ndarray
        Blue (DAPI) channel of the FISH image.
    model : cellpose model
        Loaded Cellpose model (from :func:`load_model`).
    niter : int
        Number of Cellpose iterations.
    flow_threshold : float
        Cellpose flow error threshold.
    cellprob_threshold : float
        Cellpose cell-probability threshold.
    diameter : float or None
        Expected nucleus diameter in pixels. None = auto-estimate.
    resize_ratio : float
        Scale factor applied before inference to match training resolution.

    Returns
    -------
    np.ndarray
        Integer label image of segmented nuclei (border objects removed).
    """
    img_normalized = normalize(B_channel, 1, 99, axis=(0, 1))
    img_uint8 = (img_normalized * 255).astype(np.uint8)
    H0, W0 = img_uint8.shape[:2]
    new_h, new_w = int(H0 * resize_ratio), int(W0 * resize_ratio)
    img_small = cv2.resize(img_uint8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    masks, _, _ = model.eval(
        img_small,
        niter=niter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        diameter=diameter,
    )
    nuclei_labels = cv2.resize(
        masks.astype(np.uint16), (W0, H0), interpolation=cv2.INTER_NEAREST
    ).astype(np.int32)
    return clear_border(nuclei_labels)
