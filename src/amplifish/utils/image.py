"""
Image-array utilities shared across the fish subpackage.

Two groups of helpers, all operating directly on image arrays:

* **Channel layout** — :func:`split_color_channels`.
* **Correction** — the standard pre-processing applied before nucleus
  segmentation and spot detection: percentile contrast stretch and
  white-top-hat speckle enhancement, with an optional global ``hi_floor``
  calibration that prevents dim images from having their background
  over-amplified. These are deliberately small, composable functions so they
  can be reused on any channel layout:

  * :func:`stretch_channel`    — single-channel percentile stretch to ``[0, 1]``.
  * :func:`enhance_channel`    — white-top-hat speckle enhancement (a thin
    wrapper around :func:`amplifish.segmentation._primitives.enhance_speckles`).
  * :func:`calibrate_hi_floors`— compute a robust global upper bound per channel
    from a collection of images.
  * :func:`correct_image`      — convenience for the common
    target (red) / centromere (green) / nuclei (blue) FISH layout: returns a
    display RGB (stretch only) and a corrected RGB (stretch + top-hat on the two
    spot channels).

Example
-------
>>> from amplifish.utils.image import calibrate_hi_floors, correct_image
>>> floors = calibrate_hi_floors({'R': r_imgs, 'G': g_imgs, 'B': b_imgs})
>>> display_rgb, corrected_rgb = correct_image(R, G, B, hi_floors=floors)
"""

import logging
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
from skimage.util import img_as_ubyte

from amplifish.segmentation._primitives import enhance_speckles

log = logging.getLogger(__name__)


def split_color_channels(color_image: np.ndarray) -> tuple:
    """
    Split a 3-channel RGB image into separate Red, Green, Blue float64 arrays.

    Each output channel is scaled to [0.0, 1.0] regardless of the input dtype
    (uint8, uint16, float, etc.).

    Parameters
    ----------
    color_image : np.ndarray
        3D array of shape (H, W, 3).

    Returns
    -------
    tuple of (R, G, B) : three 2D float64 arrays.

    Raises
    ------
    ValueError
        If the input does not have exactly 3 channels.
    """
    if color_image.ndim != 3 or color_image.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel (RGB) array.")

    max_val = np.iinfo(color_image.dtype).max if np.issubdtype(color_image.dtype, np.integer) else 1.0
    float_image = color_image.astype(np.float64) / max_val
    return float_image[:, :, 0], float_image[:, :, 1], float_image[:, :, 2]


def stretch_channel(
    channel: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.9,
    hi_floor: float = 0.0,
) -> np.ndarray:
    """Percentile contrast-stretch a single channel to ``[0, 1]`` float32.

    Parameters
    ----------
    channel : np.ndarray
        2-D grayscale image (any numeric dtype).
    p_low, p_high : float
        Lower / upper percentiles used as the stretch bounds.
    hi_floor : float
        Global minimum for the upper bound. If the image's own ``p_high``
        intensity falls below this value, ``hi_floor`` is used instead. This
        prevents dim images from having their background over-amplified.
        Pass ``0.0`` (the default) to disable.

    Returns
    -------
    np.ndarray
        float32 image scaled to ``[0, 1]``. Returns all zeros if the computed
        upper bound is not greater than the lower bound (e.g. a blank tile).
    """
    ch = channel.astype(np.float32)
    lo = np.percentile(ch, p_low)
    hi = max(np.percentile(ch, p_high), hi_floor)
    if hi <= lo:
        return np.zeros_like(ch)
    return np.clip((ch - lo) / (hi - lo), 0.0, 1.0)


def enhance_channel(channel: np.ndarray, disk_radius: int) -> np.ndarray:
    """White-top-hat speckle enhancement for one channel.

    Highlights small bright features (FISH spots) and suppresses background.
    Thin wrapper around :func:`amplifish.segmentation._primitives.enhance_speckles`
    that always returns float32.

    Parameters
    ----------
    channel : np.ndarray
        Grayscale image. Best applied to an already-stretched ``[0, 1]`` image.
    disk_radius : int
        Radius of the disk structuring element — choose slightly larger than
        the expected spot radius (e.g. ``20`` for target, ``10`` for centromere
        spots, matching the spot-detection defaults).

    Returns
    -------
    np.ndarray
        float32 enhanced image.
    """
    return enhance_speckles(channel, disk_radius=disk_radius).astype(np.float32)


def calibrate_hi_floors(
    channel_images: Mapping[str, Iterable[np.ndarray]],
    p_high: float = 99.9,
    floor_percentile: float = 10.0,
) -> Dict[str, float]:
    """Compute a robust global upper-bound (``hi_floor``) per channel.

    For each channel, the ``p_high`` intensity of every supplied image is
    collected and the ``floor_percentile``-th percentile of those values is
    returned. Feeding the result back into :func:`stretch_channel` as
    ``hi_floor`` keeps dim images from being over-stretched, while bright
    images are unaffected (their own ``p_high`` exceeds the floor).

    Parameters
    ----------
    channel_images : mapping of str to iterable of np.ndarray
        Maps a channel name (e.g. ``'R'``, ``'G'``, ``'B'``) to the images to
        calibrate from. Images are processed one at a time, so a generator of
        lazily-loaded arrays is fine for large datasets.
    p_high : float
        Upper percentile used as each image's bright reference, matching the
        value later passed to :func:`stretch_channel`.
    floor_percentile : float
        Percentile across images used as the global floor (lower = more
        permissive). ``10`` matches the cell-line calibration.

    Returns
    -------
    dict of str to float
        Channel name → ``hi_floor`` value. Channels with no usable images map
        to ``0.0`` (i.e. no floor).
    """
    floors: Dict[str, float] = {}
    for name, images in channel_images.items():
        highs = []
        for img in images:
            arr = np.asarray(img, dtype=np.float32)
            highs.append(float(np.percentile(arr, p_high)))
        if highs:
            floors[name] = float(np.percentile(highs, floor_percentile))
        else:
            log.warning("calibrate_hi_floors: no images for channel %r; floor=0", name)
            floors[name] = 0.0
    log.info("Calibrated hi_floors: %s", floors)
    return floors


def correct_image(
    R: np.ndarray,
    G: np.ndarray,
    B: np.ndarray,
    hi_floors: Optional[Mapping[str, float]] = None,
    p_low: float = 1.0,
    p_high: float = 99.9,
    target_radius: int = 20,
    centromere_radius: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Correct a three-channel FISH image (target / centromere / nuclei).

    Convenience wrapper around :func:`stretch_channel` and
    :func:`enhance_channel` for the common FISH layout where the red channel
    holds target-gene spots, green holds centromere spots, and blue holds the
    DAPI nucleus stain. Returns two uint8 RGB images:

    * **display**   — percentile-stretched only, for human viewing.
    * **corrected** — stretch + white-top-hat on the two spot channels (R, G);
      the nucleus channel (B) is stretched only. This mirrors the first steps
      of :func:`amplifish.segmentation.spots.detect_target_spots` /
      ``detect_centromere_spots`` and is the recommended background image for
      segmentation overlays.

    Parameters
    ----------
    R, G, B : np.ndarray
        Target (red), centromere (green), and nucleus/DAPI (blue) channels.
    hi_floors : mapping of str to float, optional
        Per-channel ``hi_floor`` values with keys ``'R'``, ``'G'``, ``'B'``
        (e.g. from :func:`calibrate_hi_floors`). Missing keys default to 0.
    p_low, p_high : float
        Percentile stretch bounds, applied to all three channels.
    target_radius, centromere_radius : int
        White-top-hat disk radii for the R and G channels respectively.

    Returns
    -------
    (display_rgb, corrected_rgb) : tuple of np.ndarray
        Two uint8 RGB arrays of shape ``(H, W, 3)``.
    """
    floors = dict(hi_floors) if hi_floors else {}
    fr, fg, fb = floors.get("R", 0.0), floors.get("G", 0.0), floors.get("B", 0.0)

    R_s = stretch_channel(R, p_low, p_high, hi_floor=fr)
    G_s = stretch_channel(G, p_low, p_high, hi_floor=fg)
    B_s = stretch_channel(B, p_low, p_high, hi_floor=fb)

    display_rgb = img_as_ubyte(np.stack([R_s, G_s, B_s], axis=2))

    R_corr = enhance_channel(R_s, disk_radius=target_radius)
    G_corr = enhance_channel(G_s, disk_radius=centromere_radius)
    corrected_rgb = img_as_ubyte(np.stack([R_corr, G_corr, B_s], axis=2))

    return display_rgb, corrected_rgb
