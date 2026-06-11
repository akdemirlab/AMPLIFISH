"""
FISH spot segmentation for target and centromere channels.

Detects fluorescent spots in FISH images using speckle enhancement followed
by threshold-based object identification.
"""

import logging

import numpy as np

from amplifish.segmentation._primitives import enhance_speckles, identify_primary_objects

logger = logging.getLogger(__name__)


def detect_target_spots(
    R_channel: np.ndarray,
    *,
    disk_radius: int = 20,
    min_diameter: int = 5,
    max_diameter: int = 100,
    hole_diameter: int = 1,
    clear_border_objects: bool = True,
    threshold_method: str = 'otsu',
    threshold_kwargs: dict = None,
    use_log_preprocessing: bool = True,
    clumping_method: str = 'none',
    threshold_smooth_scale: float = 1.3488,
    threshold_correction_factor: float = 1.0,
    sigma_for_declumping: float = 20,
    verbose: bool = False,
) -> np.ndarray:
    """
    Detect target FISH spots in the red channel.

    The caller is responsible for contrast stretching before passing the image
    (e.g. ``skimage.exposure.rescale_intensity`` with percentile bounds).

    Parameters
    ----------
    R_channel : np.ndarray
        Red channel of the FISH image (2-D float in [0, 1]).
    disk_radius : int
        Structuring-element radius for the white top-hat speckle filter.
        Choose slightly larger than the expected spot radius. Default: 20.
    min_diameter : int
        Minimum spot diameter in pixels; smaller objects are discarded. Default: 5.
    max_diameter : int
        Maximum spot diameter in pixels; larger objects are discarded. Default: 100.
    hole_diameter : int
        Holes smaller than ``hole_diameter²`` pixels are filled. Default: 1.
    clear_border_objects : bool
        Remove objects touching the image boundary. Default: True.
    threshold_method : str
        Thresholding strategy: ``'otsu'``, ``'triangle'``, ``'li'``, or ``'yen'``.
        Default: ``'otsu'``.
    threshold_kwargs : dict, optional
        Method-specific keyword arguments. For ``'otsu'``: ``{'classes': 3}``.
        For ``'li'``: ``{'tolerance': ...}`` or ``{'initial_guess': ...}``.
        Ignored by ``'triangle'`` and ``'yen'``.
    use_log_preprocessing : bool
        Apply log1p transform before thresholding to enhance dim spots. Default: True.
    clumping_method : str
        Declumping strategy: ``'none'`` for direct labeling, ``'watershed'`` for
        watershed-based declumping. Default: ``'none'``.
    threshold_smooth_scale : float
        Gaussian sigma applied before threshold calculation (0 = disabled). Default: 1.3488.
    threshold_correction_factor : float
        Multiplier applied to the computed threshold. Default: 1.0.
    sigma_for_declumping : float
        Gaussian sigma for smoothing during watershed seed detection. Default: 20.
    verbose : bool
        When ``True``, enables diagnostic plots inside the primitives. Default: False.

    Returns
    -------
    np.ndarray
        2-D int32 label image; 0 = background, 1..N = spot IDs.
    """
    filtered = enhance_speckles(
        R_channel,
        disk_radius=disk_radius,
        plot_for_testing=verbose,
        plot_cmap='Reds',
    )
    return identify_primary_objects(
        filtered,
        min_diameter=min_diameter,
        max_diameter=max_diameter,
        hole_diameter=hole_diameter,
        clear_border_objects=clear_border_objects,
        threshold_method=threshold_method,
        threshold_kwargs=threshold_kwargs,
        use_log_preprocessing=use_log_preprocessing,
        clumping_method=clumping_method,
        threshold_smooth_scale=threshold_smooth_scale,
        threshold_correction_factor=threshold_correction_factor,
        sigma_for_declumping=sigma_for_declumping,
        plot_for_testing=verbose,
    )


def detect_centromere_spots(
    G_channel: np.ndarray,
    *,
    disk_radius: int = 10,
    min_diameter: int = 5,
    max_diameter: int = 30,
    hole_diameter: int = 1,
    clear_border_objects: bool = True,
    threshold_method: str = 'otsu',
    threshold_kwargs: dict = None,
    use_log_preprocessing: bool = True,
    clumping_method: str = 'none',
    threshold_smooth_scale: float = 1.3488,
    threshold_correction_factor: float = 1.0,
    sigma_for_declumping: float = 20,
    verbose: bool = False,
) -> np.ndarray:
    """
    Detect centromere FISH spots in the green channel.

    The caller is responsible for contrast stretching before passing the image
    (e.g. ``skimage.exposure.rescale_intensity`` with percentile bounds).

    Parameters
    ----------
    G_channel : np.ndarray
        Green channel of the FISH image (2-D float in [0, 1]).
    disk_radius : int
        Structuring-element radius for the white top-hat speckle filter.
        Choose slightly larger than the expected spot radius. Default: 10.
    min_diameter : int
        Minimum spot diameter in pixels; smaller objects are discarded. Default: 5.
    max_diameter : int
        Maximum spot diameter in pixels; larger objects are discarded. Default: 30.
    hole_diameter : int
        Holes smaller than ``hole_diameter²`` pixels are filled. Default: 1.
    clear_border_objects : bool
        Remove objects touching the image boundary. Default: True.
    threshold_method : str
        Thresholding strategy: ``'otsu'``, ``'triangle'``, ``'li'``, or ``'yen'``.
        Default: ``'otsu'``.
    threshold_kwargs : dict, optional
        Method-specific keyword arguments. For ``'otsu'``: ``{'classes': 3}``.
        For ``'li'``: ``{'tolerance': ...}`` or ``{'initial_guess': ...}``.
        Ignored by ``'triangle'`` and ``'yen'``.
    use_log_preprocessing : bool
        Apply log1p transform before thresholding to enhance dim spots. Default: True.
    clumping_method : str
        Declumping strategy: ``'none'`` for direct labeling, ``'watershed'`` for
        watershed-based declumping. Default: ``'none'``.
    threshold_smooth_scale : float
        Gaussian sigma applied before threshold calculation (0 = disabled). Default: 1.3488.
    threshold_correction_factor : float
        Multiplier applied to the computed threshold. Default: 1.0.
    sigma_for_declumping : float
        Gaussian sigma for smoothing during watershed seed detection. Default: 20.
    verbose : bool
        When ``True``, enables diagnostic plots inside the primitives. Default: False.

    Returns
    -------
    np.ndarray
        2-D int32 label image; 0 = background, 1..N = spot IDs.
    """
    filtered = enhance_speckles(
        G_channel,
        disk_radius=disk_radius,
        plot_for_testing=verbose,
        plot_cmap='Greens',
    )
    return identify_primary_objects(
        filtered,
        min_diameter=min_diameter,
        max_diameter=max_diameter,
        hole_diameter=hole_diameter,
        clear_border_objects=clear_border_objects,
        threshold_method=threshold_method,
        threshold_kwargs=threshold_kwargs,
        use_log_preprocessing=use_log_preprocessing,
        clumping_method=clumping_method,
        threshold_smooth_scale=threshold_smooth_scale,
        threshold_correction_factor=threshold_correction_factor,
        sigma_for_declumping=sigma_for_declumping,
        plot_for_testing=verbose,
    )
