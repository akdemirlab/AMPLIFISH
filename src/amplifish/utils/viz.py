"""
Visualization utilities shared across the fish subpackage.
"""

import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb

logger = logging.getLogger(__name__)


def _generate_random_cmap(n_colors=256, base_cmap='viridis', background_color='black', seed=None):
    """
    Generate a random ListedColormap with an explicit background color at index 0.

    Parameters
    ----------
    n_colors : int
        Number of colors in the colormap.
    base_cmap : str
        Base Matplotlib colormap to draw from.
    background_color : str or tuple
        Color for index 0 (background). Accepts Matplotlib color strings or RGB tuples.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    matplotlib.colors.ListedColormap
    """
    if seed is not None:
        np.random.seed(seed)

    cmap = plt.get_cmap(base_cmap)
    vals = np.linspace(0, 1, n_colors)
    object_vals = vals[1:]
    np.random.shuffle(object_vals)
    colors = cmap(np.concatenate(([0], object_vals)))

    try:
        bg_rgb = to_rgb(background_color)
        colors[0, :len(bg_rgb)] = bg_rgb
        if colors.shape[1] == 4 and len(bg_rgb) == 3:
            colors[0, 3] = 1.0
    except ValueError:
        logger.warning("Could not interpret background color '%s'. Defaulting to black.", background_color)
        colors[0, :] = (0.0, 0.0, 0.0, 1.0)

    return ListedColormap(colors)
