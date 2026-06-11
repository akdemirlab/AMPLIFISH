"""
Runtime configuration for the FISH processing pipeline.

Import via the subpackage root::

    from amplifish import PipelineConfig

Example — debug a single image (verbose inline plots, all outputs saved)::

    cfg = PipelineConfig(verbose=True)

Example — fast batch run (suppress heavy debug files)::

    cfg = PipelineConfig(
        save_debug_tiff=False,
        save_bin_heatmap=False,
        save_entropy_heatmap=False,
    )
"""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """
    Runtime configuration for :func:`amplifish.pipeline.process_precomputed_otsu`.

    All fields default to the "full output, no verbose plots" mode that matches
    the original pipeline behaviour.  Override only what you need.

    Output flags
    ------------
    save_original : bool
        Save ``<image_name>_original.tiff`` — the raw input image.
    save_segment : bool
        Save ``<image_name>_segment.tiff`` — accepted nuclei, target, and
        centromere boundaries overlaid on the input image.
    save_debug_tiff : bool
        Save ``<image_name>_debug.tiff`` — like segment but with rejected
        nuclei shown in red.  Useful for QC; can be skipped in batch runs.
    save_bin_heatmap : bool
        Save ``<image_name>_global_target_bin_heatmap.png``.
    save_entropy_heatmap : bool
        Save ``<image_name>_global_entropy_heatmap.png``.

    Diagnostics
    -----------
    verbose : bool
        When ``True``, enables ``plot_for_testing`` in the spot-detection and
        spot-nucleus assignment sub-functions.  Produces inline matplotlib
        plots showing speckle enhancement, thresholding, and parent-child
        assignment results.  Intended for single-image debugging in notebooks.
        Has no effect in batch/headless runs where matplotlib has no display.
    """

    # ── Output control ─────────────────────────────────────────────────────────
    save_original:        bool = True
    save_segment:         bool = True
    save_debug_tiff:      bool = True
    save_bin_heatmap:     bool = True
    save_entropy_heatmap: bool = True

    # ── Diagnostics ────────────────────────────────────────────────────────────
    verbose: bool = False
