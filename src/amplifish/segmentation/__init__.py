"""
FISH image segmentation subpackage.

Run a backend's ``segment_nuclei`` and ``spots.detect_*`` upstream, save the
label TIFFs, then assemble per-nucleus tables with
:func:`amplifish.pipeline.process_precomputed_otsu` (see ``demo/run_demo.py``).

Submodules
----------
nuclei
    Cellpose nucleus segmentation (cellpose ships as a core dependency).
    Provides ``load_model()`` and ``segment_nuclei()``.
spots
    FISH spot detection for target (red) and centromere (green) channels.
    Provides ``detect_target_spots()`` and ``detect_centromere_spots()``.
"""

from amplifish.segmentation.spots import detect_target_spots, detect_centromere_spots

__all__ = ["detect_target_spots", "detect_centromere_spots"]
