"""
AMPLIFISH — FISH image analysis for oncogene amplification typing.

AMPLIFISH turns multi-channel FISH microscopy images into per-nucleus feature
tables (target/centromere spot counts, Shannon entropy of target-spot spatial
distribution, nucleus shape) that distinguish amplification patterns such as
ecDNA (extrachromosomal DNA) from HSR (homogeneously staining regions).

Public API
----------
process_precomputed_otsu
    Run the downstream pipeline (spot-nucleus assignment, measurement, entropy,
    annotation, output) from pre-computed nucleus / target / centromere label
    TIFFs. Segmentation and spot detection are run upstream by the caller.
PipelineConfig
    Runtime configuration dataclass (which outputs to save, verbose plots).
PipelineResult
    Named result (``nuclei``, ``target``, ``centromere``, ``entropy`` tables).
correct_image / stretch_channel / calibrate_hi_floors
    Image correction helpers: percentile contrast stretch and white-top-hat
    speckle enhancement, with an optional global ``hi_floor`` calibration so
    dim images are not over-amplified.
setup_logging
    Enable console / file logging (the package is silent by default).

Examples
--------
Segment + detect upstream, then assemble per-nucleus tables (Cellpose backend)::

    import amplifish
    from amplifish.segmentation import nuclei as cp_seg

    amplifish.setup_logging()
    model = cp_seg.load_model(gpu=True)          # stock CellposeSAM model
    # ... run cp_seg.segment_nuclei + segmentation.spots.detect_* and save the
    # nuclei / target / centromere label TIFFs (see demo/run_demo.py) ...
    result = amplifish.process_precomputed_otsu(
        row, nuclei_dir, target_dir, centromere_dir, out_images_dir)
    result.nuclei.head()

Submodules
----------
segmentation
    ``segmentation.nuclei`` — Cellpose nucleus segmentation.
    ``segmentation.spots``  — target / centromere spot detection.
measurement
    ``measurement.objects``  — area filtering, spot-nucleus assignment, regionprops.
    ``measurement.entropy``  — per-nucleus target-spot Shannon entropy.
utils
    ``utils.image`` — channel splitting and image correction (percentile stretch
    + white-top-hat + hi_floor calibration).
    ``utils.viz``   — label colourmaps.
"""

import logging

__version__ = "0.1.0"

from amplifish.pipeline import (
    process_precomputed_otsu,
    PipelineResult,
)
from amplifish.config import PipelineConfig
from amplifish.utils.image import (
    stretch_channel,
    enhance_channel,
    calibrate_hi_floors,
    correct_image,
)


def setup_logging(level: int = logging.INFO, logfile: str = None) -> None:
    """Configure logging for AMPLIFISH.

    Call once at the start of a script or notebook.  Without calling this,
    AMPLIFISH is silent unless the caller has already attached handlers to the
    root logger (standard Python logging behaviour for libraries).

    Parameters
    ----------
    level : int
        Logging level, e.g. ``logging.INFO`` or ``logging.DEBUG``.
    logfile : str, optional
        Path to a log file.  When given, messages are written there in
        addition to the console.
    """
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("amplifish")
    logger.setLevel(level)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)


__all__ = [
    "process_precomputed_otsu",
    "PipelineResult",
    "PipelineConfig",
    "stretch_channel",
    "enhance_channel",
    "calibrate_hi_floors",
    "correct_image",
    "setup_logging",
    "__version__",
]
