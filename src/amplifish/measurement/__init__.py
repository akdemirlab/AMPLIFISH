"""
Post-segmentation measurement subpackage for FISH images.

Submodules
----------
objects
    Nucleus area filtering, FISH spot-to-nucleus assignment, regionprops
    measurement, table annotation, and segmentation image output.
    Provides ``filter_nuclei_by_area()``, ``relate_fish_to_nuclei()``,
    ``measure_objects()``, ``annotate_tables()``, ``save_segmentation_images()``.
entropy
    Shannon entropy of target-spot spatial distribution: spatial binning,
    per-nucleus entropy computation, and heatmap visualization.
    Provides ``compute_entropy_per_nucleus()``, ``calculate_global_bin_matrix()``.
"""

from amplifish.measurement.objects import (
    filter_nuclei_by_area,
    relate_fish_to_nuclei,
    measure_objects,
    annotate_tables,
    save_segmentation_images,
)
from amplifish.measurement.entropy import (
    compute_entropy_per_nucleus,
    calculate_global_bin_matrix,
)

__all__ = [
    "filter_nuclei_by_area",
    "relate_fish_to_nuclei",
    "measure_objects",
    "annotate_tables",
    "save_segmentation_images",
    "compute_entropy_per_nucleus",
    "calculate_global_bin_matrix",
]
