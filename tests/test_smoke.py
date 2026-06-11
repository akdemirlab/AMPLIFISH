"""Smoke tests for AMPLIFISH.

These exercise the dependency-light core (import surface, correction, spot
detection, entropy) on small synthetic arrays. They do not require cellpose,
torch, or a GPU.
"""

import numpy as np
import pytest


def test_import_and_public_api():
    import amplifish

    assert amplifish.__version__
    for name in [
        "process_precomputed_otsu",
        "PipelineConfig",
        "PipelineResult",
        "correct_image",
        "stretch_channel",
        "calibrate_hi_floors",
        "setup_logging",
    ]:
        assert hasattr(amplifish, name), name


def test_stretch_channel_range():
    from amplifish.utils.image import stretch_channel

    img = np.random.randint(0, 4000, size=(64, 64)).astype(np.uint16)
    out = stretch_channel(img, p_low=1.0, p_high=99.9)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_stretch_channel_blank_is_zero():
    from amplifish.utils.image import stretch_channel

    out = stretch_channel(np.zeros((32, 32), dtype=np.uint16))
    assert np.all(out == 0.0)


def test_calibrate_hi_floors_keys():
    from amplifish.utils.image import calibrate_hi_floors

    imgs = [np.random.randint(0, 1000, (32, 32)).astype(np.uint16) for _ in range(3)]
    floors = calibrate_hi_floors({"R": imgs, "G": imgs, "B": imgs})
    assert set(floors) == {"R", "G", "B"}
    assert all(isinstance(v, float) for v in floors.values())


def test_correct_image_shapes():
    from amplifish.utils.image import correct_image

    R = np.random.randint(0, 4000, (48, 48)).astype(np.uint16)
    G = R.copy()
    B = R.copy()
    display_rgb, corrected_rgb = correct_image(R, G, B)
    assert display_rgb.shape == (48, 48, 3)
    assert corrected_rgb.shape == (48, 48, 3)
    assert display_rgb.dtype == np.uint8


def test_detect_target_spots_counts():
    """Two bright disks on a dark field should yield two labelled spots."""
    from skimage.draw import disk
    from amplifish.segmentation.spots import detect_target_spots

    img = np.zeros((128, 128), dtype=np.float32)
    for cy, cx in [(40, 40), (90, 90)]:
        rr, cc = disk((cy, cx), 6, shape=img.shape)
        img[rr, cc] = 1.0

    labels = detect_target_spots(img, disk_radius=10, threshold_method="yen")
    assert labels.dtype.kind in "iu"
    assert labels.max() == 2  # integer labels: max == number of spots


def test_entropy_runs_on_synthetic():
    from amplifish.measurement.entropy import compute_entropy_per_nucleus

    nuclei = np.zeros((64, 64), dtype=np.int32)
    nuclei[10:40, 10:40] = 1  # one nucleus
    target_mask = np.zeros((64, 64), dtype=bool)
    target_mask[15, 15] = target_mask[30, 30] = True

    data = compute_entropy_per_nucleus(nuclei, target_mask)
    assert len(data) == 1
    assert data[0]["Nucleus_Label"] == 1
    assert "target_entropy" in data[0]


def test_pipeline_config_defaults():
    from amplifish import PipelineConfig

    cfg = PipelineConfig()
    assert cfg.save_segment is True
    assert cfg.verbose is False
