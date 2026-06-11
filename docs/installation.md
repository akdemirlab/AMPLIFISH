# Installation & system requirements

## System requirements

### Operating systems tested

- Ubuntu 22.04.4 LTS (Linux kernel 6.8.0) — GPU run
- macOS 13.7.8 (Ventura, Apple Silicon / arm64) — CPU-only run (`--cpu`)

The code is pure Python and has no OS-specific dependencies; it is expected to
work on Linux, macOS, and Windows where the dependencies are available.

### Non-standard hardware

- **GPU is optional.** A CUDA GPU accelerates the CellposeSAM nucleus
  segmentation step; the demo was developed on an NVIDIA RTX A6000 (CUDA 12.x).
- AMPLIFISH runs on CPU only (`--cpu`); segmentation is slower but produces
  very similar results. The classical target/centromere spot detection is
  bit-identical on CPU and GPU; only the CellposeSAM nucleus segmentation can
  differ at a borderline nucleus boundary, causing small numerical variation in
  the per-nucleus summaries (see [demo.md](demo.md)). No other special hardware
  is required. ~8 GB RAM is sufficient for the demo. The demo has been run
  end-to-end on CPU (macOS, Apple Silicon) and reproduces the expected ecDNA vs
  HSR separation.

## Installation

```bash
git clone https://github.com/akdemirlab/AMPLIFISH.git
cd AMPLIFISH

# A clean conda environment is recommended (Python >= 3.9).
conda create -n amplifish python=3.9 -y
conda activate amplifish

# Install the library (segmentation backend + demo deps included).
pip install -e .
```

All runtime dependencies are declared in `pyproject.toml` and installed
automatically. CellposeSAM (which pulls in PyTorch transitively) drives nucleus
segmentation and is a core dependency, so a plain install gives a fully working
pipeline and demo — no extras required.

**Typical install time:** about **3–5 minutes** on a normal desktop with a
warm pip cache (longer on first install, dominated by downloading PyTorch).

To verify the install:

```bash
python -c "import amplifish; print(amplifish.__version__)"
```

## Software dependencies

All runtime dependencies are declared in `pyproject.toml` and installed
automatically by `pip` (see [Installation](#installation)). The package was
developed and tested with the versions below.

| Package | Required | Tested |
| --- | --- | --- |
| Python | >=3.9 | 3.9.23 |
| numpy | >=1.22 | 2.0.2 |
| pandas | >=1.4 | 2.3.3 |
| scipy | >=1.8 | 1.13.1 |
| scikit-image | >=0.19 | 0.24.0 |
| matplotlib | >=3.5 | 3.9.4 |
| cellpose | >=4.0 | 4.1.1 (CellposeSAM) |
| csbdeep | (any) | 0.8.2 |
| opencv-python | >=4.5 | 4.13.0 |
| scikit-learn | >=1.0 | 1.6.1 |

`cellpose` pulls in **PyTorch** (tested: torch 2.8.0+cu128) and `scikit-image`
pulls in **tifffile** (tested: 2024.8.30) transitively.
