# AMPLIFISH

**FISH image analysis for oncogene amplification typing (ecDNA vs HSR).**

AMPLIFISH turns multi-channel FISH microscopy images into per-nucleus feature
tables — target/centromere spot counts, nucleus shape, and the Shannon entropy
of the target-spot spatial distribution — that separate **ecDNA**
(extrachromosomal DNA) from **HSR** (homogeneously staining regions). The
pipeline is unsupervised: no labelled training data or fine-tuning required.

## Documentation

Full documentation lives in [docs/](docs/index.md), including a guide to
[using AMPLIFISH on your own data](docs/usage.md).

## Installation

```bash
git clone https://github.com/akdemirlab/AMPLIFISH.git
cd AMPLIFISH
conda create -n amplifish python=3.9 -y && conda activate amplifish
pip install -e .
```

Dependencies (including the CellposeSAM segmentation backend) install
automatically. See [docs/installation.md](docs/installation.md) for system
requirements and details.

## Demo

```bash
cd demo && bash demo.sh        # or: python run_demo.py --cpu
```

Runs the full pipeline on two bundled Colo320 FISH images (1 ecDNA + 1 HSR) and
writes per-nucleus feature tables plus a figure separating the two amplification
types. See [docs/demo.md](docs/demo.md) for expected output and run time.

## Checklist

Pointers to where each required item lives:

- **Source code:** [src/amplifish/](src/amplifish/)
- **Demo dataset:** [demo/data/raw/](demo/data/raw/) — 2 real Colo320 FISH images (1 ecDNA + 1 HSR)
- **System requirements** (dependencies + OS + version numbers, versions tested, non-standard hardware): [docs/installation.md](docs/installation.md)
- **Installation guide** (instructions + typical install time): [docs/installation.md](docs/installation.md)
- **Demo** (instructions to run, expected output, expected run time): [docs/demo.md](docs/demo.md), with reference results in [demo/expected_output/](demo/expected_output/)
- **Instructions for use** (run on your own data): [docs/usage.md](docs/usage.md)

## License & citation

MIT License, Copyright (c) Akdemir Lab. See [LICENSE](LICENSE).

If you use AMPLIFISH in your research, please cite the accompanying manuscript
(citation to be added on publication).
