# AMPLIFISH documentation

AMPLIFISH turns multi-channel FISH (fluorescence in-situ hybridisation)
microscopy images into per-nucleus feature tables — target-gene spot counts,
centromere spot counts, nucleus shape, and the Shannon entropy of the
target-spot spatial distribution — that distinguish oncogene amplification
patterns such as **ecDNA** (extrachromosomal DNA; scattered double-minutes)
from **HSR** (homogeneously staining regions; a single clustered locus).

The pipeline is unsupervised: it extracts interpretable features per nucleus,
and those features separate the amplification types. No labelled training data
or model fine-tuning is required.

## Contents

- [Installation & system requirements](installation.md)
- [Running the demo](demo.md)
- [Using AMPLIFISH on your own data](usage.md)

## Repository layout

```
amplifish/
├── src/amplifish/            # the library (pip-installable)
│   ├── segmentation/         # nucleus (CellposeSAM) + FISH spot detection
│   ├── measurement/          # spot-to-nucleus assignment, regionprops, Shannon entropy
│   ├── utils/image.py        # channel split + correction (stretch + white-top-hat + hi_floor)
│   ├── pipeline.py           # process_precomputed_otsu (assemble per-nucleus tables)
│   └── ...
├── demo/
│   ├── data/                 # 2 example FISH images (1 ecDNA + 1 HSR) + metadata
│   ├── run_demo.py           # one-command end-to-end demo
│   ├── demo.sh               # thin wrapper around run_demo.py
│   └── expected_output/      # reference tables + figure to check your run against
├── docs/                     # this documentation
├── tests/                    # smoke tests (pytest)
├── pyproject.toml
└── LICENSE                   # MIT, Akdemir Lab
```
