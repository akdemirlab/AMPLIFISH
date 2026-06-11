# Using AMPLIFISH on your own data

AMPLIFISH expects, per image, three single-channel TIFFs:

```
<prefix>_txred.tiff   # red   — target gene (e.g. MYC)
<prefix>_fitc.tiff    # green — centromere probe
<prefix>_dapi.tiff    # blue  — DAPI nuclei
```

List your images in a metadata CSV (see `demo/data/metadata_demo.csv`):

| column | meaning |
| --- | --- |
| `image_id` | unique image name |
| `type` | amplification label (e.g. `ecDNA`, `HSR`, or `Unknown`) |
| `tiff_path` | path prefix to the three channel TIFFs (no `_dapi.tiff` suffix) |
| `patient`, `sample` | *optional* grouping labels carried through to the tables (default to `image_id` / `type`) |

Then run the demo script against your CSV:

```bash
python demo/run_demo.py --metadata /path/to/your_metadata.csv --outdir /path/to/output
```

Or call the library directly:

```python
import amplifish
from amplifish.utils.image import stretch_channel
from amplifish.segmentation import nuclei as cp_seg
from amplifish.segmentation.spots import detect_target_spots

amplifish.setup_logging()
model = cp_seg.load_model(gpu=True)          # pretrained CellposeSAM
nuclei = cp_seg.segment_nuclei(dapi_image, model)
target = detect_target_spots(stretch_channel(txred_image), threshold_method="yen")
```

See `demo/run_demo.py` for the full, documented workflow.
