# Running the demo

The repository ships two real Colo320 FISH images (1 ecDNA + 1 HSR) so the full
pipeline can be run out of the box.

## Instructions to run

```bash
cd demo
bash demo.sh            # or: python run_demo.py
# Force CPU if you have no GPU:
python run_demo.py --cpu
```

The pretrained CellposeSAM (`cpsam`) weights are **not bundled** with this
repository — cellpose downloads them (~1.2 GB) on the first run and caches them
under `~/.cellpose/models/`, so the first run needs internet access. Subsequent
runs load from the cache offline. No fine-tuning is required.

## Expected output

Results are written to `demo/output/`:

- `tables/per_nucleus_features.csv` — one row per nucleus with `n_target`,
  `n_centromere`, `target_entropy`, `area`, and the amplification `type`.
- `tables/per_image_summary.csv` — per-image aggregates.
- `tables/{nuclei,target,centromere,entropy}.csv` — full measurement tables.
- `images/segment/` — segmentation overlays (nuclei + spots on each image).
- `images/preview/` — stretch-only display previews (no top-hat) of each image,
  for quick visual inspection.
- `figures/amplification_type_separation.png` — boxplots of the two most
  discriminating per-nucleus features (with the unsupervised KMeans type-recovery
  rate in the title).

The per-image summary should match (small numerical variation is normal):

| image | type | n_nuclei | mean target / nucleus | mean centromere / nucleus | mean target entropy |
| --- | --- | --- | --- | --- | --- |
| Colo320_DM_0005 | ecDNA | 13 | 22.1 | 2.5 | 3.86 |
| Colo320_HSR_0004 | HSR | 13 | 7.7 | 3.0 | 2.65 |

ecDNA nuclei carry **more** target spots and **higher** spatial entropy than
HSR nuclei; unsupervised KMeans on the per-nucleus features recovers the known
amplification type at ~81%. A committed reference copy of the summary table and
figure lives in `demo/expected_output/` for comparison.

## Expected run time

About **2 minutes** end-to-end on a GPU workstation (including model load).

On CPU the run is dominated by nucleus segmentation: about **12 minutes** of
compute (~6 minutes per image), measured on an Apple Silicon Mac (macOS 13,
`--cpu`). The first run additionally downloads the ~1.2 GB CellposeSAM weights
once (~1 minute on a fast connection, longer on a slow one), for roughly
**13 minutes** end-to-end.
