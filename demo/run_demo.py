#!/usr/bin/env python
"""
AMPLIFISH end-to-end demo.

Runs the full AMPLIFISH pipeline on the two bundled Colo320 FISH images
(1 ecDNA + 1 HSR) and writes per-nucleus feature tables plus a figure showing
that the extracted features separate the two amplification types.

Pipeline (per image)
---------------------
1. Image correction      — percentile stretch + white-top-hat (amplifish.utils.image)
2. Nucleus segmentation  — CellposeSAM on the DAPI channel (pretrained; cellpose
   downloads the weights once on first run, then caches them)
3. Target spot detection — TxRed channel  (amplifish.segmentation.spots)
4. Centromere detection  — FITC channel
5. Measurement + entropy — per-nucleus regionprops, spot assignment, Shannon
                           entropy of target-spot spatial distribution
6. Feature analysis      — aggregate per-nucleus features, cluster, and plot
                           ecDNA vs HSR separation

Usage
-----
    python run_demo.py                 # GPU if available
    python run_demo.py --cpu           # force CPU
    python run_demo.py --outdir /tmp/amplifish_demo

Outputs land in ``demo/output/`` by default:
    tables/    nuclei.csv, target.csv, centromere.csv, entropy.csv,
               per_nucleus_features.csv, per_image_summary.csv
    images/    segmentation overlays (segment/), corrected RGB inputs
               (corrected/), display previews (preview/, stretch only)
    figures/   amplification_type_separation.png
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from skimage import io

import amplifish
from amplifish.utils.image import calibrate_hi_floors, correct_image, stretch_channel
from amplifish.segmentation import nuclei as cp_seg
from amplifish.segmentation.spots import detect_target_spots, detect_centromere_spots
from amplifish import process_precomputed_otsu, PipelineConfig

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))

# Stretch / detection parameters (match the cell-line pipeline).
P_LOW, P_HIGH = 1.0, 99.9
TARGET_RADIUS, CENTROMERE_RADIUS = 20, 10
THRESHOLD_METHOD = "yen"

log = logging.getLogger("amplifish.demo")


def channel_paths(tiff_prefix):
    """Return (txred, fitc, dapi) absolute paths for one image prefix."""
    prefix = os.path.join(DEMO_DIR, tiff_prefix)
    return f"{prefix}_txred.tiff", f"{prefix}_fitc.tiff", f"{prefix}_dapi.tiff"


def main():
    parser = argparse.ArgumentParser(description="Run the AMPLIFISH demo.")
    parser.add_argument("--metadata", default=os.path.join(DEMO_DIR, "data", "metadata_demo.csv"))
    parser.add_argument("--outdir", default=os.path.join(DEMO_DIR, "output"))
    parser.add_argument("--cpu", action="store_true", help="Force CPU (no GPU).")
    args = parser.parse_args()

    amplifish.setup_logging(level=logging.INFO)

    # ── Output layout ─────────────────────────────────────────────────────────
    nuclei_dir    = os.path.join(args.outdir, "01_nuclei")
    target_dir    = os.path.join(args.outdir, "02_target")
    centromere_dir = os.path.join(args.outdir, "02_centromere")
    corrected_dir = os.path.join(args.outdir, "images", "corrected")
    preview_dir   = os.path.join(args.outdir, "images", "preview")
    images_dir    = os.path.join(args.outdir, "images")
    tables_dir    = os.path.join(args.outdir, "tables")
    figures_dir   = os.path.join(args.outdir, "figures")
    for d in (nuclei_dir, target_dir, centromere_dir, corrected_dir, preview_dir, images_dir, tables_dir, figures_dir):
        os.makedirs(d, exist_ok=True)

    meta = pd.read_csv(args.metadata)
    meta["image_name"] = meta["image_id"]
    rows = meta.to_dict("records")
    log.info("Loaded %d images from %s", len(rows), args.metadata)

    # ── Step 0: calibrate global hi_floors across all demo images ─────────────
    # Prevents dim images from being over-stretched. Done once over the batch.
    r_imgs, g_imgs, b_imgs = [], [], []
    for row in rows:
        txred, fitc, dapi = channel_paths(row["tiff_path"])
        r_imgs.append(io.imread(txred))
        g_imgs.append(io.imread(fitc))
        b_imgs.append(io.imread(dapi))
    hi_floors = calibrate_hi_floors(
        {"R": r_imgs, "G": g_imgs, "B": b_imgs}, p_high=P_HIGH, floor_percentile=10
    )

    # ── Step 1-4: correction, segmentation, spot detection ────────────────────
    log.info("Loading pretrained CellposeSAM model (gpu=%s)...", not args.cpu)
    model = cp_seg.load_model(gpu=not args.cpu)

    for row, R, G, B in zip(rows, r_imgs, g_imgs, b_imgs):
        name = row["image_name"]
        log.info("[%s] correcting + segmenting...", name)

        # 1. Correction: save corrected RGB (used as overlay background later)
        #    plus a stretch-only display preview (no top-hat) for human viewing.
        display, corrected = correct_image(
            R, G, B, hi_floors=hi_floors, p_low=P_LOW, p_high=P_HIGH,
            target_radius=TARGET_RADIUS, centromere_radius=CENTROMERE_RADIUS,
        )
        io.imsave(os.path.join(corrected_dir, f"{name}_corrected.tiff"),
                  corrected, check_contrast=False)
        io.imsave(os.path.join(preview_dir, f"{name}_preview.png"),
                  display, check_contrast=False)

        # 2. Nuclei: cap bright outliers at the calibrated floor, then segment.
        hi = max(float(np.percentile(B, P_HIGH)), hi_floors["B"])
        nuclei = cp_seg.segment_nuclei(np.clip(B, None, hi), model)
        io.imsave(os.path.join(nuclei_dir, f"{name}_nuclei_labels.tiff"),
                  nuclei.astype(np.uint16), check_contrast=False)

        # 3. Target spots (TxRed) — stretch then detect.
        R_s = stretch_channel(R, P_LOW, P_HIGH, hi_floor=hi_floors["R"])
        target = detect_target_spots(R_s, threshold_method=THRESHOLD_METHOD)
        io.imsave(os.path.join(target_dir, f"{name}_target_labels.tiff"),
                  target.astype(np.uint16), check_contrast=False)

        # 4. Centromere spots (FITC) — stretch then detect.
        G_s = stretch_channel(G, P_LOW, P_HIGH, hi_floor=hi_floors["G"])
        centromere = detect_centromere_spots(G_s, threshold_method=THRESHOLD_METHOD)
        io.imsave(os.path.join(centromere_dir, f"{name}_centromere_labels.tiff"),
                  centromere.astype(np.uint16), check_contrast=False)

        log.info("[%s] %d nuclei, %d target, %d centromere",
                 name, int(nuclei.max()), int(target.max()), int(centromere.max()))

    # ── Step 5: measurement + entropy (downstream pipeline) ───────────────────
    cfg = PipelineConfig(
        verbose=False, save_bin_heatmap=False, save_entropy_heatmap=False,
        save_original=False, save_segment=True, save_debug_tiff=False,
    )
    nuclei_tabs, target_tabs, centromere_tabs, entropy_tabs = [], [], [], []
    for row in rows:
        r = dict(row)
        r["file_path"] = os.path.join(corrected_dir, f"{row['image_name']}_corrected.tiff")
        # process_precomputed_otsu tags output rows with patient_id/sample_id.
        # These are optional in the CSV; default to the image name and type.
        r["patient"] = row.get("patient", row["image_name"])
        r["sample"] = row.get("sample", row["type"])
        result = process_precomputed_otsu(
            r, nuclei_dir, target_dir, centromere_dir, images_dir, config=cfg
        )
        if result is None:
            log.warning("[%s] assembly failed — skipping", row["image_name"])
            continue
        nuclei_tabs.append(result.nuclei)
        target_tabs.append(result.target)
        centromere_tabs.append(result.centromere)
        entropy_tabs.append(result.entropy)

    nuclei     = pd.concat(nuclei_tabs, ignore_index=True)
    target     = pd.concat(target_tabs, ignore_index=True)
    centromere = pd.concat(centromere_tabs, ignore_index=True)
    entropy    = pd.concat(entropy_tabs, ignore_index=True)

    nuclei.to_csv(os.path.join(tables_dir, "nuclei.csv"), index=False)
    target.to_csv(os.path.join(tables_dir, "target.csv"), index=False)
    centromere.to_csv(os.path.join(tables_dir, "centromere.csv"), index=False)
    entropy.to_csv(os.path.join(tables_dir, "entropy.csv"), index=False)

    # ── Step 6: per-nucleus feature table ─────────────────────────────────────
    # Spots carry a Nucleus_Label parent; NaN = not inside any nucleus (dropped).
    n_target = (target.dropna(subset=["Nucleus_Label"])
                .groupby(["image_name", "Nucleus_Label"]).size().rename("n_target"))
    n_centromere = (centromere.dropna(subset=["Nucleus_Label"])
                    .groupby(["image_name", "Nucleus_Label"]).size().rename("n_centromere"))
    ent = entropy[["image_name", "Nucleus_Label", "target_entropy"]]

    feat = (nuclei
            .merge(n_target, on=["image_name", "Nucleus_Label"], how="left")
            .merge(n_centromere, on=["image_name", "Nucleus_Label"], how="left")
            .merge(ent, on=["image_name", "Nucleus_Label"], how="left"))
    feat[["n_target", "n_centromere"]] = feat[["n_target", "n_centromere"]].fillna(0)
    feat["target_entropy"] = feat["target_entropy"].fillna(0.0)
    feat.to_csv(os.path.join(tables_dir, "per_nucleus_features.csv"), index=False)

    # ── Per-image summary ─────────────────────────────────────────────────────
    summary = (feat.groupby(["image_name", "type"])
               .agg(n_nuclei=("Nucleus_Label", "size"),
                    mean_target_per_nucleus=("n_target", "mean"),
                    mean_centromere_per_nucleus=("n_centromere", "mean"),
                    mean_target_entropy=("target_entropy", "mean"))
               .round(3).reset_index())
    summary.to_csv(os.path.join(tables_dir, "per_image_summary.csv"), index=False)
    log.info("Per-image summary:\n%s", summary.to_string(index=False))

    # ── Step 6b: feature separation figure (ecDNA vs HSR) ─────────────────────
    make_figure(feat, os.path.join(figures_dir, "amplification_type_separation.png"))

    print("\nDemo complete. Outputs written to:", args.outdir)
    print("  tables/per_image_summary.csv")
    print("  figures/amplification_type_separation.png")


def make_figure(feat, out_path):
    """Feature boxplots showing ecDNA vs HSR separation.

    Also runs unsupervised KMeans (k=2) on the standardised features and reports
    how well the clusters recover the known amplification types.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    features = ["n_target", "n_centromere", "target_entropy", "area"]
    df = feat.dropna(subset=features).copy()
    X = StandardScaler().fit_transform(df[features].to_numpy())

    df["cluster"] = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)

    # Cluster ↔ type agreement (max over the two label-to-cluster assignments).
    ct = pd.crosstab(df["cluster"], df["type"])
    agree = (ct.max(axis=1).sum()) / len(df)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    data = [df.loc[df["type"] == t, "n_target"] for t in ["ecDNA", "HSR"]]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks([1, 2], ["ecDNA", "HSR"])
    ax.set(ylabel="target spots per nucleus", title="Target spot count")

    ax = axes[1]
    data = [df.loc[df["type"] == t, "target_entropy"] for t in ["ecDNA", "HSR"]]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks([1, 2], ["ecDNA", "HSR"])
    ax.set(ylabel="target-spot Shannon entropy", title="Spatial entropy")

    fig.suptitle(
        "AMPLIFISH: per-nucleus FISH features separate ecDNA from HSR "
        f"(KMeans recovers type: {agree:.0%})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Saved figure: %s (KMeans–type agreement %.0f%%)", out_path, agree * 100)


if __name__ == "__main__":
    sys.exit(main())
