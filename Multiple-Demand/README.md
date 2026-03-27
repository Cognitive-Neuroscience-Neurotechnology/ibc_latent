# Multiple-Demand Analysis: MD Mapping, MD-vs-FPN, and Easy/Hard Coupling

This folder contains a compact three-stage workflow:

1. Build subject and group MD maps from difficulty contrasts.
2. Compare MD maps against subject-specific FPN_A/FPN_B subnetworks.
3. Quantify Easy/Hard activation and condition-specific coupling to DMN/DAN.

The goal is to quantify both:

- where MD overlaps with FPN subnetworks (binary overlap), and
- how strong MD activation is inside those subnetworks (value-based maps and summary stats).
- how strongly FPN/FPNA/FPNB couple to DMN and DAN during hard vs easy blocks.

## Code Overview

### 1) md_mapping.py

Builds MD maps from predefined hard-versus-easy contrasts.

- Input: fixed-effects contrast maps in fsLR CIFTI space.
- Output: subject MD mean maps, optional thresholded maps, and optional group map.
- Supports geodesic smoothing via Workbench.

Typical outputs:

- sub-XX_MD_mean.dscalar.nii
- sub-XX_MD_mean_top20pct.dscalar.nii (or z-thresholded variant)
- group/group_MD_mean.dscalar.nii (if group enabled)

### 2) md_vs_fpn.py

Compares MD maps to FPN_A/FPN_B labels from Infomap relabeled files.

Approach A (threshold overlap):

- Threshold MD map and measure overlap with FPN_A, FPN_B, and FPN_A|B.
- Saves overlap CSV metrics and overlap dscalar maps.

Approach B (MD values in masks):

- Summarizes raw MD values within FPN_A/FPN_B/FPN_A|B.
- Saves dscalar maps with MD values inside masks and zero elsewhere.

Typical outputs:

- individual_subjects.csv
- group_vs_individual_masks.csv
- group_vs_consensus_masks.csv
- overlap_dscalars/
- approach_b_md_value_maps/

### 3) md_easy_hard_fpn_coupling.py

Computes Easy/Hard metrics for the four MD tasks (HcpWm, Stroop, Catell, Attention).

- Activation: mean values in FPN, FPN_A, FPN_B for easy, hard, and hard-minus-easy.
- Coupling: condition-restricted correlations between FPN seeds (FPN/FPN_A/FPN_B) and targets (DMN/DAN).
- Uses task z-maps, task dtseries, events TSV files, relabeled FPN masks, and resting-state network labels.

Typical outputs:

- md_easy_hard_fpn_coupling/per_subject_results.csv
- md_easy_hard_fpn_coupling/group_summary.csv

## Minimal Workflow

1. Run MD mapping to create MD maps.
2. Run MD-vs-FPN comparison using the MD maps and relabeled FPN maps.
3. Run Easy/Hard activation + coupling analysis.
4. Visualize dscalar outputs in Workbench and use CSVs for statistics.

## Direct Python Usage

### Step 1: MD mapping

```bash
python md_mapping.py \
    --all-subjects \
    --group \
    --contrast-base /ptmp/hmueller2/2025_ibc_latent/outputs/glm/contrast_maps_fsLR \
    --output /ptmp/hmueller2/2025_ibc_latent/outputs/md_system/vertex_wise \
    --smooth 4 \
    --threshold-percent 20
```

### Step 2: MD vs FPN

```bash
python md_vs_fpn.py \
    --all-subjects \
    --md-dir /ptmp/hmueller2/2025_ibc_latent/outputs/md_system/vertex_wise \
    --subnetwork-dir /ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_derivation/infomap \
    --output /ptmp/hmueller2/2025_ibc_latent/outputs/md_system/vertex_wise/md_vs_fpn \
    --threshold-percent 20
```

## SLURM Usage

### MD mapping job

```bash
sbatch md_mapping_SLURM.sh
```

Useful environment overrides:

- MODE=vertex|parcels|both
- RUN_GROUP=1 or RUN_GROUP=0
- THRESHOLD_PERCENT=20
- SMOOTH_FWHM=4

Example:

```bash
MODE=both RUN_GROUP=1 THRESHOLD_PERCENT=20 sbatch md_mapping_SLURM.sh
```

### MD-vs-FPN job

```bash
sbatch md_vs_fpn_SLURM.sh
```

Optional override:

```bash
THRESHOLD_PERCENT=20 sbatch md_vs_fpn_SLURM.sh
```

### Easy/Hard coupling job

```bash
sbatch md_easy_hard_fpn_coupling_SLURM.sh
```

Optional overrides:

- K_INDEX=0 (row index in relabeled FPN file)
- STRICT=1 (fail fast on missing subject/task inputs)

Example:

```bash
K_INDEX=0 STRICT=0 sbatch md_easy_hard_fpn_coupling_SLURM.sh
```

### Monitor jobs

```bash
squeue -u $USER
tail -f logs/*.out
```

## Input Expectations

- MD mapping expects fixed-effects z-score contrast maps under task-specific fsLR folders.
- MD-vs-FPN expects:
  - MD maps at md-dir/sub-XX/sub-XX_MD_mean.dscalar.nii
  - FPN labels at subnetwork-dir/sub-XX/XX_FPN_infomap_communities_kmeans_relabeled.dlabel.nii

## Visualization

```bash
wb_view /path/to/file.dscalar.nii
```

Recommended checks:

1. MD mean and thresholded maps from md_mapping.
2. overlap_dscalars maps for Approach A.
3. approach_b_md_value_maps for Approach B value-preserving interpretation.
