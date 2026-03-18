# Multiple Demand (MD) System Mapping

This directory contains scripts for mapping the Multiple Demand system in individual subjects using difficulty-based task contrasts.

## Overview

The Multiple Demand (MD) system is a domain-general network that activates across various cognitively demanding tasks. This script identifies the MD system by averaging z-score maps from contrasts that manipulate task difficulty.

**Two analysis approaches:**

1. **Vertex-wise analysis** (`md_mapping.py`): Analyzes ~59k cortical surface vertices individually
2. **Parcel-based analysis** (`md_mapping_parcels.py`): Uses HCP-MMP1 360 cortical parcels for region-level analysis

Both approaches follow methods similar to Assem et al. (2020) "A Domain-General Cognitive Core Defined in Multimodally Parcellated Human Cortex."

## Contrasts Used

The script uses the following difficulty-based contrasts:

- **HcpWm**: `2back-0back`
- **ItemRecognition**: `encode5-encode1`
- **MDTB**: `2back_hard-easy`, `search_hard-easy`, `semantic_hard-easy`, `finger_complex-simple`
- **GoodBadUgly**: `dot_hard-easy`
- **Stroop**: `incongruent-congruent`
- **Catell**: `hard-easy`

## Requirements

```bash
pip install numpy nibabel pandas
```

## Key Features

### Smoothing
- Surface-based Gaussian smoothing (default: 4mm FWHM)
- Recommended for reducing noise while preserving boundaries
- Can be disabled with `--smooth 0`

### Thresholding
- Statistical thresholding to identify reliable MD regions
- Example thresholds:
  - `--threshold 2.3`: p < 0.01 (one-tailed)  
  - `--threshold 3.1`: p < 0.001 (one-tailed)
- Saves both thresholded and unthresholded maps

### Parcellation
- Use `md_mapping_parcels.py` for HCP-MMP1 360-parcel analysis
- Provides region-level MD scores for easier interpretation
- Outputs CSV files with parcel names and z-scores

## Usage

### SLURM Cluster (Recommended)

For running on HPC systems with SLURM:

#### Option 1: Single Job (All subjects)
```bash
# Edit paths in md_mapping_SLURM.sh if needed, then:
sbatch md_mapping_SLURM.sh
```

#### Option 2: Array Job (Parallel processing)
```bash
# Process all subjects in parallel (faster):
sbatch md_mapping_SLURM_array.sh

# After array jobs complete, compute group map:
sbatch md_mapping_SLURM_group.sh
```

#### Monitor jobs
```bash
# Check job status
squeue -u $USER

# View output
tail -f logs/md_mapping_*.out
```

### Direct Python Execution

For running directly (e.g., on login nodes for testing, not recommended for full analysis):

#### Process a single subject

```bash
python md_mapping.py \
    --subject 01 \
    --contrast-base /path/to/contrast_maps_fsLR \
    --output /path/to/output/md_maps
```

### Process multiple subjects

```bash
python md_mapping.py \
    --subjects 01 02 04 05 06 07 \
    --contrast-base /path/to/contrast_maps_fsLR \
    --output /path/to/output/md_maps
```

### Process all available subjects

```bash
python md_mapping.py \
    --all-subjects \
    --contrast-base /path/to/contrast_maps_fsLR \
    --output /path/to/output/md_maps
```

### Compute group-level MD map

```bash
python md_mapping.py \
    --all-subjects \
    --group \
    --contrast-base /path/to/contrast_maps_fsLR \
    --output /path/to/output/md_maps
```

### With thresholding (recommended for identifying MD regions)

```bash
python md_mapping.py \
    --all-subjects \
    --group \
    --threshold 2.3 \
    --contrast-base /path/to/contrast_maps_fsLR \
    --output /path/to/output/md_maps
```

### Parcel-based analysis (360 cortical regions)

```bash
# Individual subjects
python md_mapping_parcels.py \
    --all-subjects \
    --contrast-base /path/to/contrast_maps_fsLR \
    --output /path/to/output/md_maps_parcels

# With group analysis
python md_mapping_parcels.py \
    --all-subjects \
    --group \
    --contrast-base /path/to/contrast_maps_fsLR \
    --output /path/to/output/md_maps_parcels
```

## Output Structure

### Vertex-wise Analysis (`md_mapping.py`)

```
output_dir/
├── sub-01/
│   ├── sub-01_MD_mean.dscalar.nii          # Mean MD map (average z-scores)
│   ├── sub-01_MD_mean_thresh_2.3.dscalar.nii  # Thresholded map (if --threshold used)
│   ├── sub-01_MD_std.dscalar.nii           # Standard deviation across contrasts
│   ├── sub-01_MD_contrasts.txt             # List of contrasts used
│   └── individual_contrasts/               # Individual contrast contributions
│       ├── HcpWm_2back-0back.dscalar.nii
│       ├── MDTB_2back_hard-easy.dscalar.nii
│       └── ...
├── sub-02/
│   └── ...
└── group/                                   # Group-level maps (if --group is used)
    ├── group_MD_mean.dscalar.nii           # Group mean
    ├── group_MD_mean_thresh_2.3.dscalar.nii  # Thresholded group map
    ├── group_MD_std.dscalar.nii            # Group standard deviation
    ├── group_MD_sem.dscalar.nii            # Group standard error
    ├── group_HcpWm_2back-0back_mean.dscalar.nii   # Group mean for HcpWm contrast
    ├── group_ItemRecognition_encode5-encode1_mean.dscalar.nii  # Group mean for ItemRecognition contrast
    ├── group_Stroop_incongruent-congruent_mean.dscalar.nii     # Group mean for Stroop contrast
    ├── group_Catell_hard-easy_mean.dscalar.nii                 # Group mean for Catell contrast
    └── group_MD_info.txt                   # Subject list and info
```

### Parcel-based Analysis (`md_mapping_parcels.py`)

```
output_dir/
├── sub-01/
│   ├── sub-01_MD_parcels.csv               # Parcel names and z-scores (CSV)
│   ├── sub-01_MD_parcels.npz               # Numpy array format
│   ├── sub-01_MD_parcels_info.txt          # Info about analysis
│   └── individual_contrasts/               # Individual contrast CSVs
│       ├── HcpWm_2back-0back.csv
│       └── ...
├── sub-02/
│   └── ...
└── group/
    ├── group_MD_parcels.csv                # Group parcel z-scores
    ├── group_MD_parcels.npz                # Numpy array format
    └── group_MD_parcels_info.txt           # Top parcels and subject list
```

## Input Data Requirements

The script expects fixed-effects contrast maps in fsLR space with the following directory structure:

```
contrast_base/
└── sub-01/
    └── res_task-HcpWm_space-fsLR_dir-ffx/
        └── z_score_maps/
            ├── 2back-0back.dscalar.nii
            └── ...
```

## Interpreting Results

- **MD mean map**: Higher z-scores indicate stronger and more consistent MD system activation across difficulty contrasts
- **MD std map**: Shows variability across different difficulty manipulations
- Typical MD regions include:
  - Lateral prefrontal cortex
  - Anterior cingulate cortex / pre-SMA
  - Anterior insula / frontal operculum
  - Intraparietal sulcus

## Visualization

You can visualize the output CIFTI files using:

```bash
# Using Connectome Workbench
wb_view sub-01_MD_mean.dscalar.nii

# Or through the GUI
workbench
```

## Notes

- The script automatically handles missing contrasts (e.g., if a subject doesn't have all tasks)
- At least 2 contrasts are required per subject for group analysis
- The script uses fixed-effects z-score maps, which should be computed beforehand using `run_fixed_effects_only.py`

## References

- Duncan, J. (2010). The multiple-demand (MD) system of the primate brain: mental programs for intelligent behaviour. *Trends in Cognitive Sciences*, 14(4), 172-179.
- Fedorenko, E., Duncan, J., & Kanwisher, N. (2013). Broad domain generality in focal regions of frontal and parietal cortex. *PNAS*, 110(41), 16616-16621.
