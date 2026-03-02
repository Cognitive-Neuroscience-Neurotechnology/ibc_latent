# Quick Reference: MD System Mapping

## Quick Start

```bash
# 1. Navigate to the Multiple-Demand directory
cd /home/hmueller2/ibc_code/ibc_latent/Multiple-Demand

# 2. Run the analysis for all subjects with group map
python md_mapping.py \
    --all-subjects \
    --group \
    --contrast-base /ptmp/hmueller2/2025_ibc_latent/outputs/glm/contrast_maps_fsLR \
    --output /ptmp/hmueller2/2025_ibc_latent/outputs/md_system

# 3. View results
wb_view /ptmp/hmueller2/2025_ibc_latent/outputs/md_system/group/group_MD_mean.dscalar.nii
```

## Commands Cheat Sheet

### Basic Analysis
```bash
# Single subject
python md_mapping.py --subject 01 --contrast-base <path> --output <path>

# Multiple subjects
python md_mapping.py --subjects 01 02 04 --contrast-base <path> --output <path>

# All subjects
python md_mapping.py --all-subjects --contrast-base <path> --output <path>

# With group analysis
python md_mapping.py --all-subjects --group --contrast-base <path> --output <path>

# Without saving individual contrasts (saves disk space)
python md_mapping.py --all-subjects --no-individual-contrasts --contrast-base <path> --output <path>
```

### Visualization
```bash
# Get summary statistics
python visualize_md_maps.py summarize <path_to_md_map.dscalar.nii>

# Plot histogram
python visualize_md_maps.py histogram <path_to_md_map.dscalar.nii> --output histogram.png

# Compare all subjects
python visualize_md_maps.py compare <md_maps_directory> --output <output_dir>
```

### Using SLURM (Recommended for HPC)
```bash
# Option 1: Process all subjects in a single job
sbatch md_mapping_SLURM.sh
# Option 2: Process subjects in parallel (faster)
sbatch md_mapping_SLURM_array.sh

# After array jobs complete, compute group map:
sbatch md_mapping_SLURM_group.sh

# Check job status
squeue -u $USER

# View logs
tail -f logs/md_mapping_*.out
```

### Using the batch script (for local testing)
```bash
# Edit paths in run_md_mapping.sh first
bash run_md_mapping.sh
```

## Output Files

**Individual Subject:**
- `sub-XX_MD_mean.dscalar.nii` - Main result: average z-score across MD contrasts
- `sub-XX_MD_std.dscalar.nii` - Variability across contrasts
- `sub-XX_MD_contrasts.txt` - List of which contrasts were used
- `individual_contrasts/*.dscalar.nii` - Individual contrast contributions (optional)

**Group:**
- `group_MD_mean.dscalar.nii` - Group average
- `group_MD_std.dscalar.nii` - Group standard deviation
- `group_MD_sem.dscalar.nii` - Group standard error
- `group_MD_info.txt` - Subject list

## Viewing in Workbench

```bash
# View single map
wb_view sub-01_MD_mean.dscalar.nii

# View with threshold
wb_view sub-01_MD_mean.dscalar.nii &
# Then in GUI: File > Open File > load your map
# Adjust threshold in the Layers panel
```

## Common Workflows

### 1. Individual Subject Mapping
```bash
python md_mapping.py \
    --subject 01 \
    --contrast-base /path/to/contrasts \
    --output ./md_results
```

### 2. Group Analysis
```bash
python md_mapping.py \
    --all-subjects \
    --group \
    --contrast-base /path/to/contrasts \
    --output ./md_results

# Then visualize
python visualize_md_maps.py compare ./md_results --output ./figures
```

### 3. Quality Check
```bash
# Check what contrasts are available for each subject
for subject in 01 02 04 05; do
    echo "Subject $subject:"
    cat md_results/sub-${subject}/sub-${subject}_MD_contrasts.txt | grep "Number of contrasts"
done
```

## Expected MD Regions

The Multiple Demand system typically includes:
- **Lateral Prefrontal Cortex** (middle frontal gyrus, inferior frontal sulcus)
- **Anterior Cingulate / Pre-SMA** (medial frontal)
- **Anterior Insula / Frontal Operculum** (lateral frontal-insular junction)
- **Intraparietal Sulcus** (posterior parietal)

## Troubleshooting

**No contrasts found:**
- Check that fixed-effects analysis has been run (`run_fixed_effects_only.py`)
- Verify the contrast base path is correct
- Check that z_score_maps directories exist

**Fewer contrasts than expected:**
- Some subjects may not have completed all tasks
- This is normal and the script handles it automatically

**Need different contrasts:**
- Edit the `MD_CONTRASTS` dictionary at the top of `md_mapping.py`

## Files in this Directory

- `md_mapping.py` - Main analysis script
- `visualize_md_maps.py` - Visualization utilities
- `run_md_mapping.sh` - Batch processing script
- `README.md` - Full documentation
- `QUICK_REFERENCE.md` - This file
