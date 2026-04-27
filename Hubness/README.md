# Hubness Analyses

This folder contains a hubness workflow with the goals to:
- Analyze connectivity on a network and parcellated network level
- Use 360 Glasser parcels, overlap them with individual networks and build new split parcels
- Analyze parcel-level resting-state FC to find general hubs (calculate resting-state participation coefficient).
- Analyze parcel-level task-based PPI and GVC for flexible coupling analyses.
- Illustrate static and flexible connectivity in graph-theoretical networks (e.g. spring-embedded)

**Key Design Decisions**:
- SLURM: every script should run on its own but using SLURM (parralel subjects) and inside a container

- Split method: Vertex overlap thresholding (simple, interpretable, respects misalignment, but makes networks smaller)
  - Aggregation fallback: Max-overlap hard assignment (simple, deterministic)
- GVC: Mean variability across all parcel connections (captures global connectivity volatility)
- Task PC: Computed on variability matrix (measures task-modulation distribution)
- Get inspiration from e.g. Cole et al 2013 (Flexible Hub) and 

## Design Principles

- Keep shared and reusable logic in `hubness_utils.py`.
- Support both analysis levels where relevant:
  - `network_parcel` level (split parcels; n~450, threshold-dependent)
  - `network` level (canonical networks; n~20)
- Support two FPN handling modes:
  - unified FPN (default)
  - split FPNA/FPNB using subject-specific derivations from `2025_ibc_latent/outputs/subnetworks/subnetwork_derivation/infomap/sub-xx/xx_FPN_infomap_communities_kmeans_relabeled.dlabel.nii`
- Use Infomap label-table colors from `Bipartite_PhysicalCommunities+AlgorithmicLabeling.dlabel.nii` for plots/dlabels; only override FPNA/FPNB with teal/blue in split mode.
- Keep per-subject processing modular, with optional group aggregation as a separate post-processing step.


## Scripts and workflow

1) `define_networks.py`
  - Inputs:   
  - Splits each Glasser parcel into retained network-overlap fragments using a configurable parcel-fraction threshold.
  - Writes per-subject split masks plus a split manifest, and keeps a hard parcel summary for legacy consumers.

2) `static_hubs.py`
  - Computes subject-level static parcel FC (360x360), participation coefficient, and strength.
  - In `network` mode, also creates a circular/star network plot using collapsed network FC.
  - In `network_parcel` mode, keeps spring-embedded split-parcel visualization.

3) `static_hubs_group.py`
  - Aggregates per-subject collapsed network FC matrices (`subject_fc_network_collapsed.npz`) across subjects.
  - Writes group mean/std FC and a group circular/star network plot.

4) `Flexibility/flexible_hub_ppi.py`
  - Stage 1 of the flexibility pipeline.
  - Computes and saves per-subject task PPI outputs once, without plotting.

5) `Flexibility/flexible_hub.py`
  - Stage 2 of the flexibility pipeline.
  - Loads saved PPI outputs, computes flexibility metrics (GVC, variability participation, network-to-network variability), and renders circular/spring plots.

6) `Flexibility/flexible_hub_group.py`
  - Aggregates subject-level flexibility outputs and writes group variability summaries and plots.


### Example usage

Step 1 with SLURM array (one subject per task):
```bash
sbatch Hubness/define_networks_SLURM.sh
```

Step 1 directly (unified FPN, default):
```bash
python Hubness/define_networks.py --subjects 01 02 --fpn-mode unified
```

Step 1 directly (split FPNA/FPNB):
```bash
python Hubness/define_networks.py --subjects 01 02 --fpn-mode split
```

Step 1 with SLURM using split FPN mode:
```bash
FPN_MODE=split sbatch Hubness/define_networks_SLURM.sh
```

Step 2 with SLURM array:
```bash
sbatch Hubness/static_hubs_SLURM.sh
```

Step 2 directly (network level):
```bash
python Hubness/static_hubs.py --subjects 01 02 --analysis-level network --fpn-mode unified
```

Step 2 directly (split network-parcel level):
```bash
python Hubness/static_hubs.py --subjects 01 02 --analysis-level network_parcel --fpn-mode split --overlap-threshold 0.30
```

Step 2 with SLURM using split FPN mode:
```bash
FPN_MODE=split sbatch Hubness/static_hubs_SLURM.sh
```

Step 3 directly (group aggregation and circular plot):
```bash
python Hubness/static_hubs_group.py --output-dir /ptmp/hmueller2/2025_ibc_latent/outputs/hubness
```

Step 3 with SLURM (group-only):
```bash
sbatch Hubness/static_hubs_group_SLURM.sh
```

Flexibility pipeline, stage 1 (compute and save PPI):
```bash
sbatch Hubness/Flexibility/flexible_hub_ppi_SLURM.sh
```

Flexibility pipeline, stage 2 (metrics and plots from saved PPI):
```bash
sbatch Hubness/Flexibility/flexible_hub_SLURM.sh
```

Flexibility pipeline, group aggregation:
```bash
sbatch Hubness/Flexibility/flexible_hub_group_SLURM.sh
```

## Outputs

### Network Assignments

- `sub-XX/parcel_network_assignment_subject.csv`
  - for all 360 parcels, mapping to which top 1 network they belong to + assignment_fraction (% overlap)
  - could be used for a hard assignment if not wanting to use parcel_slit method
- `sub-XX/parcel_split_manifest_subject_t0%%.csv`
  - for all 360 parcels, all overlapping networks,, overlapping fracition, vertex counts, and if retained or not (depending on threshold -> in name, e.g. _t030)
- `sub-XX/split_parcels_t0%%/*.dscalar.nii`
  - folders with all unique cut out parcels saved as dscalar.nii
- `sub-XX/sub-XX_split_parcels_combined_t0%%_parcel_colored.dlabel`
  - assembly of all spilt parcels, with every parcel colored uniquely
- `sub-XX/sub-XX_split_parcels_combined_t0%%_network_colored.dlabel`
  - assembly of all spilt parcels, colored according to the networks they belong to

### Static Hubness (Resting-State)

- `sub-XX/static/static_hub_metrics_subject.csv`
  - per-subject parcel-level static hubness metrics
- `sub-XX/static/subject_fc_360x360.npz`
  - per-subject parcel FC matrix
- `sub-XX/static/subject_fc_network_collapsed.npz`
  - per-subject network FC matrix (network x network), when `--save-network-fc` is used
- `sub-XX/static/circular_plot_network_edgesXXX_<metric>.png`
  - network-level circular/star visualization (center node = highest absolute network strength)
- `sub-XX/static/spring_plot_network_parcel_edgesXXX_nodeYYY_<metric>.png`
  - split-parcel spring visualization in `network_parcel` mode

### Group Static Hubness (Post-processing)

- `group/static/group_fc_network_collapsed.npz`
  - group mean/std network FC matrix and subject list
- `group/static/group_network_strength_summary.csv`
  - absolute network strength ranking from group mean FC
- `group/static/circular_plot_group_network_edgesXXX_<metric>.png`
  - group-level circular/star network visualization


### Dependencies

**Required** (standard):
- matplotlib, numpy, pandas, seaborn, (networkx)
