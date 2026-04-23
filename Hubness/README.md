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

## STILL TO CHANGE: 

- in general: try to keep as many functions reusable and put them into `hubness_utils.py`
- for each step of the pipeline, make sure if to do on network_parcel (n=~450) or network (n=20) level and make sure to accept the two types of inputs.
- include 2 possible ways of handling FPN (frontoparietal), either as one network or by separating into FPNA and FPNB
  - for the separation use 
- for plots, always use the network coloring from infomap outputs (`Bipartite_PhysicalCommunities+AlgorithmicLabeling.dlabel.nii`), except when separating FPN into FPNA and FPNB (for those 2 you can use teal and blue).
- Find out if using GROUP makes sense here. Otherwise get rid of "SKIP_GROUP" etc. and shorten code to stay on individual level for now


## Scripts and workflow

1) `define_networks.py`
  - Inputs:   
  - Splits each Glasser parcel into retained network-overlap fragments using a configurable parcel-fraction threshold.
  - Writes per-subject split masks plus a split manifest, and keeps a hard parcel summary for legacy consumers.

2) `static_hubs.py`
  - Computes subject-level static parcel FC (360x360), participation coefficient, and strength.


### Example usage

Step 1 with SLURM array (one subject per task):
```bash
sbatch Hubness/define_networks_SLURM.sh
```

Step 2 with SLURM array:
```bash
sbatch Hubness/static_hubs_SLURM.sh
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

- `static_hub_metrics_group.csv` (PC, strength per parcel)
  - f
- `sub-XX/static_hub_metrics_subject.csv`
  - f


### Dependencies

**Required** (standard):
- matplotlib, numpy, pandas, seaborn, (networkx)
