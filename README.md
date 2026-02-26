# ibc_latent
Analysis code for my master thesis (and prior lab rotation) using the publicly available IBC (Individual Brain Charting) dataset.

**Goal**: Investigate functional specialization in the Frontoparietal Network using IBC data.

Raw data can be found on [OpenNeuro](https://openneuro.org/datasets/ds002685/versions/1.3.1). Preprocessed version of the data on [Ebrains, V2.0](https://search.kg.ebrains.eu/instances/44214176-0e8c-48de-8cff-4b6f9593415d). Overview about acquisition, tasks, preprocessing can be found in the [Official IBC Documentation](https://individual-brain-charting.github.io/docs/tasks.html#attention). Contrast Maps can be found on [Ebrains, V3.0](https://search.kg.ebrains.eu/instances/131add71-e838-4dab-b953-7b7a69ac5d8f).

## Project Structure

```
Data Info/                      # Dataset information and metadata
├── Session and task overview files
└── Contrast descriptors and counts

Infomap/                        # Community detection and network derivation using Infomap algorithm
├── Plotting/                   # Visualization scripts for network communities
└── Utilities/                  # Helper functions for Infomap analysis

Lab Rotation/                   # Analysis code from prior lab rotation project
├── ICA/                        # Independent Component Analysis on FPN vertices
├── MDS and Clustering/         # Dimensionality reduction and hierarchical clustering
├── RSA/                        # Representational Similarity Analysis pipeline
└── images_presentation/        # Figures and visualizations for presentations

MSCcodebase/                    # External utilities from Midnight Scan Club repository
└── Utilities/                  # MATLAB tools for brain surface analysis, CIFTI I/O, and Infomap

Preprocessing/                  # fMRIPrep pipeline and preprocessing scripts
├── Aradia/                     # Cluster-specific configuration files
└── Dependencies/               # Required dependencies for preprocessing

public_analysis/                # IBC public analysis repository (contrast map generation)
├── ibc_data/                   # Data handling utilities
├── ibc_public/                 # Public analysis scripts
├── papers_scripts/             # Scripts from IBC papers
└── scripts/                    # General analysis scripts

Subnetworks/                    # Subnetwork derivation and analysis
├── Subnetwork_Analysis/        # Activation and PPI analyses on derived subnetworks
└── Subnetwork_Derivation/      # Scripts for deriving FPN subnetworks from individual networks
```

## Analysis Order

1. Preprocessing
    - See [Preprocessing](Preprocessing) for fMRIPrep and related scripts.
2. Contrast map creation
    - See [public_analysis/ibc_public](public_analysis/ibc_public) for contrast map generation.
3. Infomap
    - See [Infomap](Infomap) for community detection and network derivation.
4. Subnetwork derivation
    - See [Subnetworks/Subnetwork_Derivation](Subnetworks/Subnetwork_Derivation) for building subnetworks.
5. Subnetwork analysis
    - See [Subnetworks/Subnetwork_Analysis](Subnetworks/Subnetwork_Analysis) for activation and ppi analyses.

## Dependencies / Environment

- Python (recommend using a virtual environment)
- fMRIPrep (container or local install)
- Workbench (for CIFTI/GIFTI operations)
- MATLAB (for MSCcodebase utilities)

## Local Paths

Set these environment variables so local paths do not appear in the repo:

## Contact

- Name: Hannah Müller
- Email: hannah.mueller@tuebingen.mpg.de