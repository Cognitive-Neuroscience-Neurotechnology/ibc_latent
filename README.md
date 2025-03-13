# ibc_latent
Analysis code using the publicly available IBC (Individual Brain Charting) dataset.

**Goal**: Build latent representations of multiple cognitive tasks to better understand the Frontoparietal Network.

Raw data can be found on [OpenNeuro](https://openneuro.org/datasets/ds002685/versions/1.3.1). Preprocessed version of the data on [Ebrains, V2.0](https://search.kg.ebrains.eu/instances/44214176-0e8c-48de-8cff-4b6f9593415d). Overview about acquisition, tasks, preprocessing can be found in the [Official IBC Documentation](https://individual-brain-charting.github.io/docs/tasks.html#attention). Contrast Maps can be found on [Ebrains, V3.0](https://search.kg.ebrains.eu/instances/131add71-e838-4dab-b953-7b7a69ac5d8f).

## General Workflow of analysis
GLM -> RSA -> MDS -> Clusterung -> Mapping to Task Representations
(Alternative: GLM -> ICA)
## Project Structure

```
Data Info                             
├── common_task_sessions.csv    # For each task and subject, overview about in which sessions tasks were performed
├── contrasts_descriptor.pdf    # Description of how contrast maps were received from IBC data | version: v3.0 (from Ebrains, Thirion 2024)
└── subject_contrast_counts.csv # Counts how many task contrasts were received for each subject

GLM                             # Creating beta maps for each subject and task condition/contrast
├── src
│   ├── config.py               # Defines the directories.
│   ├── GLM_pipeline.py         # Main logic for the GLM pipeline.
│   └── tasks_contrasts.py      # Dictionary of tasks and their corresponding conditions -> contrasts.
├── README.md
└── requirements.txt            # Project dependencies

ICA
├── src                      
│   ├── config_ICA.py           #  
│   ├── ica_on_contrasts.py     # Finding individual components across all vertices in FPN
│   └── tasks_contrasts.py      # contrasts used for ICA
├── ICA_Visualization.ipynb     # Jupyter Notebook for visualization of components
├── README.md
└── requirements.txt            # Project dependencies   

MDS and Clustering                       
├── DimRed_Visualization.ipynb  # Visualization of RSMs and MDS results & comparing RSMs between clusters
├── FPN_clusters_results.py     # Plotting all subjects' clusters onto surface
├── hierarchical_clustering_.py # Second approach to create clusters after MDS
├── Index_parcel_cluster.py     # Map cluster labels to parcel indices and labels
├── mds_on_aligned_rdm.py       # Multidimensional Scaling on RA matrix
├── Surface_Visualization.ipynb # Visualization of obtained clusters in FPN parcels on surface
├── t-sne_on_aligned_rdm.py     # Alternative dimensionality reduction with clustering
├── README.md           
└── requirements.txt            # Project dependencies

RSA                             # Representational Similarity Analysis
├── src
│   ├── config_RSA.py           # Defines the directories.
│   ├── count_contrast.py       # Creates scv file with amount of task contrast done
│   ├── RA_calculation.py       # Representational Alignment with RSMs
│   ├── RSA_pipeline.py         # Main logic for RSA pipeline.
│   ├── RSA_test_one.py         # Test script for RSA.
│   └── tasks_contrasts.py      # Dictionary of task contrasts used for RSA.
├── FPN_vertices_parcels.py     # Vertex-to-parcel mapping for Frontoparietal parcels
├── glossary.md                 # Glossary of terms
├── RSM_Visualization.ipynb     # Jupyter notebook for RSM visualization
├── README.md                   # Project documentation
└── requirements.txt            # Project dependencies
```
