# ibc_latent
Analysis code using the publicly available IBC (Individual Brain Charting) dataset.

**Goal**: Build latent representations of multiple cognitive tasks to better understand the Frontoparietal Network.

Raw data can be found on [OpenNeuro](https://openneuro.org/datasets/ds002685/versions/1.3.1). Preprocessed version of the data on [Ebrains, V2.0](https://search.kg.ebrains.eu/instances/44214176-0e8c-48de-8cff-4b6f9593415d). Overview about acquisition, tasks, preprocessing can be found in the [Official IBC Documentation](https://individual-brain-charting.github.io/docs/tasks.html#attention). UPDATE: Contrast Maps can be found on [Ebrains, V3.0](https://search.kg.ebrains.eu/instances/131add71-e838-4dab-b953-7b7a69ac5d8f).

## Project Structure

```
GLM                                # Creating beta maps for each subject and task condition/contrast
├── src                      
│   ├── config.py                  # Defines the directories.
│   ├── GLM_pipeline.py            # Main logic for the GLM pipeline.
│   └── tasks_contrasts.py         # Dictionary of tasks and their corresponding conditions -> contrasts.
├── First_GLM.ipynb                # Jupyter notebook to explore GLM outputs (tasks, design matrix,...)
├── README.md                
└── requirements.txt               # Project dependencies
RSA                                # Representational Similarity Analysis
├── Parcellation                      
│   ├── FPN_vertices_parcels.py    # ???
├── src                      
│   ├── config_RSA.py              # Defines the directories.
│   ├── RSA_pipeline.py            # ???
│   ├── RSA_test_one.py            # ???
│   └── tasks_contrasts.py         # Dictionary of task contrasts that were chosen to be used for RSA
├── glossary.md                    # ???
├── RSM_Visualization.ipynb        # ???
├── !!README.md                      # !! Missing
└── !!requirements.txt               # !! Missing
latent_analysis                    # Directory for latent representation analysis
├── src                      
│   ├── 
├── README.md                
└── requirements.txt               # Project dependencies
common_task_sessions.csv           # Contain for each subject an overview about in which sessions (+ how often) each task was done.
data_descriptor_contrasts.pdf      # Descriptor of methods used in contrast calculations (from Ebrains, Thirion 2024)
```
