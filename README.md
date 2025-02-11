# ibc_latent
Analysis code using the publicly available IBC (Individual Brain Charting) dataset.

**Goal**: Build latent representations of multiple cognitive tasks to better understand the Frontoparietal Network.

Raw data can be found on [OpenNeuro](https://openneuro.org/datasets/ds002685/versions/1.3.1). Preprocessed version of the data on [Ebrains](https://search.kg.ebrains.eu/instances/44214176-0e8c-48de-8cff-4b6f9593415d). Overview about acquisition, tasks, preprocessing can be found in the [Official IBC Documentation](https://individual-brain-charting.github.io/docs/tasks.html#attention). UPDATE: Contrast Maps can be found on [Ebrains](https://search.kg.ebrains.eu/instances/131add71-e838-4dab-b953-7b7a69ac5d8f).

## Project Structure

```
GLM                          # Creating beta maps for each subject and task condition/contrast
├── src                      
│   ├── config.py            # Defines the directories.
│   ├── GLM_pipeline.py      # Main logic for the GLM pipeline.
│   └── tasks_contrasts.py   # Dictionary of tasks and their corresponding conditions -> contrasts.
├── README.md                
└── requirements.txt         # Project dependencies
RSA                          # !! This folder will be added later to build latent representations.
├── src                      
│   ├── config.py            # Defines the directories.
├── README.md                
common_task_sessions.csv     # Contain for each subject an overview about in which sessions (+ how often) each task was done.
First_GLM.ipynb              # Jupyter notebook to explore GLM outputs (tasks, design matrix,...)
```
