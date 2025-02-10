# GLM Pipeline

## Overview
This project implements a Generalized Linear Model (GLM) pipeline. It includes functionality for selecting and processing various tasks based on user-defined conditions. It is based on BIDS structure and used with preprocessed data of the IBC dataset.

## Project Structure
```
GLM
├── src                      # Source code
│   ├── config.py            # Defines the directories
│   ├── GLM_pipeline.py      # Main logic for the GLM pipeline (Run this!)
│   └── task_contrasts.py   # Dictionary of tasks and their corresponding conditions -> 
├── README.md                # Project documentation
└── requirements.txt         # Project dependencies
```

## Files Description

- **src/GLM_pipeline.py**: Contains the main logic for the GLM pipeline, including the selection and processing of tasks.
  
- **src/task_contrasts.py**: This contain a list of tasks and their corresponding conditions based on user selections. It should be transformed into a dictionary with the contrasts (e.g. condition1 - condition2) that want to used for comparing beta maps.

- **config.py**: Defines the base_dir (where raw data is stored) and the output_dir (where the beta maps will be stored).

- **requirements.txt**: Lists the dependencies required for the project.

## Usage
To run the GLM pipeline, execute the `GLM_pipeline.py` script in the `src` directory. Ensure that all dependencies listed in `requirements.txt` are installed.

## Installation
To install the required dependencies, run:
```
pip install -r requirements.txt
```
