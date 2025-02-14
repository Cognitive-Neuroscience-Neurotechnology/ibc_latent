# RSA Pipeline

This project implements a Representational Similarity Analysis (RSA) pipeline designed to analyze brain imaging data. The pipeline processes contrast maps and utilizes the Glasser atlas to extract relevant features for analysis.

## Project Structure

- **data/**
  - **contrast_maps/**: Contains the contrast maps used in the RSA analysis.
  - **atlas/**: Holds the atlas files, such as the Glasser atlas, used for mapping vertices to parcels.
  - **output/**: Stores the output results from the RSA pipeline, including correlation matrices and any processed data.

- **notebooks/**: 
  - **First_RSA.ipynb**: A Jupyter notebook that contains the initial code and documentation for the RSA analysis. It serves as a reference for the pipeline implementation.

- **src/**: 
  - **\_\_init\_\_.py**: Marks the src directory as a Python package.
  - **load_data.py**: Contains functions for loading contrast maps and atlas data, handling file paths and data formats.
  - **process_data.py**: Implements core RSA processing functions, including averaging data across sessions, extracting FPN activations, and calculating correlation matrices.
  - **utils.py**: Includes utility functions that support the main processing functions, such as data normalization and reshaping.

- **scripts/**: 
  - **run_pipeline.py**: The entry point for executing the RSA pipeline. It iterates through all subjects, sessions, contrasts, and hemispheres, calling the appropriate functions from the src modules.

- **requirements.txt**: Lists the Python package dependencies required for the project, such as numpy, pandas, nibabel, and scipy.

## Setup Instructions

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required packages using pip:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the RSA pipeline, execute the following command:
```
python scripts/run_pipeline.py
```

This will initiate the analysis across all specified subjects, sessions, contrasts, and hemispheres, generating the output in the designated output directory.

## Acknowledgments

This project utilizes the Glasser atlas and various Python libraries for data processing and analysis.