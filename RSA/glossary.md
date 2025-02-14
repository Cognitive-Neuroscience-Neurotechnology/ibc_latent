## Variables

- **`base_dir`**: The base directory where the subject data is stored. Defined in `config_RSA.py`.
- **`output_dir`**: The directory where the output files (RDMs) will be saved. Defined in `config_RSA.py`.
- **`network_partition_path`**: The file path to the network partition file, which contains information about different brain networks.
- **`network_partition`**: A DataFrame loaded from the network partition file, containing information about brain parcels and their network assignments.
- **`fpn_parcels`**: A subset of `network_partition` containing only the parcels that belong to the frontoparietal network (Networkkey = 7).
- **`fpn_parcels_names`**: A list of parcel names that belong to the frontoparietal network.
- **`lh_annot_file`**: The file path to the left hemisphere annotation file.
- **`rh_annot_file`**: The file path to the right hemisphere annotation file.
- **`labels_lh`**: Labels for the left hemisphere parcels, loaded from the annotation file.
- **`ctab_lh`**: Color table for the left hemisphere parcels, loaded from the annotation file.
- **`names_lh`**: Names of the left hemisphere parcels, loaded from the annotation file.
- **`labels_rh`**: Labels for the right hemisphere parcels, loaded from the annotation file.
- **`ctab_rh`**: Color table for the right hemisphere parcels, loaded from the annotation file.
- **`names_rh`**: Names of the right hemisphere parcels, loaded from the annotation file.
- **`vertices_lh`**: An array of vertex indices for the left hemisphere.
- **`vertices_rh`**: An array of vertex indices for the right hemisphere.
- **`lh_parcel_mapping`**: A dictionary mapping each vertex in the left hemisphere to its corresponding parcel name.
- **`rh_parcel_mapping`**: A dictionary mapping each vertex in the right hemisphere to its corresponding parcel name.
- **`fpn_parcels_lh_mapping`**: A dictionary mapping each vertex in the left hemisphere to its corresponding frontoparietal parcel name.
- **`fpn_parcels_rh_mapping`**: A dictionary mapping each vertex in the right hemisphere to its corresponding frontoparietal parcel name.
- **`subjects`**: A list of subject IDs extracted from the `base_dir`.
- **`hemispheres`**: A list of hemispheres to process (`'lh'` for left hemisphere and `'rh'` for right hemisphere).
- **`session_dirs`**: A list of session directories for the current subject.
- **`file_paths`**: A list of file paths to the surface maps for the current subject, task, contrast, and hemisphere.
- **`parcel_data`**: A dictionary where keys are parcel names and values are activation data arrays for each parcel.
- **`activations`**: A 2D NumPy array where rows represent different observations (e.g., time points, trials) and columns represent different features (e.g., voxels, channels).
- **`rdm`**: The Representational Dissimilarity Matrix (RDM) computed for each parcel.
- **`output_file`**: The file path where the RDM will be saved.

## Functions

- **`load_surface_map(file_path)`**: Loads a surface map from the specified file path and returns the data as a NumPy array.
- **`extract_parcel_data(data, parcel_mapping)`**: Extracts activation data for each parcel based on the provided parcel mapping and returns a dictionary where keys are parcel names and values are activation data arrays.
- **`average_sessions(file_paths)`**: Loads surface maps from multiple sessions, averages the data across sessions, and returns the averaged data as a NumPy array.