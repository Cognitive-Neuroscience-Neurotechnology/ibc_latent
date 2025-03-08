import os
import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import nibabel as nib
from task_contrasts import task_contrasts
from config_ICA import base_dir, output_dir
import seaborn as sns
import matplotlib.pyplot as plt

def load_contrast_map(file_path):
    try:
        img = nib.load(file_path)
        data = np.array([darray.data for darray in img.darrays])
        data = data.squeeze()  # Remove any singleton dimensions
        return data
    except FileNotFoundError:
        return None

def average_sessions(file_paths):
    data_sessions = [load_contrast_map(fp) for fp in file_paths]
    data_sessions = [data for data in data_sessions if data is not None]
    if not data_sessions:
        return None
    average_data = np.mean(data_sessions, axis=0)
    return average_data

def perform_ica(matrix, n_components):
    """Perform Independent Component Analysis (ICA) on the provided matrix."""
    ica = FastICA(n_components=n_components, random_state=0, max_iter=500)
    components = ica.fit_transform(matrix)  # Reduce contrasts
    return components, ica

def visualize_components_heatmap(components, title, output_dir):
    """Visualize the ICA components as a heatmap and save the plot."""
    plt.figure(figsize=(15, 10))
    sns.heatmap(components, cmap='viridis', annot=True)
    plt.title(f'{title} - ICA Components Heatmap')
    plt.xlabel('Components')
    plt.ylabel('Contrasts')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title}_components_heatmap.png'))
    plt.close()

def visualize_component_clusters(components, title, output_dir):
    """Visualize hierarchical clustering of vertices based on ICA components."""
    from scipy.cluster.hierarchy import dendrogram, linkage
    
    # Perform hierarchical clustering
    linked = linkage(components.T, 'ward')  # Transpose to get vertices as observations
    
    plt.figure(figsize=(18, 10))
    dendrogram(linked, 
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True)
    plt.title(f'{title} - Hierarchical Clustering of Vertices by Component Weights')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title}_vertices_dendrogram.png'))
    plt.close()

def main(n_ica_components, n_reduced_task_contrasts):
    # Process each subject
    subjects = [d.split('-')[1] for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('sub-')]
    for subject in subjects:
        print(f"Processing subject {subject}...")
        
        contrast_maps = []
        for hemisphere in ['lh', 'rh']:
            for task, contrasts in task_contrasts.items():
                for contrast in contrasts:
                    file_paths = [
                        os.path.join(
                            base_dir, 
                            f'sub-{subject}', 
                            session, 
                            f'sub-{subject}_ses-{session.split("-")[1]}_task-{task}_dir-ffx_space-fsaverage7_hemi-{hemisphere}_ZMap-{contrast}.gii'
                        ) 
                        for session in os.listdir(os.path.join(base_dir, f'sub-{subject}')) if session.startswith('ses-')
                    ]
                    
                    # Check if files exist
                    file_paths = [fp for fp in file_paths if os.path.exists(fp)]
                    
                    if not file_paths:
                        continue
                    
                    # Average the data across sessions if there are multiple sessions
                    if len(file_paths) > 1:
                        data = average_sessions(file_paths)
                    else:
                        data = load_contrast_map(file_paths[0])
                    
                    if data is not None:
                        contrast_maps.append(data)
        
        if not contrast_maps:
            print(f"No contrast maps found for subject {subject}. Skipping...")
            continue
        
        # Stack contrast maps to create a matrix (contrasts x vertices)
        matrix = np.vstack(contrast_maps)
        
        # Apply ICA to the matrix
        components, ica = perform_ica(matrix, n_ica_components)  # Reduce contrasts
        
        # Create subfolder in output_dir based on the number of components
        components_output_dir = os.path.join(output_dir, f'{n_ica_components}_components')
        os.makedirs(components_output_dir, exist_ok=True)
        
        # Save ICA components
        np.save(os.path.join(components_output_dir, f'sub-{subject}_ica_components.npy'), components)
#        print(f"ICA components for subject {subject} with shape {components.shape} (components × contrasts) saved to {components_output_dir}")
        
        # Visualize and save the components as heatmap
        visualize_components_heatmap(components, f'sub-{subject}', components_output_dir)
#        print(f"ICA component heatmap for subject {subject} saved to {components_output_dir}")
        
        # Add hierarchical clustering visualization
        #visualize_component_clusters(components, f'sub-{subject}', components_output_dir)
        #print(f"ICA component clustering for subject {subject} saved to {components_output_dir}")
        
        # Reduce the number of contrasts to a smaller set of overall reduced task contrasts
        reduced_task_contrasts = ica.transform(matrix)[:, :n_reduced_task_contrasts]
        np.save(os.path.join(components_output_dir, f'sub-{subject}_reduced_task_contrasts.npy'), reduced_task_contrasts)
#        print(f"Reduced task contrasts for subject {subject} with shape {reduced_task_contrasts.shape} (vertices × reduced contrasts) saved to {components_output_dir}")

if __name__ == "__main__":
    main(n_ica_components=4, n_reduced_task_contrasts=5)  # Specify the number of ICA components and reduced task contrasts here