import os
import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

def load_topographic_alignment_matrices(directory):
    """Load topographic alignment matrices from the specified directory."""
    matrices = []
    for file in os.listdir(directory):
        if file.endswith('.npy'):
            matrix = np.load(os.path.join(directory, file))
            matrices.append((file, matrix))
    return matrices

def perform_ica(matrix, n_components):
    """Perform Independent Component Analysis (ICA) on the provided matrix."""
    ica = FastICA(n_components=n_components, random_state=0)
    reshaped_matrix = matrix.reshape(matrix.shape[0], -1)  # Reshape to have multiple samples
    components = ica.fit_transform(reshaped_matrix)
    return components

def visualize_components(components, title):
    """Visualize the ICA components."""
    n_components = components.shape[1]
    plt.figure(figsize=(15, 5))
    for i in range(n_components):
        plt.subplot(1, n_components, i + 1)
        plt.plot(components[:, i])
        plt.title(f'{title} - Component {i + 1}')
    plt.tight_layout()
    plt.show()

def main():
    # Directory containing the topographic alignment matrices
    topographic_alignment_dir = '/home/hmueller2/ibc_code/ibc_output_RA_npy/raw/topographic_alignment/'
    
    # Load matrices
    matrices = load_topographic_alignment_matrices(topographic_alignment_dir)
    
    # Perform ICA and visualize for each subject
    for file, matrix in matrices:
        n_components = min(matrix.shape[0], 10)  # Set the number of components
        components = perform_ica(matrix, n_components)
        visualize_components(components, file)

if __name__ == "__main__":
    main()