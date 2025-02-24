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
            matrices.append(matrix)
    return matrices

def perform_ica(matrices, n_components):
    """Perform Independent Component Analysis (ICA) on the provided matrices."""
    ica = FastICA(n_components=n_components, random_state=0)
    # Flatten the matrices for ICA
    flattened_matrices = np.array([matrix.flatten() for matrix in matrices])
    components = ica.fit_transform(flattened_matrices)
    return components

def visualize_components(components):
    """Visualize the ICA components."""
    n_components = components.shape[1]
    plt.figure(figsize=(15, 5))
    for i in range(n_components):
        plt.subplot(1, n_components, i + 1)
        plt.plot(components[:, i])
        plt.title(f'Component {i + 1}')
    plt.tight_layout()
    plt.show()

def main():
    # Directory containing the topographic alignment matrices
    topographic_alignment_dir = 'home/hmueller2/ibc_code/ibc_output_RA_npy/raw/topographic_alignment'
    
    # Load matrices
    matrices = load_topographic_alignment_matrices(topographic_alignment_dir)
    
    # Perform ICA
    n_components = min(len(matrices), 10)  # Set the number of components
    components = perform_ica(matrices, n_components)
    
    # Visualize the results
    visualize_components(components)

if __name__ == "__main__":
    main()