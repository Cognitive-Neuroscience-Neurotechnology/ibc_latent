import os
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt

def load_rsm(file_path):
    """Load an RA from a .npy file."""
    print(f"Loading RA from {file_path}")
    return np.load(file_path)

def hierarchical_clustering(data, n_clusters=5):
    """Perform hierarchical clustering on the data."""
    print("Performing hierarchical clustering")
    Z = linkage(data, method='ward')
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    return labels, Z

def plot_dendrogram(Z, title, output_dir, subject):
    """Plot the dendrogram and save the plot."""
    print(f"Plotting dendrogram for {subject}")
    plt.figure(figsize=(10, 8))
    dendrogram(Z)
    plt.title(title)
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    
    # Save the plot
    plot_file = os.path.join(output_dir, f'dendrogram_{subject}.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"Dendrogram saved to {plot_file}")

def main():
    base_dir = '/home/hmueller2/ibc_code'
    topographic_alignment_RSM_dir = os.path.join(base_dir, 'ibc_output_RA', 'raw', 'topographic_alignment', 'rsm')
    output_dir = os.path.join(base_dir, 'ibc_output_Hierarchical')
    os.makedirs(output_dir, exist_ok=True)

    # Debug: Print the directory contents
    print(f"Contents of {topographic_alignment_RSM_dir}:")
    print(os.listdir(topographic_alignment_RSM_dir))

    # Get the list of subjects from the filenames in the directory
    subjects = [f.split('_')[-1].replace('.npy', '') for f in os.listdir(topographic_alignment_RSM_dir) if f.startswith('topographic_alignment_sub-')]
    print(f"Found subjects: {subjects}")

    all_hier_labels = {}

    for subject in subjects:
        rsm_file = os.path.join(topographic_alignment_RSM_dir, f'topographic_alignment_{subject}.npy')
        rsm = load_rsm(rsm_file)
        
        # Perform hierarchical clustering
        hier_labels, Z = hierarchical_clustering(rsm)
        all_hier_labels[subject] = hier_labels
        
        # Plot the dendrogram and save the plot
        plot_dendrogram(Z, f'Dendrogram for {subject}', output_dir, subject)
        print(f"Hierarchical labels for {subject}: {hier_labels}")

if __name__ == "__main__":
    main()
