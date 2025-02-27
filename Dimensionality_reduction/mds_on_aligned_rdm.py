import os
import numpy as np
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_rdm(file_path):
    """Load an RDM from a .npy file."""
    return np.load(file_path)

def perform_mds(rdm, n_components=2):
    """Perform MDS on the RDM."""
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
    mds_coords = mds.fit_transform(rdm)
    return mds_coords

def cluster_data(data, n_clusters=5):
    """Cluster the data using KMeans."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels

def plot_clusters(data, labels, title, output_dir, subject):
    """Plot the clustered data and save the plot."""
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(title)
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.colorbar()
    
    # Save the plot
    plot_file = os.path.join(output_dir, f'clusters_{subject}.png')
    plt.savefig(plot_file)
    plt.close()

def main():
    base_dir = '/home/hmueller2/ibc_code'
    topographic_alignment_RDM_dir = os.path.join(base_dir, 'ibc_output_RA', 'raw', 'topographic_alignment', 'rdm')
    output_dir = os.path.join(base_dir, 'ibc_output_MDS')
    os.makedirs(output_dir, exist_ok=True)

    # Get the list of subjects from the filenames in the directory
    subjects = [f.split('_')[-1].replace('.npy', '') for f in os.listdir(topographic_alignment_RDM_dir) if f.startswith('topographic_alignment_rdm_')]

    for subject in subjects:
        rdm_file = os.path.join(topographic_alignment_RDM_dir, f'topographic_alignment_rdm_{subject}.npy')
        rdm = load_rdm(rdm_file)
        
        # Perform MDS
        mds_coords = perform_mds(rdm)
        
        # Cluster the MDS coordinates
        labels = cluster_data(mds_coords)
        
        # Plot the clusters and save the plot
        plot_clusters(mds_coords, labels, f'Clusters for {subject}', output_dir, subject)

if __name__ == "__main__":
    main()