import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_rdm(file_path):
    """Load an RDM from a .npy file."""
    return np.load(file_path)

def perform_tsne(rdm, n_components=3, perplexity=25, max_iter=5000): # Parameters: n_components = number of dimensions, perplexity = number of neighbors
    """Perform t-SNE on the RDM."""
    tsne = TSNE(n_components=n_components, perplexity=perplexity, max_iter=max_iter, metric='precomputed', init='random', random_state=42)
    tsne_coords = tsne.fit_transform(rdm)
    kl_divergence = tsne.kl_divergence_
    return tsne_coords, kl_divergence

def cluster_data(data, n_clusters=2): # Number of clusters (assumed subnetworks)
    """Cluster the data using KMeans."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels

def plot_clusters(data, labels, title, output_dir, subject):
    """Plot the clustered data and save the plot."""
    if data.shape[1] == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
        plt.title(title)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.colorbar()
    elif data.shape[1] == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', s=50)
        ax.set_title(title)
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_zlabel('t-SNE Dimension 3')
        fig.colorbar(sc)

    # Save the plot
    plot_file = os.path.join(output_dir, f'clusters_{subject}.png')
    plt.savefig(plot_file)
    plt.close()

def main():
    base_dir = '/home/hmueller2/ibc_code'
    topographic_alignment_RDM_dir = os.path.join(base_dir, 'ibc_output_RA', 'raw', 'topographic_alignment', 'rdm')
    output_dir = os.path.join(base_dir, 'ibc_output_tSNE', 'run_03')
    os.makedirs(output_dir, exist_ok=True)

    # Get the list of subjects from the filenames in the directory
    subjects = [f.split('_')[-1].replace('.npy', '') for f in os.listdir(topographic_alignment_RDM_dir) if f.startswith('topographic_alignment_rdm_')]

    for subject in subjects:
        rdm_file = os.path.join(topographic_alignment_RDM_dir, f'topographic_alignment_rdm_{subject}.npy')
        rdm = load_rdm(rdm_file)
        
        # Perform t-SNE
        tsne_coords, kl_divergence = perform_tsne(rdm)
        print(f'KL Divergence for {subject}: {kl_divergence}')

        # Cluster the t-SNE coordinates
        labels = cluster_data(tsne_coords)
        
        # Plot the clusters and save the plot
        plot_clusters(tsne_coords, labels, f'Clusters for {subject}', output_dir, subject)

if __name__ == "__main__":
    main()