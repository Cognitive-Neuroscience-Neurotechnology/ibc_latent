import os
import numpy as np
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

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

def hierarchical_clustering(data, n_clusters=5):
    """Perform hierarchical clustering on the data."""
    Z = linkage(data, method='ward')
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    return labels, Z

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

def plot_dendrogram(Z, title, output_dir, subject):
    """Plot the dendrogram and save the plot."""
    plt.figure(figsize=(10, 8))
    dendrogram(Z)
    plt.title(title)
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    
    # Save the plot
    plot_file = os.path.join(output_dir, f'dendrogram_{subject}.png')
    plt.savefig(plot_file)
    plt.close()

def compute_similarity(labels1, labels2):
    """Compute the similarity between two clusterings."""
    ari = adjusted_rand_score(labels1, labels2) # Adjusted Rand Index: 0 is random, 1 is perfect match
    nmi = normalized_mutual_info_score(labels1, labels2) # Normalized Mutual Information: 0 is no shared info, 1 is perfect match
    return ari, nmi

def main():
    base_dir = '/home/hmueller2/ibc_code'
    topographic_alignment_RDM_dir = os.path.join(base_dir, 'ibc_output_RA', 'raw', 'topographic_alignment', 'rdm')
    output_dir = os.path.join(base_dir, 'ibc_output_MDS')
    os.makedirs(output_dir, exist_ok=True)

    # Get the list of subjects from the filenames in the directory
    subjects = [f.split('_')[-1].replace('.npy', '') for f in os.listdir(topographic_alignment_RDM_dir) if f.startswith('topographic_alignment_rdm_')]

    all_kmeans_labels = {}
    all_hier_labels = {}

    for subject in subjects:
        rdm_file = os.path.join(topographic_alignment_RDM_dir, f'topographic_alignment_rdm_{subject}.npy')
        rdm = load_rdm(rdm_file)
        
        # Perform MDS
        mds_coords = perform_mds(rdm)
        
        # Cluster the MDS coordinates using KMeans
        kmeans_labels = cluster_data(mds_coords)
        all_kmeans_labels[subject] = kmeans_labels
        
        # Plot the KMeans clusters and save the plot
        plot_clusters(mds_coords, kmeans_labels, f'KMeans Clusters for {subject}', output_dir, subject)
        
        # Perform hierarchical clustering
        hier_labels, Z = hierarchical_clustering(mds_coords)
        all_hier_labels[subject] = hier_labels
        
        # Plot the hierarchical clusters and save the plot
        plot_clusters(mds_coords, hier_labels, f'Hierarchical Clusters for {subject}', output_dir, subject)
        
        # Plot the dendrogram and save the plot
        plot_dendrogram(Z, f'Dendrogram for {subject}', output_dir, subject)

    # Compute and print the similarity between the clusters of different subjects
    subjects_pairs = [(subjects[i], subjects[j]) for i in range(len(subjects)) for j in range(i+1, len(subjects))]
    for subj1, subj2 in subjects_pairs:
        ari_kmeans, nmi_kmeans = compute_similarity(all_kmeans_labels[subj1], all_kmeans_labels[subj2])
        ari_hier, nmi_hier = compute_similarity(all_hier_labels[subj1], all_hier_labels[subj2])
        print(f'Similarity between {subj1} and {subj2}:')
        print(f'  KMeans - ARI: {ari_kmeans:.2f}, NMI: {nmi_kmeans:.2f}')
        print(f'  Hierarchical - ARI: {ari_hier:.2f}, NMI: {nmi_hier:.2f}')

if __name__ == "__main__":
    main()