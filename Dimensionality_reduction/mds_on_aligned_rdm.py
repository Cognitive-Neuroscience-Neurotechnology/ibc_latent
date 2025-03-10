import os
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

def load_rdm(file_path):
    """Load an RDM from a .npy file."""
    return np.load(file_path)

def perform_mds(rdm, n_components=5): # n_components gives the number of dimensions to reduce to
    """Perform MDS on the RDM."""
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
    mds_coords = mds.fit_transform(rdm)
    stress = mds.stress_
    # Compute R-squared value
    r_squared = 1 - (stress / np.sum(rdm ** 2))
    return mds_coords, r_squared, stress

def cluster_data(data, n_clusters=2):
    """Cluster the data using KMeans."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    inertia = kmeans.inertia_
    silhouette_avg = silhouette_score(data, labels)
    return labels, inertia, silhouette_avg

def hierarchical_clustering(data, n_clusters=3):
    """Perform hierarchical clustering on the data."""
    Z = linkage(data, method='ward')
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    return labels, Z

def plot_clusters(data, labels, title, output_dir, subject):
    """Plot the clustered data and save the plot."""
    if data.shape[1] == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
        plt.title(title)
        plt.xlabel('MDS Dimension 1')
        plt.ylabel('MDS Dimension 2')
        plt.colorbar()
        
        # Save the plot
        plot_file = os.path.join(output_dir, f'clusters_{subject}_2d.png')
        plt.savefig(plot_file)
        plt.close()
    elif data.shape[1] == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', s=50)
        ax.set_title(title)
        ax.set_xlabel('MDS Dimension 1')
        ax.set_ylabel('MDS Dimension 2')
        ax.set_zlabel('MDS Dimension 3')
        fig.colorbar(scatter)
        
        # Save the plot
        plot_file = os.path.join(output_dir, f'clusters_{title}_{subject}_3d.png')
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
    
    # Create a new subfolder for the current run
    existing_runs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('run_')]
    run_number = len(existing_runs) + 1
    current_run_dir = os.path.join(output_dir, f'run_{run_number:02d}')
    os.makedirs(current_run_dir, exist_ok=True)

    # Get the list of subjects from the filenames in the directory
    subjects = [f.split('_')[-1].replace('.npy', '') for f in os.listdir(topographic_alignment_RDM_dir) if f.startswith('topographic_alignment_rdm_')]

    all_kmeans_labels = {}
    all_hier_labels = {}
    r_squared_values = {}
    stress_values = {}
    inertia_values = {}
    silhouette_scores = {}

    for subject in subjects:
        rdm_file = os.path.join(topographic_alignment_RDM_dir, f'topographic_alignment_rdm_{subject}.npy')
        rdm = load_rdm(rdm_file)
        
        # Perform MDS
        mds_coords, r_squared, stress = perform_mds(rdm)
        r_squared_values[subject] = r_squared
        stress_values[subject] = stress
        
        # Cluster the MDS coordinates using KMeans
        kmeans_labels, inertia, silhouette_avg = cluster_data(mds_coords)
        all_kmeans_labels[subject] = kmeans_labels
        inertia_values[subject] = inertia
        silhouette_scores[subject] = silhouette_avg
        
        # Save the KMeans labels to a CSV file
        kmeans_labels = kmeans_labels + 1  # Change labels from 0,1 to 1,2
        kmeans_labels_file = os.path.join(current_run_dir, f'kmeans_labels_{subject}.csv')
        np.savetxt(kmeans_labels_file, kmeans_labels, delimiter=",", fmt='%d')
        
        # Plot the KMeans clusters and save the plot
        plot_clusters(mds_coords, kmeans_labels, f'KMeans Clusters for {subject}', current_run_dir, subject)
        
        # Perform hierarchical clustering
        hier_labels, Z = hierarchical_clustering(mds_coords)
        all_hier_labels[subject] = hier_labels
        
        # Plot the clusters in dimensions and save the plot
        plot_clusters(mds_coords, hier_labels, f'Hierarchical Clusters for {subject}', current_run_dir, subject)
        
        # Plot the dendrogram and save the plot
        plot_dendrogram(Z, f'Dendrogram for {subject}', current_run_dir, subject)

    # Save the R-squared values, Kruskal's Stress, Inertia, and Silhouette Score to a file
    metrics_file = os.path.join(current_run_dir, 'mds_metrics.csv')
    metrics_df = pd.DataFrame({
        'Subject': subjects,
        'R_squared': [r_squared_values[subj] for subj in subjects],
        'Stress': [stress_values[subj] for subj in subjects],
        'Inertia': [inertia_values[subj] for subj in subjects],
        'Silhouette_Score': [silhouette_scores[subj] for subj in subjects]
    })
    metrics_df.to_csv(metrics_file, index=False)

    # Compute and print the similarity between the clusters of different subjects
    subjects_pairs = [(subjects[i], subjects[j]) for i in range(len(subjects)) for j in range(i+1, len(subjects))]
    for subj1, subj2 in subjects_pairs:
        ari_kmeans, nmi_kmeans = compute_similarity(all_kmeans_labels[subj1], all_kmeans_labels[subj2])
        ari_hier, nmi_hier = compute_similarity(all_hier_labels[subj1], all_hier_labels[subj2])
        #print(f'Similarity between {subj1} and {subj2}:')
        #print(f'  KMeans - ARI: {ari_kmeans:.2f}, NMI: {nmi_kmeans:.2f}')
        #print(f'  Hierarchical - ARI: {ari_hier:.2f}, NMI: {nmi_hier:.2f}')

if __name__ == "__main__":
    main()