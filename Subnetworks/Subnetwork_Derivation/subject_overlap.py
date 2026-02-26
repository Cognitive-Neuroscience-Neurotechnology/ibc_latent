"""
Calculate between-subject overlap for k-means and infomap clustering approaches.
Compare the two methods based on their between-subject overlap.
"""
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from nibabel import load as nib_load

# Configuration
working_dir = Path('/ptmp/hmueller2/2025_ibc_latent/outputs')
kmeans_dir = working_dir / 'subnetworks' / 'kmeans'
infomap_dir = working_dir / 'subnetworks' / 'infomap'
output_dir = working_dir / 'subnetworks' / 'comparison' / 'between_subject_analysis'
output_dir.mkdir(parents=True, exist_ok=True)

subjects_file = working_dir / 'subjects_resting.txt'
subjects = [line.strip().split()[0].replace('sub-', '') for line in open(subjects_file)]
k = 2  # Number of clusters

print(f"Calculating between-subject overlap for {len(subjects)} subjects (k={k})")
print("="*70)

# Load FPN mask from first available subject's relabeled file
print("\n[1/4] Loading FPN mask...")
first_subject = subjects[0]
sample_file = kmeans_dir / f'sub-{first_subject}' / f'sub-{first_subject}_kmeans_on_vertices_relabeled.dlabel.nii'

if not sample_file.exists():
    raise FileNotFoundError(f"Cannot find sample file for sub-{first_subject}. Check paths:\n  {sample_file}")

sample_data = nib_load(str(sample_file)).get_fdata()[0, :64984]
fpn_mask = sample_data > 0  # Any labeled vertex is in FPN

print(f"  ✓ FPN mask contains {fpn_mask.sum()} vertices")

# Initialize dictionaries to store clustering results
kmeans_labels = {}
infomap_labels = {}

# Load clustering results for each subject
print("\n[2/4] Loading clustering results...")
missing_kmeans = []
missing_infomap = []

for subject in subjects:
    # Load k-means labels
    kmeans_file = kmeans_dir / f'sub-{subject}' / f'sub-{subject}_kmeans_on_vertices_relabeled.dlabel.nii'
    if not kmeans_file.exists():
        missing_kmeans.append(subject)
        continue
    kmeans_data = nib_load(str(kmeans_file)).get_fdata()[0, :64984]
    kmeans_labels[subject] = kmeans_data[fpn_mask]
    
    # Load infomap labels
    infomap_file = infomap_dir / f'sub-{subject}' / f'{subject}_FPN_infomap_communities_kmeans_relabeled.dlabel.nii'
    if not infomap_file.exists():
        missing_infomap.append(subject)
        continue
    infomap_data = nib_load(str(infomap_file)).get_fdata()[0, :64984]
    infomap_labels[subject] = infomap_data[fpn_mask]

print(f"  ✓ Loaded k-means labels for {len(kmeans_labels)} subjects")
print(f"  ✓ Loaded infomap labels for {len(infomap_labels)} subjects")

if missing_kmeans:
    print(f"  ⚠ Missing k-means files for: {', '.join(missing_kmeans)}")
if missing_infomap:
    print(f"  ⚠ Missing infomap files for: {', '.join(missing_infomap)}")

# Only analyze subjects that have both clustering results
common_subjects = set(kmeans_labels.keys()) & set(infomap_labels.keys())
print(f"  → {len(common_subjects)} subjects have both k-means and infomap results")

if len(common_subjects) == 0:
    raise ValueError("No subjects have both clustering results!")

# Filter to common subjects
kmeans_labels = {s: kmeans_labels[s] for s in common_subjects}
infomap_labels = {s: infomap_labels[s] for s in common_subjects}

# Calculate between-subject overlap for each method
def calculate_between_subject_overlap(labels_dict, method_name):
    subjects_list = sorted(list(labels_dict.keys()))
    n_subjects = len(subjects_list)
    ari_matrix = np.zeros((n_subjects, n_subjects))
    nmi_matrix = np.zeros((n_subjects, n_subjects))
    overlap_pct_matrix = np.zeros((n_subjects, n_subjects))
    
    print(f"\n[3/4] Calculating between-subject overlap for {method_name}...")
    total_pairs = n_subjects * (n_subjects - 1) // 2
    pair_count = 0
    
    for i in range(n_subjects):
        for j in range(i + 1, n_subjects):
            labels_i = labels_dict[subjects_list[i]]
            labels_j = labels_dict[subjects_list[j]]
            
            # Remove background (0) vertices
            valid_mask = (labels_i > 0) & (labels_j > 0)
            labels_i_valid = labels_i[valid_mask]
            labels_j_valid = labels_j[valid_mask]
            
            if len(labels_i_valid) == 0 or len(labels_j_valid) == 0:
                print(f"  ⚠ No valid vertices for pair {subjects_list[i]} - {subjects_list[j]}")
                continue
            
            # Calculate ARI and NMI
            ari = adjusted_rand_score(labels_i_valid, labels_j_valid)
            nmi = normalized_mutual_info_score(labels_i_valid, labels_j_valid)
            
            # Calculate percentage overlap (direct label agreement)
            overlap_pct = 100 * np.sum(labels_i_valid == labels_j_valid) / len(labels_i_valid)
            
            ari_matrix[i, j] = ari
            ari_matrix[j, i] = ari
            nmi_matrix[i, j] = nmi
            nmi_matrix[j, i] = nmi
            overlap_pct_matrix[i, j] = overlap_pct
            overlap_pct_matrix[j, i] = overlap_pct
            
            pair_count += 1
            if pair_count % 50 == 0 or pair_count == total_pairs:
                print(f"  Progress: {pair_count}/{total_pairs} pairs completed ({100*pair_count/total_pairs:.1f}%)")
    
    return ari_matrix, nmi_matrix, overlap_pct_matrix, subjects_list

kmeans_ari, kmeans_nmi, kmeans_overlap_pct, subjects_used = calculate_between_subject_overlap(kmeans_labels, "k-means")
infomap_ari, infomap_nmi, infomap_overlap_pct, _ = calculate_between_subject_overlap(infomap_labels, "infomap")

# Save results to CSV
print("\n[4/4] Saving results...")
kmeans_ari_csv = output_dir / f'kmeans_between_subject_ari_k{k}.csv'
kmeans_nmi_csv = output_dir / f'kmeans_between_subject_nmi_k{k}.csv'
kmeans_overlap_csv = output_dir / f'kmeans_between_subject_overlap_pct_k{k}.csv'
infomap_ari_csv = output_dir / f'infomap_between_subject_ari_k{k}.csv'
infomap_nmi_csv = output_dir / f'infomap_between_subject_nmi_k{k}.csv'
infomap_overlap_csv = output_dir / f'infomap_between_subject_overlap_pct_k{k}.csv'

pd.DataFrame(kmeans_ari, index=subjects_used, columns=subjects_used).to_csv(kmeans_ari_csv)
pd.DataFrame(kmeans_nmi, index=subjects_used, columns=subjects_used).to_csv(kmeans_nmi_csv)
pd.DataFrame(kmeans_overlap_pct, index=subjects_used, columns=subjects_used).to_csv(kmeans_overlap_csv)
pd.DataFrame(infomap_ari, index=subjects_used, columns=subjects_used).to_csv(infomap_ari_csv)
pd.DataFrame(infomap_nmi, index=subjects_used, columns=subjects_used).to_csv(infomap_nmi_csv)
pd.DataFrame(infomap_overlap_pct, index=subjects_used, columns=subjects_used).to_csv(infomap_overlap_csv)

print(f"  ✓ Saved: {kmeans_ari_csv}")
print(f"  ✓ Saved: {kmeans_nmi_csv}")
print(f"  ✓ Saved: {kmeans_overlap_csv}")
print(f"  ✓ Saved: {infomap_ari_csv}")
print(f"  ✓ Saved: {infomap_nmi_csv}")
print(f"  ✓ Saved: {infomap_overlap_csv}")

# Compare the two methods
print("\n" + "="*70)
print("COMPARING K-MEANS VS INFOMAP")
print("="*70)

n_subjects = len(subjects_used)
kmeans_ari_mean = np.mean(kmeans_ari[np.triu_indices(n_subjects, k=1)])
infomap_ari_mean = np.mean(infomap_ari[np.triu_indices(n_subjects, k=1)])
kmeans_nmi_mean = np.mean(kmeans_nmi[np.triu_indices(n_subjects, k=1)])
infomap_nmi_mean = np.mean(infomap_nmi[np.triu_indices(n_subjects, k=1)])
kmeans_overlap_mean = np.mean(kmeans_overlap_pct[np.triu_indices(n_subjects, k=1)])
infomap_overlap_mean = np.mean(infomap_overlap_pct[np.triu_indices(n_subjects, k=1)])

kmeans_ari_std = np.std(kmeans_ari[np.triu_indices(n_subjects, k=1)])
infomap_ari_std = np.std(infomap_ari[np.triu_indices(n_subjects, k=1)])
kmeans_nmi_std = np.std(kmeans_nmi[np.triu_indices(n_subjects, k=1)])
infomap_nmi_std = np.std(infomap_nmi[np.triu_indices(n_subjects, k=1)])
kmeans_overlap_std = np.std(kmeans_overlap_pct[np.triu_indices(n_subjects, k=1)])
infomap_overlap_std = np.std(infomap_overlap_pct[np.triu_indices(n_subjects, k=1)])

comparison_results = {
    'Metric': ['ARI', 'NMI', 'Overlap (%)'],
    'K-means_mean': [kmeans_ari_mean, kmeans_nmi_mean, kmeans_overlap_mean],
    'K-means_std': [kmeans_ari_std, kmeans_nmi_std, kmeans_overlap_std],
    'Infomap_mean': [infomap_ari_mean, infomap_nmi_mean, infomap_overlap_mean],
    'Infomap_std': [infomap_ari_std, infomap_nmi_std, infomap_overlap_std],
    'Difference': [kmeans_ari_mean - infomap_ari_mean, 
                   kmeans_nmi_mean - infomap_nmi_mean,
                   kmeans_overlap_mean - infomap_overlap_mean]
}
comparison_df = pd.DataFrame(comparison_results)
comparison_csv = output_dir / f'between_subject_comparison_k{k}.csv'
comparison_df.to_csv(comparison_csv, index=False)

print("\nBetween-Subject Overlap Comparison:")
print(comparison_df.to_string(index=False))
print(f"\n  ✓ Saved: {comparison_csv}")

# Interpretation
print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("Higher ARI/NMI = More consistent clustering across subjects")
print("Higher Overlap % = More vertices assigned to same cluster across subjects")
print("ARI/NMI closer to 0 = Low between-subject agreement")
print(f"\nNumber of subjects analyzed: {len(subjects_used)}")
print(f"Number of subject pairs analyzed: {n_subjects * (n_subjects - 1) // 2}")

if kmeans_ari_mean > infomap_ari_mean:
    print(f"\n→ K-means shows HIGHER between-subject consistency")
    print(f"  ARI: {kmeans_ari_mean:.3f} vs {infomap_ari_mean:.3f}")
    print(f"  Overlap: {kmeans_overlap_mean:.1f}% vs {infomap_overlap_mean:.1f}%")
else:
    print(f"\n→ Infomap shows HIGHER between-subject consistency")
    print(f"  ARI: {infomap_ari_mean:.3f} vs {kmeans_ari_mean:.3f}")
    print(f"  Overlap: {infomap_overlap_mean:.1f}% vs {kmeans_overlap_mean:.1f}%")

print("\n" + "="*70)
print("✓ Between-subject overlap analysis complete.")
print("="*70)