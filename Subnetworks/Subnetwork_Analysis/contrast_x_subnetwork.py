"""
Analyze functional roles of FPN subnetworks by comparing contrast map activations.
Two approaches:
A) Mean z-score within each subnetwork
B) Overlap between thresholded z-maps and subnetworks
"""

import sys
import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

def permutation_test(z_subnet1, z_subnet2, n_permutations=10000):
    """
    Permutation test to assess if mean difference is significant.
    Returns: observed difference, p-value
    """
    observed_diff = np.mean(z_subnet1) - np.mean(z_subnet2)
    
    # Pool data
    pooled = np.concatenate([z_subnet1, z_subnet2])
    n1 = len(z_subnet1)
    
    # Generate null distribution
    null_diffs = np.zeros(n_permutations)
    for i in range(n_permutations):
        shuffled = np.random.permutation(pooled)
        null_diffs[i] = np.mean(shuffled[:n1]) - np.mean(shuffled[n1:])
    
    # Two-tailed p-value
    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
    
    return observed_diff, p_value

subject = sys.argv[1]
subnetwork_dir = f'/ptmp/hmueller2/Downloads/subnetworks/infomap/sub-{subject}'
contrast_base = f'/ptmp/hmueller2/Downloads/contrast_maps_fsLR/sub-{subject}'
output_dir = f'/ptmp/hmueller2/Downloads/subnetwork_analysis_results/sub-{subject}'
os.makedirs(output_dir, exist_ok=True)

# Load FPN subnetwork masks
subnetwork_path = os.path.join(subnetwork_dir, f'{subject}_FPN_infomap_communities_kmeans_relabeled.dscalar.nii')
if not os.path.exists(subnetwork_path):
    print(f"Subnetwork file not found: {subnetwork_path}")
    sys.exit(1)

subnetwork_img = nib.load(subnetwork_path)
subnetwork_data = subnetwork_img.get_fdata()

print(f"Available k solutions: {subnetwork_data.shape[0]}")
print(f"Using k=2 solution (index 1)")

# Access k=2 solution (second row/timepoint, index 1)
subnet1_mask = (subnetwork_data[1] == 1)  # Changed from [0] to [1]
subnet2_mask = (subnetwork_data[1] == 2)  # Changed from [0] to [1]

print(f"Subject: {subject}")
print(f"Subnetwork A (label=1): {subnet1_mask.sum()} vertices")
print(f"Subnetwork B (label=2): {subnet2_mask.sum()} vertices")
print(f"{'='*60}\n")

# ========== Find all fixed-effects contrast z-maps ==========
task_dirs = sorted(glob.glob(os.path.join(contrast_base, 'res_task-*_space-fsLR_dir-ffx')))

if not task_dirs:
    print(f"No fixed-effects contrast directories found in {contrast_base}")
    sys.exit(1)

results = []

for task_dir in task_dirs:
    task_name = os.path.basename(task_dir).split('_')[0].replace('res_task-', '')
    z_map_dir = os.path.join(task_dir, 'z_score_maps')
    
    if not os.path.exists(z_map_dir):
        print(f"No z_score_maps found in {task_dir}")
        continue
    
    z_maps = sorted(glob.glob(os.path.join(z_map_dir, '*.dscalar.nii')))
    
    for z_map_path in z_maps:
        contrast_name = os.path.basename(z_map_path).replace('.dscalar.nii', '')
        
        # Load z-score map
        z_img = nib.load(z_map_path)
        z_data = z_img.get_fdata()[0]  # Shape: (n_vertices,)
        
        # ========== APPROACH A: Mean activation ==========
        z_subnet1 = z_data[subnet1_mask]
        z_subnet2 = z_data[subnet2_mask]
        
        mean_a = np.mean(z_subnet1)
        mean_b = np.mean(z_subnet2)
        median_a = np.median(z_subnet1)
        median_b = np.median(z_subnet2)
        std_a = np.std(z_subnet1)
        std_b = np.std(z_subnet2)
        
        # Weighted mean (absolute z-scores as weights to emphasize strong activations)
        weights_a = np.abs(z_subnet1)
        weights_b = np.abs(z_subnet2)
        weighted_mean_a = np.average(z_subnet1, weights=weights_a) if weights_a.sum() > 0 else 0
        weighted_mean_b = np.average(z_subnet2, weights=weights_b) if weights_b.sum() > 0 else 0
        
        # T-test
        tstat, pval_ttest = stats.ttest_ind(z_subnet1, z_subnet2)
        
        # Permutation test (more robust)
        observed_diff, pval_perm = permutation_test(z_subnet1, z_subnet2, n_permutations=10000)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
        
        # ========== APPROACH B: Overlap with thresholded maps ==========
        thresholds = [1.96, 2.0, 2.5, 3.0]  # 1.96 ≈ p<0.05 two-tailed
        overlap_results = {}
        
        for thresh in thresholds:
            # Positive activations
            activated = z_data > thresh
            overlap_a_pos = np.sum(activated & subnet1_mask)
            overlap_b_pos = np.sum(activated & subnet2_mask)
            prop_a_pos = overlap_a_pos / subnet1_mask.sum()
            prop_b_pos = overlap_b_pos / subnet2_mask.sum()
            
            # Negative activations (deactivations)
            deactivated = z_data < -thresh
            overlap_a_neg = np.sum(deactivated & subnet1_mask)
            overlap_b_neg = np.sum(deactivated & subnet2_mask)
            prop_a_neg = overlap_a_neg / subnet1_mask.sum()
            prop_b_neg = overlap_b_neg / subnet2_mask.sum()
            
            # Store with cleaner names
            thresh_str = str(thresh).replace('.', '_')
            overlap_results[f'overlap_a_pos_z{thresh_str}'] = overlap_a_pos
            overlap_results[f'overlap_b_pos_z{thresh_str}'] = overlap_b_pos
            overlap_results[f'prop_a_pos_z{thresh_str}'] = prop_a_pos
            overlap_results[f'prop_b_pos_z{thresh_str}'] = prop_b_pos
            overlap_results[f'overlap_a_neg_z{thresh_str}'] = overlap_a_neg
            overlap_results[f'overlap_b_neg_z{thresh_str}'] = overlap_b_neg
            overlap_results[f'prop_a_neg_z{thresh_str}'] = prop_a_neg
            overlap_results[f'prop_b_neg_z{thresh_str}'] = prop_b_neg
        
        # ========== Store results ==========
        result = {
            'subject': subject,
            'task': task_name,
            'contrast': contrast_name,
            # Approach A - Basic statistics
            'mean_subnet_a': mean_a,
            'mean_subnet_b': mean_b,
            'median_subnet_a': median_a,
            'median_subnet_b': median_b,
            'std_subnet_a': std_a,
            'std_subnet_b': std_b,
            'weighted_mean_subnet_a': weighted_mean_a,
            'weighted_mean_subnet_b': weighted_mean_b,
            # Differences and effect sizes
            'mean_diff_a_minus_b': mean_a - mean_b,
            'cohens_d': cohens_d,
            # Statistical tests
            'ttest_tstat': tstat,
            'ttest_pval': pval_ttest,
            'perm_test_pval': pval_perm,
            # Approach B
            **overlap_results
        }
        
        results.append(result)
        
        # Determine which network is more involved (based on mean)
        dominant = "A" if mean_a > mean_b else "B"
        sig_marker = "***" if pval_perm < 0.001 else "**" if pval_perm < 0.01 else "*" if pval_perm < 0.05 else ""
        
        print(f"{task_name}/{contrast_name}: "
              f"Mean A={mean_a:.3f}, B={mean_b:.3f} → {dominant} "
              f"(p_perm={pval_perm:.4f}{sig_marker}, d={cohens_d:.3f})")

# ========== Save results to CSV ==========
if results:
    results_df = pd.DataFrame(results)
    output_csv = os.path.join(output_dir, 'contrast_x_subnetwork_analysis.csv')
    results_df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to: {output_csv}")
    
    # ========== Summary statistics ==========
    print(f"\n{'='*60}")
    print("SUMMARY: Which network is more involved per contrast?")
    print(f"{'='*60}")
    
    # Count how often each network is dominant
    results_df['dominant_network'] = results_df.apply(
        lambda row: 'A' if row['mean_subnet_a'] > row['mean_subnet_b'] else 'B',
        axis=1
    )
    
    # Filter significant differences (permutation test p < 0.05)
    significant = results_df[results_df['perm_test_pval'] < 0.05]
    
    print(f"\nTotal contrasts analyzed: {len(results_df)}")
    print(f"Significant differences (p<0.05, permutation): {len(significant)}")
    print(f"\nNetwork dominance (all contrasts):")
    print(results_df['dominant_network'].value_counts())
    
    if len(significant) > 0:
        print(f"\nNetwork dominance (significant only):")
        print(significant['dominant_network'].value_counts())
        
        print(f"\nTop 10 strongest differences (A > B):")
        top_a = results_df.nlargest(10, 'mean_diff_a_minus_b')[['task', 'contrast', 'mean_diff_a_minus_b', 'cohens_d', 'perm_test_pval']]
        print(top_a.to_string(index=False))
        
        print(f"\nTop 10 strongest differences (B > A):")
        top_b = results_df.nsmallest(10, 'mean_diff_a_minus_b')[['task', 'contrast', 'mean_diff_a_minus_b', 'cohens_d', 'perm_test_pval']]
        print(top_b.to_string(index=False))
    
    # ========== Overlap analysis summary ==========
    print(f"\n{'='*60}")
    print("OVERLAP ANALYSIS (at z > 2.0 threshold):")
    print(f"{'='*60}")
    
    overlap_summary = results_df[['task', 'contrast', 'prop_a_pos_z2_0', 'prop_b_pos_z2_0']].copy()
    overlap_summary['diff_a_minus_b'] = overlap_summary['prop_a_pos_z2_0'] - overlap_summary['prop_b_pos_z2_0']
    
    print("\nHighest overlap with Subnetwork A:")
    print(overlap_summary.nlargest(5, 'diff_a_minus_b')[['task', 'contrast', 'prop_a_pos_z2_0', 'prop_b_pos_z2_0']].to_string(index=False))
    
    print("\nHighest overlap with Subnetwork B:")
    print(overlap_summary.nsmallest(5, 'diff_a_minus_b')[['task', 'contrast', 'prop_a_pos_z2_0', 'prop_b_pos_z2_0']].to_string(index=False))

else:
    print("No results to save!")

print(f"\n✓ Analysis complete for subject {subject}!")