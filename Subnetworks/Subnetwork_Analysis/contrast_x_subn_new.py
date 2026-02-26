"""
Analyze functional roles of FPN subnetworks (FPN_A and FPN_B).

Approach A (Primary): Compare mean z-score activations between subnetworks
Approach B (Validation): Calculate overlap with thresholded activation maps

Goal: Identify distinct functional roles (e.g., introspective vs visuospatial)

After this, run "aggregate_subjects.py"
"""

import sys
import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.stats import false_discovery_control
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_effect_size(subnet_a, subnet_b):
    """Calculate Cohen's d effect size."""
    mean_a, mean_b = np.mean(subnet_a), np.mean(subnet_b)
    std_a, std_b = np.std(subnet_a, ddof=1), np.std(subnet_b, ddof=1)
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
    return (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0

def permutation_test(subnet_a, subnet_b, n_permutations=10000):
    """Non-parametric permutation test for mean difference."""
    observed_diff = np.mean(subnet_a) - np.mean(subnet_b)
    pooled = np.concatenate([subnet_a, subnet_b])
    n1 = len(subnet_a)
    
    null_diffs = np.zeros(n_permutations)
    for i in range(n_permutations):
        shuffled = np.random.permutation(pooled)
        null_diffs[i] = np.mean(shuffled[:n1]) - np.mean(shuffled[n1:])
    
    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
    return observed_diff, p_value

def calculate_overlap_metrics(z_data, mask_a, mask_b, thresholds=[1.96, 2.0, 2.5]):
    """Calculate proportion of subnetwork vertices exceeding threshold."""
    overlap_metrics = {}
    
    for thresh in thresholds:
        # Positive activations
        activated = z_data > thresh
        n_a_pos = np.sum(activated & mask_a)
        n_b_pos = np.sum(activated & mask_b)
        
        # Negative activations
        deactivated = z_data < -thresh
        n_a_neg = np.sum(deactivated & mask_a)
        n_b_neg = np.sum(deactivated & mask_b)
        
        thresh_key = f"z{str(thresh).replace('.', '_')}"
        overlap_metrics.update({
            f'prop_a_pos_{thresh_key}': n_a_pos / mask_a.sum(),
            f'prop_b_pos_{thresh_key}': n_b_pos / mask_b.sum(),
            f'prop_a_neg_{thresh_key}': n_a_neg / mask_a.sum(),
            f'prop_b_neg_{thresh_key}': n_b_neg / mask_b.sum(),
        })
    
    return overlap_metrics

# ========== Setup paths ==========
subject = sys.argv[1] if len(sys.argv) > 1 else input("Enter subject ID: ")

subnetwork_dir = f'/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_derivation/infomap/sub-{subject}'
contrast_base = f'/ptmp/hmueller2/2025_ibc_latent/outputs/glm/contrast_maps_fsLR/sub-{subject}'
output_dir = f'/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_activation/sub-{subject}'
os.makedirs(output_dir, exist_ok=True)

# ========== Load FPN subnetworks ==========
subnetwork_path = os.path.join(subnetwork_dir, 
    f'{subject}_FPN_infomap_communities_kmeans_relabeled.dscalar.nii')

if not os.path.exists(subnetwork_path):
    print(f"ERROR: Subnetwork file not found: {subnetwork_path}")
    sys.exit(1)

subnetwork_img = nib.load(subnetwork_path)
subnetwork_data = subnetwork_img.get_fdata()

# Use k=2 solution (index 1)
fpn_a_mask = (subnetwork_data[1] == 1)
fpn_b_mask = (subnetwork_data[1] == 2)

print(f"\nSubject: {subject}")
print(f"FPN_A vertices: {fpn_a_mask.sum()}")
print(f"FPN_B vertices: {fpn_b_mask.sum()}")
print(f"{'='*70}\n")

# ========== Process all contrasts ==========
task_dirs = sorted(glob.glob(os.path.join(contrast_base, 'res_task-*_space-fsLR_dir-ffx')))

if not task_dirs:
    print(f"ERROR: No task directories found in {contrast_base}")
    sys.exit(1)

results = []

for task_dir in task_dirs:
    task_name = os.path.basename(task_dir).split('_space-')[0].replace('res_task-', '')
    z_map_dir = os.path.join(task_dir, 'z_score_maps')
    
    if not os.path.exists(z_map_dir):
        continue
    
    z_maps = sorted(glob.glob(os.path.join(z_map_dir, '*.dscalar.nii')))
    
    for z_map_path in z_maps:
        contrast_name = os.path.basename(z_map_path).replace('.dscalar.nii', '')
        
        # Load z-score map
        z_img = nib.load(z_map_path)
        z_data = z_img.get_fdata()[0]
        
        # Extract subnetwork activations
        z_fpn_a = z_data[fpn_a_mask]
        z_fpn_b = z_data[fpn_b_mask]
        
        # ========== APPROACH A: Mean activation analysis ==========
        mean_a = np.mean(z_fpn_a)
        mean_b = np.mean(z_fpn_b)
        median_a = np.median(z_fpn_a)
        median_b = np.median(z_fpn_b)
        
        # Effect size
        cohens_d = calculate_effect_size(z_fpn_a, z_fpn_b)
        
        # Statistical tests
        _, p_ttest = stats.ttest_ind(z_fpn_a, z_fpn_b)
        mean_diff, p_perm = permutation_test(z_fpn_a, z_fpn_b)
        
        # ========== APPROACH B: Overlap analysis ==========
        overlap_metrics = calculate_overlap_metrics(z_data, fpn_a_mask, fpn_b_mask)
        
        # ========== Store results ==========
        result = {
            'subject': subject,
            'task': task_name,
            'contrast': contrast_name,
            'task_contrast': f"{task_name}_{contrast_name}",
            # Primary metrics (Approach A)
            'mean_fpn_a': mean_a,
            'mean_fpn_b': mean_b,
            'median_fpn_a': median_a,
            'median_fpn_b': median_b,
            'mean_diff_a_minus_b': mean_diff,
            'cohens_d': cohens_d,
            'abs_cohens_d': abs(cohens_d),
            # Statistical significance
            'p_ttest': p_ttest,
            'p_perm': p_perm,
            'significant': p_perm < 0.05,
            # Dominance classification
            'dominant_network': 'FPN_A' if mean_a > mean_b else 'FPN_B',
            'dominance_strength': abs(mean_diff),
            # Validation metrics (Approach B)
            **overlap_metrics
        }
        
        results.append(result)
        
        # Progress indicator
        sig = "***" if p_perm < 0.001 else "**" if p_perm < 0.01 else "*" if p_perm < 0.05 else ""
        print(f"{task_name:20s} | {contrast_name:40s} | "
              f"A={mean_a:+.2f} B={mean_b:+.2f} d={cohens_d:+.2f} p={p_perm:.3f}{sig}")

# ========== Save detailed results ==========
if not results:
    print("ERROR: No results generated!")
    sys.exit(1)

df = pd.DataFrame(results)

# Apply FDR correction
df['p_perm_fdr'] = false_discovery_control(df['p_perm'].values, method='bh')
df['significant_fdr'] = df['p_perm_fdr'] < 0.05

output_csv = os.path.join(output_dir, 'fpn_subnetwork_contrast_analysis.csv')
df.to_csv(output_csv, index=False)
print(f"\n✓ Detailed results saved: {output_csv}\n")

# ========== Generate summary statistics ==========
print(f"{'='*70}")
print("SUMMARY: Functional differentiation of FPN_A vs FPN_B")
print(f"{'='*70}\n")

print(f"Total contrasts analyzed: {len(df)}")
print(f"Significant differences (p<0.05): {df['significant'].sum()}")
print(f"Significant differences after FDR correction (p<0.05): {df['significant_fdr'].sum()}")
print(f"\nDominance distribution (all contrasts):")
print(df['dominant_network'].value_counts())

sig_df = df[df['significant_fdr']]
if len(sig_df) > 0:
    print(f"\nDominance distribution (significant only):")
    print(sig_df['dominant_network'].value_counts())
    
    # Top contrasts favoring each network
    print(f"\n{'='*70}")
    print("TOP 15 CONTRASTS FAVORING FPN_A (introspective/internal?)")
    print(f"{'='*70}")
    top_a = df.nlargest(15, 'mean_diff_a_minus_b')[
        ['task', 'contrast', 'mean_diff_a_minus_b', 'cohens_d', 'p_perm']
    ]
    print(top_a.to_string(index=False))
    
    print(f"\n{'='*70}")
    print("TOP 15 CONTRASTS FAVORING FPN_B (visuospatial/external?)")
    print(f"{'='*70}")
    top_b = df.nsmallest(15, 'mean_diff_a_minus_b')[
        ['task', 'contrast', 'mean_diff_a_minus_b', 'cohens_d', 'p_perm']
    ]
    print(top_b.to_string(index=False))

# ========== Validation: Compare Approach A vs Approach B ==========
print(f"\n{'='*70}")
print("VALIDATION: Correlation between mean activation and overlap")
print(f"{'='*70}\n")

corr_a = df[['mean_diff_a_minus_b', 'prop_a_pos_z2_0']].corr().iloc[0, 1]
corr_b = df[['mean_diff_a_minus_b', 'prop_b_pos_z2_0']].corr().iloc[0, 1]
print(f"Correlation (mean_diff vs overlap_A): r={corr_a:.3f}")
print(f"Correlation (mean_diff vs overlap_B): r={corr_b:.3f}")
print("→ High correlation validates that both approaches capture similar patterns")

# ========== Export for pattern analysis ==========
# Create a pivot table for clustering/visualization
pivot_data = df.pivot_table(
    index='task_contrast',
    values=['mean_diff_a_minus_b', 'cohens_d', 'p_perm'],
    aggfunc='first'
)
pivot_csv = os.path.join(output_dir, 'fpn_subnetwork_pivot_for_clustering.csv')
pivot_data.to_csv(pivot_csv)
print(f"\n✓ Pivot table for clustering saved: {pivot_csv}")

# Summary by task (aggregate across contrasts)
task_summary = df.groupby('task').agg({
    'mean_diff_a_minus_b': ['mean', 'std', 'count'],
    'cohens_d': 'mean',
    'significant': 'sum'
}).round(3)
task_summary.columns = ['_'.join(col) for col in task_summary.columns]
task_summary_csv = os.path.join(output_dir, 'fpn_subnetwork_task_summary.csv')
task_summary.to_csv(task_summary_csv)
print(f"✓ Task-level summary saved: {task_summary_csv}")

print(f"\n{'='*70}")
print("NEXT STEPS for pattern discovery:")
print(f"{'='*70}")
print("1. Load fpn_subnetwork_contrast_analysis.csv")
print("2. Manually annotate contrasts with cognitive categories")
print("3. Use hierarchical clustering on 'mean_diff_a_minus_b' to find patterns")
print("4. Test hypotheses: e.g., FPN_A → introspection, FPN_B → visuospatial")
print(f"\n✓ Analysis complete for subject {subject}!\n")
