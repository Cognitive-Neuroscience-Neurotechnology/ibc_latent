"""
Aggregate contrast x subnetwork results across all subjects.
"""

import os
import glob
import pandas as pd
import numpy as np
from scipy import stats

results_base = '/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_activation'
output_dir = os.path.join(results_base, 'group_analysis')
os.makedirs(output_dir, exist_ok=True)

# Find all subject CSV files
subject_csvs = sorted(glob.glob(os.path.join(results_base, 'sub-*/contrast_x_subnetwork_analysis.csv')))

if not subject_csvs:
    print("No subject results found!")
    exit(1)

print(f"Found {len(subject_csvs)} subjects")

# Load and concatenate all subjects
all_data = []
for csv_path in subject_csvs:
    subject = os.path.basename(os.path.dirname(csv_path))
    df = pd.read_csv(csv_path)
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)
print(f"Total rows: {len(combined_df)}")

# Save combined data
combined_csv = os.path.join(output_dir, 'all_subjects_combined.csv')
combined_df.to_csv(combined_csv, index=False)
print(f"✓ Combined data saved to: {combined_csv}")

# ========== Group-level statistics ==========
print(f"\n{'='*60}")
print("GROUP-LEVEL ANALYSIS")
print(f"{'='*60}")

# Group by task and contrast
group_stats = []

for (task, contrast), group in combined_df.groupby(['task', 'contrast']):
    n_subjects = len(group)
    
    # Average across subjects
    mean_a_avg = group['mean_subnet_a'].mean()
    mean_b_avg = group['mean_subnet_b'].mean()
    mean_diff_avg = group['mean_diff_a_minus_b'].mean()
    
    # Standard error
    mean_a_se = group['mean_subnet_a'].sem()
    mean_b_se = group['mean_subnet_b'].sem()
    mean_diff_se = group['mean_diff_a_minus_b'].sem()
    
    # One-sample t-test: is difference consistently different from zero?
    t_stat, p_val = stats.ttest_1samp(group['mean_diff_a_minus_b'], 0)
    
    # Effect size
    cohens_d_avg = group['cohens_d'].mean()
    
    # Overlap statistics
    prop_a_avg = group['prop_a_pos_z2_0'].mean()
    prop_b_avg = group['prop_b_pos_z2_0'].mean()
    
    group_stats.append({
        'task': task,
        'contrast': contrast,
        'n_subjects': n_subjects,
        'mean_subnet_a': mean_a_avg,
        'mean_subnet_a_se': mean_a_se,
        'mean_subnet_b': mean_b_avg,
        'mean_subnet_b_se': mean_b_se,
        'mean_diff_a_minus_b': mean_diff_avg,
        'mean_diff_se': mean_diff_se,
        'group_tstat': t_stat,
        'group_pval': p_val,
        'cohens_d_avg': cohens_d_avg,
        'prop_a_overlap_z2': prop_a_avg,
        'prop_b_overlap_z2': prop_b_avg,
    })

group_df = pd.DataFrame(group_stats)

# FDR correction for multiple comparisons
from statsmodels.stats.multitest import multipletests
_, group_df['group_pval_fdr'], _, _ = multipletests(group_df['group_pval'], method='fdr_bh')

# Save group statistics
group_csv = os.path.join(output_dir, 'group_level_statistics.csv')
group_df.to_csv(group_csv, index=False)
print(f"✓ Group statistics saved to: {group_csv}")

# Summary
significant_group = group_df[group_df['group_pval_fdr'] < 0.05]
print(f"\nTotal contrasts: {len(group_df)}")
print(f"Significant at FDR < 0.05: {len(significant_group)}")

if len(significant_group) > 0:
    print(f"\nTop 10 contrasts favoring Subnetwork A:")
    top_a = group_df.nlargest(10, 'mean_diff_a_minus_b')[['task', 'contrast', 'mean_diff_a_minus_b', 'group_pval_fdr', 'cohens_d_avg']]
    print(top_a.to_string(index=False))
    
    print(f"\nTop 10 contrasts favoring Subnetwork B:")
    top_b = group_df.nsmallest(10, 'mean_diff_a_minus_b')[['task', 'contrast', 'mean_diff_a_minus_b', 'group_pval_fdr', 'cohens_d_avg']]
    print(top_b.to_string(index=False))

print("\n✓ Group-level aggregation complete!")