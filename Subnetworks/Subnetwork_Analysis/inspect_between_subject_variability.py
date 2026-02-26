"""
Analyze cross-subject variability for the top 30 contrasts identified in group analysis.
Shows range, mean, and SD for each contrast across all 8 subjects.
This is ran on the results from /subnetwork_analysis_results/
"""

import pandas as pd
import numpy as np
import glob
import os

print(f"{'='*90}")
print("CROSS-SUBJECT VARIABILITY ANALYSIS FOR TOP 30 CONTRASTS")
print(f"{'='*90}\n")

# Load group-level stats to identify top contrasts
group_stats_path = '/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_activation/group_analysis/group_level_stats.csv'
group_stats = pd.read_csv(group_stats_path)

# Get top 15 contrasts for each network (by mean_diff_a_minus_b_mean)
# Most positive = FPN_A dominant
top_fpn_a = group_stats.nlargest(15, 'mean_diff_a_minus_b_mean')[
    ['task_contrast', 'task', 'contrast', 'mean_diff_a_minus_b_mean', 'cohens_d_mean']
].copy()
top_fpn_a['network'] = 'FPN_A'

# Most negative = FPN_B dominant
top_fpn_b = group_stats.nsmallest(15, 'mean_diff_a_minus_b_mean')[
    ['task_contrast', 'task', 'contrast', 'mean_diff_a_minus_b_mean', 'cohens_d_mean']
].copy()
top_fpn_b['network'] = 'FPN_B'

print("TOP 15 FPN_A DOMINANT CONTRASTS (most positive mean_diff):")
print(top_fpn_a[['task', 'contrast', 'mean_diff_a_minus_b_mean']].to_string(index=False))
print(f"\nTOP 15 FPN_B DOMINANT CONTRASTS (most negative mean_diff):")
print(top_fpn_b[['task', 'contrast', 'mean_diff_a_minus_b_mean']].to_string(index=False))

# Combine top contrasts
top_contrasts = pd.concat([top_fpn_a, top_fpn_b])

# Get all subject result files
subject_files = sorted(glob.glob('/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_activation/sub-*/fpn_subnetwork_contrast_analysis.csv'))
subjects = [os.path.basename(os.path.dirname(f)).replace('sub-', '') for f in subject_files]

print(f"\n\nFound {len(subjects)} subjects: {', '.join(subjects)}\n")

# Load all subject data
all_subject_data = []
for subj_file, subj_id in zip(subject_files, subjects):
    df = pd.read_csv(subj_file)
    df['subject'] = subj_id
    all_subject_data.append(df)

combined_df = pd.concat(all_subject_data, ignore_index=True)

# Analyze range for each top contrast
print(f"{'='*90}")
print("CROSS-SUBJECT VARIABILITY FOR TOP 30 CONTRASTS")
print(f"{'='*90}\n")

results = []

for network in ['FPN_A', 'FPN_B']:
    top_list = top_fpn_a if network == 'FPN_A' else top_fpn_b
    
    print(f"\n{'='*90}")
    print(f"{network} DOMINANT CONTRASTS (N=15, ranked by group mean)")
    print(f"{'='*90}\n")
    print(f"{'Rank':<5} {'Task':<20} {'Contrast':<40} {'Min':<8} {'Max':<8} {'Range':<8} {'Mean':<8} {'SD':<8} {'N'}")
    print("-"*90)
    
    for rank, (_, row) in enumerate(top_list.iterrows(), 1):
        contrast = row['task_contrast']
        task = row['task']
        contrast_name = row['contrast']
        
        # Get data for this contrast across subjects
        contrast_data = combined_df[combined_df['task_contrast'] == contrast]['mean_diff_a_minus_b']
        
        if len(contrast_data) > 0:
            min_val = contrast_data.min()
            max_val = contrast_data.max()
            range_val = max_val - min_val
            mean_val = contrast_data.mean()
            std_val = contrast_data.std()
            n_subjects = len(contrast_data)
            
            # Truncate names for display
            task_short = task[:19]
            contrast_short = contrast_name[:39]
            
            print(f"{rank:<5} {task_short:<20} {contrast_short:<40} {min_val:+7.3f} {max_val:+7.3f} {range_val:7.3f} {mean_val:+7.3f} {std_val:7.3f} {n_subjects:2d}")
            
            results.append({
                'rank': rank,
                'network': network,
                'task': task,
                'contrast': contrast_name,
                'task_contrast': contrast,
                'group_mean': row['mean_diff_a_minus_b_mean'],
                'group_cohens_d': row['cohens_d_mean'],
                'min': min_val,
                'max': max_val,
                'range': range_val,
                'mean': mean_val,
                'std': std_val,
                'cv': (std_val / abs(mean_val)) if abs(mean_val) > 0.001 else np.nan,  # coefficient of variation
                'n_subjects': n_subjects
            })
        else:
            print(f"{rank:<5} {task_short:<20} {contrast_short:<40} {'NO DATA':^50}")

# Save detailed summary
results_df = pd.DataFrame(results)
output_dir = '/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_activation/group_analysis'
output_file = os.path.join(output_dir, 'top30_cross_subject_variability.csv')
results_df.to_csv(output_file, index=False)
print(f"\n✓ Saved detailed results to: {output_file}")

# Summary statistics
print(f"\n{'='*90}")
print("SUMMARY STATISTICS")
print(f"{'='*90}\n")

for network in ['FPN_A', 'FPN_B']:
    network_data = results_df[results_df['network'] == network]
    
    print(f"\n{network} CONTRASTS:")
    print(f"  Average range across 15 contrasts: {network_data['range'].mean():.3f}")
    print(f"  Average SD across 15 contrasts: {network_data['std'].mean():.3f}")
    print(f"  Average coefficient of variation: {network_data['cv'].mean():.3f}")
    
    print(f"\n  Most variable contrast:")
    most_var = network_data.loc[network_data['range'].idxmax()]
    print(f"    {most_var['task']}: {most_var['contrast']}")
    print(f"    Range: {most_var['range']:.3f}, SD: {most_var['std']:.3f}")
    
    print(f"\n  Least variable contrast:")
    least_var = network_data.loc[network_data['range'].idxmin()]
    print(f"    {least_var['task']}: {least_var['contrast']}")
    print(f"    Range: {least_var['range']:.3f}, SD: {least_var['std']:.3f}")

# Consistency analysis: How many subjects show the same direction as group?
print(f"\n{'='*90}")
print("CONSISTENCY ANALYSIS")
print(f"{'='*90}\n")

for network in ['FPN_A', 'FPN_B']:
    network_contrasts = results_df[results_df['network'] == network]['task_contrast'].values
    
    print(f"\n{network} CONTRASTS:")
    print(f"{'Rank':<5} {'Task:Contrast':<50} {'Group':<8} {'Same Dir':<10}")
    print("-"*90)
    
    for rank, contrast in enumerate(network_contrasts, 1):
        contrast_data = combined_df[combined_df['task_contrast'] == contrast]['mean_diff_a_minus_b']
        group_mean = results_df[results_df['task_contrast'] == contrast]['group_mean'].iloc[0]
        
        # Count how many subjects show same direction as group
        if group_mean > 0:
            same_direction = (contrast_data > 0).sum()
        else:
            same_direction = (contrast_data < 0).sum()
        
        total = len(contrast_data)
        pct = (same_direction / total * 100) if total > 0 else 0
        
        task_contrast_short = contrast[:49]
        direction = "A>B" if group_mean > 0 else "B>A"
        
        print(f"{rank:<5} {task_contrast_short:<50} {direction:<8} {same_direction}/{total} ({pct:.0f}%)")

print(f"\n{'='*90}")
print("✓ Analysis complete!")
print(f"{'='*90}\n")