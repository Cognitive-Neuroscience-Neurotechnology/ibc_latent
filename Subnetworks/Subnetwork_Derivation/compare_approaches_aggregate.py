"""
Aggregate comparison results from compare_approaches.py across all subjects.
Generates group-level statistics and visualizations.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats

# Configuration
working_dir = Path('/ptmp/hmueller2/Downloads')
comparison_dir = working_dir / 'subnetworks' / 'comparison'
output_dir = working_dir / 'subnetworks' / 'comparison' / 'group_analysis'
output_dir.mkdir(parents=True, exist_ok=True)

subjects_file = working_dir / 'subjects_resting.txt'
subjects = [line.strip().split()[0].replace('sub-', '') for line in open(subjects_file)]
k = 2  # Number of clusters

print(f"Aggregating comparison results for {len(subjects)} subjects (k={k})")
print("="*70)

# ============================================================================
# 1. PARSE LOG FILES - Extract metrics from each subject
# ============================================================================
print("\n[1/7] Parsing subject-level log files...")

metrics_data = []
for subject in subjects:
    log_file = comparison_dir / f'sub-{subject}' / f'comparison_k{k}_log.txt'
    if not log_file.exists():
        print(f"  ⚠ Missing log file for sub-{subject}")
        continue
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract metrics using simple string parsing
    metrics = {'subject': subject}
    
    # Total FPN vertices
    if 'Total FPN vertices:' in content:
        metrics['fpn_vertices'] = int(content.split('Total FPN vertices:')[1].split('\n')[0].strip())
    
    # Valid labeled vertices
    if 'Valid labeled vertices:' in content:
        metrics['valid_vertices'] = int(content.split('Valid labeled vertices:')[1].split('\n')[0].strip())
    
    # Agreement metrics
    if 'Adjusted Rand Index:' in content:
        metrics['ari'] = float(content.split('Adjusted Rand Index:')[1].split('\n')[0].strip())
    
    if 'Normalized Mutual Information:' in content:
        metrics['nmi'] = float(content.split('Normalized Mutual Information:')[1].split('\n')[0].strip())
    
    # Agreement percentage
    if 'Agreement:' in content:
        agree_line = content.split('Agreement:')[1].split('\n')[0]
        if '(' in agree_line and '%' in agree_line:
            metrics['agreement_pct'] = float(agree_line.split('(')[1].split('%')[0].strip())
    
    # Profile correlations (for each cluster)
    for cluster_id in range(1, k+1):
        pattern = f'Cluster {cluster_id}: Profile correlation = '
        if pattern in content:
            corr_line = content.split(pattern)[1].split('\n')[0]
            corr_val = float(corr_line.split()[0])
            metrics[f'profile_corr_cluster{cluster_id}'] = corr_val
    
    metrics_data.append(metrics)

# Create DataFrame
df_metrics = pd.DataFrame(metrics_data)
print(f"  ✓ Loaded metrics for {len(df_metrics)} subjects")

# Save aggregated metrics
metrics_csv = output_dir / f'aggregated_metrics_k{k}.csv'
df_metrics.to_csv(metrics_csv, index=False)
print(f"  ✓ Saved: {metrics_csv}")

# ============================================================================
# 2. SUMMARY STATISTICS
# ============================================================================
print("\n[2/7] Computing summary statistics...")

summary_stats = df_metrics.describe()
print("\nSummary Statistics:")
print(summary_stats)

# Save summary
summary_csv = output_dir / f'summary_statistics_k{k}.csv'
summary_stats.to_csv(summary_csv)
print(f"  ✓ Saved: {summary_csv}")

# ============================================================================
# 3. AGREEMENT METRICS VISUALIZATION
# ============================================================================
print("\n[3/7] Creating agreement metrics visualizations...")

# Single comprehensive plot showing per-subject metrics
fig, ax1 = plt.subplots(figsize=(14, 7))

subjects_with_data = df_metrics['subject'].values
agreement_vals = df_metrics['agreement_pct'].values
ari_vals = df_metrics['ari'].values

# Bar plot for agreement percentage
x_pos = np.arange(len(subjects_with_data))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(subjects_with_data)))
bars = ax1.bar(x_pos, agreement_vals, color=colors, 
               edgecolor='black', linewidth=1.5, alpha=0.7, label='Vertex Agreement (%)')

# Mean line for agreement
ax1.axhline(agreement_vals.mean(), color='green', linestyle='--', linewidth=2.5, 
            label=f'Mean Agreement: {agreement_vals.mean():.1f}%', alpha=0.8, zorder=2)

# Format left y-axis (agreement) - BLACK
ax1.set_xlabel('Subject', fontsize=14, fontweight='bold')
ax1.set_ylabel('Vertex Agreement (%)', fontsize=14, fontweight='bold')
ax1.tick_params(axis='y', labelsize=12)
ax1.set_ylim(0, 100)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'sub-{s}' for s in subjects_with_data], rotation=45, ha='right', fontsize=11)
ax1.grid(alpha=0.3, axis='y', linestyle=':')

# Create second y-axis for ARI
ax2 = ax1.twinx()
ax2.scatter(x_pos, ari_vals, s=200, color='red', marker='D', 
            edgecolors='darkred', linewidth=2, alpha=0.9, 
            label='Adjusted Rand Index', zorder=3)

# Mean line for ARI
ax2.axhline(ari_vals.mean(), color='red', linestyle='--', linewidth=2.5, 
            label=f'Mean ARI: {ari_vals.mean():.3f}', alpha=0.8, zorder=2)

# Format right y-axis (ARI)
ax2.set_ylabel('Adjusted Rand Index', fontsize=14, fontweight='bold', color='red')
ax2.tick_params(axis='y', labelcolor='red', labelsize=12)
ax2.set_ylim(0, 1)
ax2.grid(alpha=0.3, axis='y', linestyle=':', color='red')

# Title and combined legend
ax1.set_title('Agreement Metrics: Communities-based vs Vertices-based K-means\n' + 
              f'Spatial Agreement (bars) and Adjusted Rand Index (diamonds) per Subject (N={len(subjects_with_data)})',
              fontsize=15, fontweight='bold', pad=20)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
           fontsize=12, framealpha=0.95, edgecolor='black', fancybox=True)

plt.tight_layout()
agreement_plot = output_dir / f'agreement_metrics_distribution_k{k}.png'
plt.savefig(agreement_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {agreement_plot}")

# ============================================================================
# 4. CONNECTIVITY PROFILE CORRELATIONS
# ============================================================================
print("\n[4/7] Analyzing connectivity profile correlations...")

profile_corr_cols = [col for col in df_metrics.columns if 'profile_corr_cluster' in col]
if profile_corr_cols:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Box plot for each cluster
    profile_data = df_metrics[profile_corr_cols].values
    positions = np.arange(len(profile_corr_cols))
    bp = ax.boxplot(profile_data, positions=positions, widths=0.6, patch_artist=True,
                     showmeans=True, meanline=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    # Overlay individual subjects
    for i, col in enumerate(profile_corr_cols):
        y_data = df_metrics[col].dropna()
        x_jitter = np.random.normal(i, 0.04, size=len(y_data))
        ax.scatter(x_jitter, y_data, alpha=0.6, s=60, color='darkblue', edgecolors='black', linewidth=0.5, zorder=3)
    
    ax.set_xticks(positions)
    ax.set_xticklabels([f'Cluster {i+1}' for i in range(len(profile_corr_cols))], fontsize=12, fontweight='bold')
    ax.set_ylabel('Profile Correlation (r)', fontsize=12, fontweight='bold')
    ax.set_title('Connectivity Profile Similarity Between Approaches\n(Correlation across 19 LSNs)', 
                 fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    profile_plot = output_dir / f'profile_correlations_k{k}.png'
    plt.savefig(profile_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {profile_plot}")

# ============================================================================
# 5. AGGREGATE CONNECTIVITY PROFILES
# ============================================================================
print("\n[5/7] Aggregating connectivity profiles across subjects...")

network_names = ["Parietal DMN", "Anterolateral DMN", "Dorsolateral DMN", "Retrosplenial DMN",
                 "Visual Lateral", "Visual Dorsal", "Visual V5", "Visual V1", "DAN", "DAN II",
                 "Language", "Salience", "Cingulo Opercular", "Medial Parietal",
                 "Somatomotor Hand", "Somatomotor Face", "Somatomotor Foot", "Auditory", "Somato Cognitive Action"]

# Create cluster names (FPNA, FPNB, etc.)
cluster_names = [f'FPN{chr(65 + i)}' for i in range(k)]  # FPNA, FPNB, FPNC, ...

# Collect all subject profiles
all_profiles = {}
for cluster_name in cluster_names:
    all_profiles[f'Comm_{cluster_name}'] = []
    all_profiles[f'Vert_{cluster_name}'] = []

for subject in subjects:
    profile_csv = comparison_dir / f'sub-{subject}' / f'connectivity_profiles_k{k}.csv'
    if not profile_csv.exists():
        print(f"  ⚠ Missing profile file for sub-{subject}")
        continue
    
    df_prof = pd.read_csv(profile_csv)
    for col in all_profiles.keys():
        if col in df_prof.columns:
            all_profiles[col].append(df_prof[col].values)

# Compute mean and SEM across subjects
aggregated_profiles = {'Network': network_names}
for key, profiles_list in all_profiles.items():
    if len(profiles_list) > 0:
        profiles_array = np.array(profiles_list)  # (n_subjects, 19 networks)
        aggregated_profiles[f'{key}_mean'] = np.nanmean(profiles_array, axis=0)
        aggregated_profiles[f'{key}_sem'] = np.nanstd(profiles_array, axis=0) / np.sqrt(len(profiles_list))

df_agg_profiles = pd.DataFrame(aggregated_profiles)
agg_profile_csv = output_dir / f'aggregated_connectivity_profiles_k{k}.csv'
df_agg_profiles.to_csv(agg_profile_csv, index=False)
print(f"  ✓ Saved: {agg_profile_csv}")

# Plot group-average profiles
fig, axes = plt.subplots(1, k, figsize=(7*k, 6))
if k == 1:
    axes = [axes]

for cluster_idx in range(k):
    cluster_name = cluster_names[cluster_idx]
    
    comm_col = f'Comm_{cluster_name}_mean'
    vert_col = f'Vert_{cluster_name}_mean'
    comm_sem_col = f'Comm_{cluster_name}_sem'
    vert_sem_col = f'Vert_{cluster_name}_sem'
    
    x = np.arange(19)
    width = 0.35
    
    axes[cluster_idx].bar(x - width/2, df_agg_profiles[comm_col], width, 
                          yerr=df_agg_profiles[comm_sem_col], label='Communities-based', 
                          alpha=0.8, color='steelblue', capsize=3)
    axes[cluster_idx].bar(x + width/2, df_agg_profiles[vert_col], width, 
                          yerr=df_agg_profiles[vert_sem_col], label='Vertices-based', 
                          alpha=0.8, color='coral', capsize=3)
    
    axes[cluster_idx].set_title(f'{cluster_name} (N={len(subjects)} subjects)', fontsize=14, fontweight='bold')
    axes[cluster_idx].set_ylabel('Mean Correlation with LSN', fontsize=12, fontweight='bold')
    axes[cluster_idx].set_xticks(x)
    axes[cluster_idx].set_xticklabels(network_names, rotation=45, ha='right', fontsize=9)
    axes[cluster_idx].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[cluster_idx].legend()
    axes[cluster_idx].grid(alpha=0.3)
    axes[cluster_idx].set_ylim(-0.3, 0.6)

plt.tight_layout()
group_profile_plot = output_dir / f'group_connectivity_profiles_k{k}.png'
plt.savefig(group_profile_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {group_profile_plot}")

# ============================================================================
# 6. CLUSTER SIZE ANALYSIS
# ============================================================================
print("\n[6/7] Analyzing cluster size differences...")

# Parse contingency tables to extract cluster sizes
size_data = []
for subject in subjects:
    log_file = comparison_dir / f'sub-{subject}' / f'comparison_k{k}_log.txt'
    if not log_file.exists():
        continue
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract cluster sizes from the size comparison section
    if 'Cluster Size Comparison' in content:
        size_section = content.split('Cluster Size Comparison')[1].split('---')[0]
        for cluster_idx in range(k):
            cluster_letter = chr(65 + cluster_idx)
            cluster_name = f'FPN{cluster_letter}'
            
            # Look for lines like "FPNA  5234  5189"
            if cluster_name in size_section:
                line = [l for l in size_section.split('\n') if cluster_name in l][0]
                parts = line.split()
                if len(parts) >= 3:
                    comm_size = int(parts[1])
                    vert_size = int(parts[2])
                    size_data.append({
                        'subject': subject,
                        'cluster': cluster_name,
                        'communities_size': comm_size,
                        'vertices_size': vert_size,
                        'size_diff': comm_size - vert_size
                    })

df_sizes = pd.DataFrame(size_data)
if not df_sizes.empty:
    size_csv = output_dir / f'cluster_sizes_k{k}.csv'
    df_sizes.to_csv(size_csv, index=False)
    print(f"  ✓ Saved: {size_csv}")
    
    # Visualize size differences
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Absolute sizes
    for cluster_name in df_sizes['cluster'].unique():
        cluster_data = df_sizes[df_sizes['cluster'] == cluster_name]
        axes[0].scatter(cluster_data['communities_size'], cluster_data['vertices_size'], 
                       s=100, alpha=0.6, label=cluster_name, edgecolors='black', linewidth=1)
    
    # Identity line
    max_size = max(df_sizes['communities_size'].max(), df_sizes['vertices_size'].max())
    axes[0].plot([0, max_size], [0, max_size], 'k--', linewidth=2, alpha=0.5)
    axes[0].set_xlabel('Communities-based Size (vertices)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Vertices-based Size (vertices)', fontsize=12, fontweight='bold')
    axes[0].set_title('Cluster Size Comparison', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Size differences
    for cluster_name in df_sizes['cluster'].unique():
        cluster_data = df_sizes[df_sizes['cluster'] == cluster_name]
        axes[1].scatter(cluster_data['subject'], cluster_data['size_diff'], 
                       s=100, alpha=0.6, label=cluster_name, edgecolors='black', linewidth=1)
    
    axes[1].axhline(0, color='black', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Subject', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Size Difference\n(Communities - Vertices)', fontsize=12, fontweight='bold')
    axes[1].set_title('Cluster Size Differences by Subject', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    size_plot = output_dir / f'cluster_size_analysis_k{k}.png'
    plt.savefig(size_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {size_plot}")

# ============================================================================
# 7. GENERATE SUMMARY REPORT
# ============================================================================
print("\n[7/7] Generating summary report...")

report_file = output_dir / f'comparison_summary_report_k{k}.txt'
with open(report_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write(f"COMPARISON SUMMARY: Communities vs Vertices K-means (k={k})\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Number of subjects analyzed: {len(df_metrics)}\n")
    f.write(f"Subjects: {', '.join(subjects)}\n\n")
    
    f.write("--- AGREEMENT METRICS ---\n")
    f.write(f"Adjusted Rand Index:\n")
    f.write(f"  Mean ± SD: {df_metrics['ari'].mean():.3f} ± {df_metrics['ari'].std():.3f}\n")
    f.write(f"  Range: [{df_metrics['ari'].min():.3f}, {df_metrics['ari'].max():.3f}]\n\n")
    
    f.write(f"Normalized Mutual Information:\n")
    f.write(f"  Mean ± SD: {df_metrics['nmi'].mean():.3f} ± {df_metrics['nmi'].std():.3f}\n")
    f.write(f"  Range: [{df_metrics['nmi'].min():.3f}, {df_metrics['nmi'].max():.3f}]\n\n")
    
    f.write(f"Spatial Agreement (%):\n")
    f.write(f"  Mean ± SD: {df_metrics['agreement_pct'].mean():.1f} ± {df_metrics['agreement_pct'].std():.1f}\n")
    f.write(f"  Range: [{df_metrics['agreement_pct'].min():.1f}%, {df_metrics['agreement_pct'].max():.1f}%]\n\n")
    
    f.write("--- CONNECTIVITY PROFILE CORRELATIONS ---\n")
    for col in profile_corr_cols:
        cluster_id = col.split('cluster')[1]
        vals = df_metrics[col].dropna()
        f.write(f"Cluster {cluster_id}:\n")
        f.write(f"  Mean ± SD: {vals.mean():.3f} ± {vals.std():.3f}\n")
        f.write(f"  Range: [{vals.min():.3f}, {vals.max():.3f}]\n\n")
    
    f.write("--- INTERPRETATION ---\n")
    f.write("Higher ARI/NMI (closer to 1) = more similar clustering solutions\n")
    f.write("Higher profile correlation = more similar connectivity patterns\n")
    f.write("Higher spatial agreement % = more vertices assigned to same cluster\n\n")
    
    # Statistical test: Are approaches significantly different?
    if 'profile_corr_cluster1' in df_metrics.columns:
        t_stat, p_val = stats.ttest_1samp(df_metrics['profile_corr_cluster1'].dropna(), 1.0, alternative='less')
        f.write(f"One-sample t-test (Cluster 1 profile correlation vs perfect agreement r=1.0):\n")
        f.write(f"  t = {t_stat:.3f}, p = {p_val:.4f}\n")
        if p_val < 0.001:
            f.write("  *** Approaches produce significantly different connectivity profiles\n")
        elif p_val < 0.05:
            f.write("  ** Approaches produce moderately different connectivity profiles\n")
        else:
            f.write("  Approaches produce similar connectivity profiles\n")

print(f"  ✓ Saved: {report_file}")

# ============================================================================
print("\n" + "="*70)
print(f"✓ Aggregation complete. Results saved to: {output_dir}")
print("="*70)
print("\nGenerated outputs:")
print(f"  1. {metrics_csv}")
print(f"  2. {summary_csv}")
print(f"  3. {agreement_plot}")
print(f"  4. {profile_plot}")
print(f"  5. {agg_profile_csv}")
print(f"  6. {group_profile_plot}")
if not df_sizes.empty:
    print(f"  7. {size_csv}")
    print(f"  8. {size_plot}")
    print(f"  9. {report_file}")
else:
    print(f"  7. {report_file}")
    print("  ⚠ Cluster size analysis skipped (no size data found)")