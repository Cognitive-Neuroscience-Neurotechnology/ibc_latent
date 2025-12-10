"""
Map FPN subnetwork activation differences to cognitive domains using Cognitive Atlas tags.
Aggregates contrast-level results by cognitive concept to identify functional specialization.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 1. LOAD DATA ==========
print("="*60)
print("COGNITIVE DOMAIN MAPPING FOR FPN SUBNETWORKS")
print("="*60)

# Load group-level FPN subnetwork results
results_base = '/ptmp/hmueller2/Downloads/subnetwork_analysis_results/group_analysis'
fpn_results = pd.read_csv(os.path.join(results_base, 'group_level_stats.csv'))
print(f"\n[1/6] Loaded FPN results: {len(fpn_results)} contrasts")
print(f"  Columns: {list(fpn_results.columns)}")

# Load contrast metadata with cognitive tags
contrast_metadata = pd.read_csv('/home/hmueller2/ibc_code/ibc_latent/Data Info/all_contrasts.tsv', sep='\t')
print(f"[1/6] Loaded contrast metadata: {len(contrast_metadata)} contrasts")
print(f"  Columns: {list(contrast_metadata.columns)}")

# ========== 2. MERGE FPN RESULTS WITH COGNITIVE TAGS ==========
print("\n[2/6] Merging FPN results with cognitive tags...")

# Merge on task and contrast
merged = pd.merge(
    fpn_results,
    contrast_metadata[['task', 'contrast', 'tags', 'pretty name']],
    on=['task', 'contrast'],
    how='left'
)

print(f"  Matched {merged['tags'].notna().sum()} / {len(merged)} contrasts")

# After the merge, add this diagnostic:
print("\n[DEBUG] Investigating missing tags...")
missing_tags = merged[merged['tags'].isna()]
if len(missing_tags) > 0:
    print(f"  WARNING: {len(missing_tags)} contrasts have no tags:")
    for _, row in missing_tags.head(10).iterrows():
        print(f"    FPN results: task='{row['task']}', contrast='{row['contrast']}'")
        
        # Check if similar entries exist in metadata
        task_matches = contrast_metadata[contrast_metadata['task'].str.contains(row['task'], case=False, na=False)]
        print(f"      → Found {len(task_matches)} tasks matching '{row['task']}' in metadata")
        
        if len(task_matches) > 0:
            contrast_matches = task_matches[task_matches['contrast'] == row['contrast']]
            if len(contrast_matches) == 0:
                print(f"      → Available contrasts in metadata for this task:")
                print(f"        {list(task_matches['contrast'].unique()[:5])}")

# ========== 3. PARSE COGNITIVE TAGS ==========
print("\n[3/6] Parsing cognitive tags...")

def parse_tags(tag_str):
    """Convert string representation of list to actual list."""
    if pd.isna(tag_str):
        return []
    try:
        # Handle string representation of Python lists: ['tag1','tag2']
        tags = ast.literal_eval(tag_str)
        return tags if isinstance(tags, list) else []
    except:
        # Fallback: split by comma and clean
        return [t.strip().strip("'\"") for t in str(tag_str).strip('[]').split(',') if t.strip()]

merged['parsed_tags'] = merged['tags'].apply(parse_tags)

# Count tags per contrast
merged['n_tags'] = merged['parsed_tags'].apply(len)
print(f"  Average tags per contrast: {merged['n_tags'].mean():.1f}")
print(f"  Contrasts with tags: {(merged['n_tags'] > 0).sum()}")

# Show example of parsed tags
sample_with_tags = merged[merged['n_tags'] > 0].iloc[0]
print(f"  Example: {sample_with_tags['task']} - {sample_with_tags['contrast']}")
print(f"    Tags: {sample_with_tags['parsed_tags']}")

# ========== 4. AGGREGATE BY COGNITIVE DOMAIN ==========
print("\n[4/6] Aggregating activations by cognitive domain...")

# Create domain-level statistics
domain_data = defaultdict(lambda: {
    'diff_values': [],
    'cohens_d_values': [],
    'contrasts': [],
    'tasks': []
})

for _, row in merged.iterrows():
    if row['n_tags'] == 0:
        continue
    
    for tag in row['parsed_tags']:
        # Use the correct column names from group_level_stats.csv
        domain_data[tag]['diff_values'].append(row['mean_diff_a_minus_b_mean'])
        domain_data[tag]['cohens_d_values'].append(row['cohens_d_mean'])
        domain_data[tag]['contrasts'].append(row['contrast'])
        domain_data[tag]['tasks'].append(row['task'])

print(f"  Collected data for {len(domain_data)} unique cognitive tags")

# Compute domain-level statistics
domain_results = []

for domain, data in domain_data.items():
    diff = np.array(data['diff_values'])
    cohens_d = np.array(data['cohens_d_values'])
    
    n_contrasts = len(diff)
    
    if n_contrasts < 2:  # Need at least 2 contrasts for meaningful stats
        continue
    
    # Aggregate statistics
    mean_diff = np.mean(diff)
    se_diff = np.std(diff) / np.sqrt(n_contrasts)
    
    # One-sample t-test: is mean difference from zero?
    t_stat, p_val = stats.ttest_1samp(diff, 0)
    
    # Effect size (mean Cohen's d across contrasts)
    mean_cohens_d = np.mean(cohens_d)
    
    # Consistency: proportion of contrasts favoring same network
    prop_fpna_favored = (diff > 0).sum() / n_contrasts
    prop_fpnb_favored = (diff < 0).sum() / n_contrasts
    consistency = max(prop_fpna_favored, prop_fpnb_favored)
    
    domain_results.append({
        'cognitive_domain': domain,
        'n_contrasts': n_contrasts,
        'n_tasks': len(set(data['tasks'])),
        'mean_diff_a_minus_b': mean_diff,
        'se_diff': se_diff,
        't_statistic': t_stat,
        'p_value': p_val,
        'mean_cohens_d': mean_cohens_d,
        'consistency': consistency,
        'fpna_favored_pct': prop_fpna_favored * 100,
        'fpnb_favored_pct': prop_fpnb_favored * 100
    })

domain_df = pd.DataFrame(domain_results)

# FDR correction
from statsmodels.stats.multitest import multipletests
_, domain_df['p_fdr'], _, _ = multipletests(domain_df['p_value'], method='fdr_bh')

# Add absolute Cohen's d for convenience
domain_df['abs_cohens_d'] = domain_df['mean_cohens_d'].abs()

print(f"  Found {len(domain_df)} cognitive domains with ≥2 contrasts")

# ========== 5. CREATE DIFFERENT RANKING SCHEMES ==========
print("\n[5/6] Creating different ranking schemes...")

# ORIGINAL: Sort by raw Cohen's d (effect size)
domain_df_original = domain_df.copy()
domain_df_original = domain_df_original.sort_values('mean_cohens_d', ascending=False)

# OPTION A: Composite score (significance + effect size + consistency)
domain_df_composite = domain_df.copy()
domain_df_composite['rank_score'] = (
    # Negative log p-value (higher = more significant)
    -np.log10(domain_df_composite['p_fdr'] + 1e-10) * 0.4 +  # 40% weight
    # Absolute effect size
    domain_df_composite['abs_cohens_d'] * 100 * 0.35 +  # 35% weight
    # Consistency (proportion favoring dominant network)
    domain_df_composite['consistency'] * 100 * 0.25  # 25% weight
)
domain_df_composite = domain_df_composite.sort_values('rank_score', ascending=False)
print(f"  Composite score - Top domain: {domain_df_composite.iloc[0]['cognitive_domain']} (score={domain_df_composite.iloc[0]['rank_score']:.2f})")

# OPTION C: High-quality domains (stringent filtering)
domain_df_highquality = domain_df[
    (domain_df['p_fdr'] < 0.05) &  # Significant
    (domain_df['abs_cohens_d'] > 0.2) &  # Medium effect
    (domain_df['consistency'] > 0.7) &  # 70%+ consistency
    (domain_df['n_contrasts'] >= 5)  # At least 5 contrasts
].copy()
domain_df_highquality = domain_df_highquality.sort_values('abs_cohens_d', ascending=False)
print(f"  High-quality domains (filtered): {len(domain_df_highquality)}")

# ========== 6. STATISTICAL SUMMARY ==========
print("\n[6/9] Statistical summary:")
print(f"  Domains with FDR < 0.05: {(domain_df['p_fdr'] < 0.05).sum()}")
print(f"  Domains with |Cohen's d| > 0.3: {(domain_df['abs_cohens_d'] > 0.3).sum()}")
print(f"  Domains with consistency > 80%: {(domain_df['consistency'] > 0.8).sum()}")

# ========== 7. SAVE RESULTS ==========
output_dir = '/ptmp/hmueller2/Downloads/subnetwork_analysis_results/cognitive_atlas'
os.makedirs(output_dir, exist_ok=True)

# Save ORIGINAL ranking (by Cohen's d)
domain_csv_original = os.path.join(output_dir, 'cognitive_domain_fpn_profiles_by_effect_size.csv')
domain_df_original.to_csv(domain_csv_original, index=False)
print(f"\n[7/9] ✓ Saved ORIGINAL (effect size) to: {domain_csv_original}")

# Save COMPOSITE ranking
domain_csv_composite = os.path.join(output_dir, 'cognitive_domain_fpn_profiles_composite_score.csv')
domain_df_composite.to_csv(domain_csv_composite, index=False)
print(f"[7/9] ✓ Saved COMPOSITE (balanced) to: {domain_csv_composite}")

# Save HIGH-QUALITY filtering
domain_csv_highquality = os.path.join(output_dir, 'cognitive_domain_fpn_profiles_high_quality.csv')
domain_df_highquality.to_csv(domain_csv_highquality, index=False)
print(f"[7/9] ✓ Saved HIGH-QUALITY (filtered) to: {domain_csv_highquality}")

# Save contrast-level merged data
merged_csv = os.path.join(output_dir, 'contrasts_with_cognitive_tags.csv')
merged.to_csv(merged_csv, index=False)
print(f"[7/9] ✓ Saved merged contrast data to: {merged_csv}")

# ========== 8. PRINT TOP FINDINGS FOR EACH APPROACH ==========
print("\n" + "="*60)
print("ORIGINAL RANKING (BY EFFECT SIZE)")
print("="*60)
print("TOP 10 DOMAINS FAVORING FPN-A:")
top_fpna_orig = domain_df_original.nlargest(10, 'mean_diff_a_minus_b')
print(top_fpna_orig[['cognitive_domain', 'n_contrasts', 'mean_cohens_d', 'p_fdr']].to_string(index=False))
print("\nTOP 10 DOMAINS FAVORING FPN-B:")
top_fpnb_orig = domain_df_original.nsmallest(10, 'mean_diff_a_minus_b')
print(top_fpnb_orig[['cognitive_domain', 'n_contrasts', 'mean_cohens_d', 'p_fdr']].to_string(index=False))

print("\n" + "="*60)
print("COMPOSITE SCORE RANKING (BALANCED)")
print("="*60)
print("TOP 10 DOMAINS:")
top_composite = domain_df_composite.nlargest(10, 'rank_score')
print(top_composite[['cognitive_domain', 'rank_score', 'mean_cohens_d', 'p_fdr', 'consistency']].to_string(index=False))

print("\n" + "="*60)
print("HIGH-QUALITY DOMAINS (STRINGENT FILTERING)")
print("="*60)
print(f"Total domains meeting criteria: {len(domain_df_highquality)}")
if len(domain_df_highquality) > 0:
    print("\nTOP 10:")
    print(domain_df_highquality.head(10)[['cognitive_domain', 'n_contrasts', 'mean_cohens_d', 'p_fdr', 'consistency']].to_string(index=False))

# ========== 9. VISUALIZATIONS FOR EACH APPROACH ==========
print("\n[8/9] Generating visualizations...")

def create_bar_plot(df, title_suffix, filename_suffix, top_n=20):
    """Create horizontal bar plot of top domains."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Select top domains
    top_domains = df.head(top_n)
    
    # Create color based on FPN preference
    colors = ['#d62728' if d > 0 else '#1f77b4' for d in top_domains['mean_diff_a_minus_b']]
    
    ax.barh(range(len(top_domains)), top_domains['mean_cohens_d'], color=colors)
    ax.set_yticks(range(len(top_domains)))
    ax.set_yticklabels(top_domains['cognitive_domain'], fontsize=9)
    ax.set_xlabel("Cohen's d (FPN-A - FPN-B)", fontsize=11)
    ax.set_title(f"Top {top_n} Cognitive Domains - {title_suffix}", fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    # Add significance markers
    for i, (_, row) in enumerate(top_domains.iterrows()):
        if row['p_fdr'] < 0.001:
            marker = '***'
        elif row['p_fdr'] < 0.01:
            marker = '**'
        elif row['p_fdr'] < 0.05:
            marker = '*'
        else:
            marker = ''
        
        if marker:
            x_pos = row['mean_cohens_d'] + (0.05 if row['mean_cohens_d'] > 0 else -0.05)
            ax.text(x_pos, i, marker, va='center', fontsize=12, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', label='FPN-A favored'),
        Patch(facecolor='#1f77b4', label='FPN-B favored')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'top_domains_{filename_suffix}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path

# Plot 1: ORIGINAL (by effect size)
plot1 = create_bar_plot(domain_df_original, "Sorted by Effect Size", "effect_size")
print(f"  ✓ Saved: {plot1}")

# Plot 2: COMPOSITE (by balanced score)
plot2 = create_bar_plot(domain_df_composite, "Composite Score Ranking", "composite_score")
print(f"  ✓ Saved: {plot2}")

# Plot 3: HIGH-QUALITY (filtered)
if len(domain_df_highquality) >= 10:
    plot3 = create_bar_plot(domain_df_highquality, "High-Quality Domains Only", "high_quality", top_n=min(20, len(domain_df_highquality)))
    print(f"  ✓ Saved: {plot3}")
else:
    print(f"  ⚠ Skipped high-quality plot (only {len(domain_df_highquality)} domains)")

# Plot 4: Scatter plot - Effect size vs. Consistency (using COMPOSITE data)
fig, ax = plt.subplots(figsize=(10, 8))

# Color by significance
sig_mask = domain_df_composite['p_fdr'] < 0.05
colors_sig = ['#d62728' if sig else '#999999' for sig in sig_mask]

ax.scatter(domain_df_composite['mean_cohens_d'], domain_df_composite['consistency'] * 100,
           s=domain_df_composite['n_contrasts'] * 3, c=colors_sig, alpha=0.6, edgecolors='black', linewidth=0.5)

ax.set_xlabel("Cohen's d (FPN-A - FPN-B)", fontsize=11)
ax.set_ylabel("Consistency (%)", fontsize=11)
ax.set_title("Cognitive Domain Effect Size vs. Consistency\n(size = number of contrasts)", 
             fontsize=12, fontweight='bold')
ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(50, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(alpha=0.3)

# Add labels for top composite score domains
top_composite_labels = domain_df_composite.nlargest(8, 'rank_score')
for _, row in top_composite_labels.iterrows():
    ax.annotate(row['cognitive_domain'], 
                xy=(row['mean_cohens_d'], row['consistency'] * 100),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.7)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d62728', label='Significant (FDR < 0.05)'),
    Patch(facecolor='#999999', label='Not significant')
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True)

plt.tight_layout()
plot4_path = os.path.join(output_dir, 'domain_effect_vs_consistency.png')
plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {plot4_path}")

# Plot 5: Comparison of ranking methods
fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)

# Get top 15 from each method
top_original = domain_df_original.head(15)
top_composite = domain_df_composite.head(15)
top_highqual = domain_df_highquality.head(15) if len(domain_df_highquality) >= 15 else domain_df_highquality

for ax, data, title in zip(axes, 
                           [top_original, top_composite, top_highqual],
                           ['Effect Size', 'Composite Score', 'High-Quality']):
    colors = ['#d62728' if d > 0 else '#1f77b4' for d in data['mean_diff_a_minus_b']]
    ax.barh(range(len(data)), data['mean_cohens_d'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data['cognitive_domain'], fontsize=8)
    ax.set_xlabel("Cohen's d", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)

plt.suptitle("Comparison of Ranking Methods - Top 15 Domains", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plot5_path = os.path.join(output_dir, 'ranking_method_comparison.png')
plt.savefig(plot5_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {plot5_path}")

def create_violin_plots_separate(df, merged_df, output_dir, top_n=10):
    """Create separate violin plots for top FPN-A and FPN-B domains with subject-level variability."""
    
    # Load subject-level data
    results_base = '/ptmp/hmueller2/Downloads/subnetwork_analysis_results'
    subject_files = []
    
    print(f"    Loading subject-level data for violin plots...")
    for subject_dir in os.listdir(results_base):
        if subject_dir.startswith('sub-'):
            for task in os.listdir(os.path.join(results_base, subject_dir)):
                task_path = os.path.join(results_base, subject_dir, task)
                if os.path.isdir(task_path):
                    stats_file = os.path.join(task_path, 'subnetwork_stats.csv')
                    if os.path.exists(stats_file):
                        try:
                            df_subj = pd.read_csv(stats_file)
                            df_subj['subject'] = subject_dir.replace('sub-', '')
                            subject_files.append(df_subj)
                        except Exception as e:
                            pass
    
    if len(subject_files) > 0:
        subject_data = pd.concat(subject_files, ignore_index=True)
        print(f"    ✓ Loaded subject-level data: {len(subject_data)} observations from {subject_data['subject'].nunique()} subjects")
        
        # Merge with cognitive tags
        subject_merged = pd.merge(
            subject_data,
            contrast_metadata[['task', 'contrast', 'tags']],
            on=['task', 'contrast'],
            how='left'
        )
        subject_merged['parsed_tags'] = subject_merged['tags'].apply(parse_tags)
    else:
        print(f"    ⚠ No subject-level data found - falling back to contrast-level")
        subject_merged = None
    
    # Get top domains for each network

    # By composite score: 'rank_score'
    # By effect size only: 'abs_cohens_d'
    # By significance: 'p_fdr'
    # By consistency: 'consistency'

    top_fpna = df[df['mean_diff_a_minus_b'] > 0].nlargest(top_n, 'consistency')
    top_fpnb = df[df['mean_diff_a_minus_b'] < 0].nlargest(top_n, 'consistency')


    # Prepare data for violin plots
    def prepare_violin_data(top_domains):
        """Extract individual subject values for each domain."""
        if subject_merged is None:
            # Fallback to contrast-level data
            violin_data = []
            for _, domain_row in top_domains.iterrows():
                domain = domain_row['cognitive_domain']
                domain_contrasts = merged_df[merged_df['parsed_tags'].apply(lambda x: domain in x if isinstance(x, list) else False)]
                for _, contrast_row in domain_contrasts.iterrows():
                    violin_data.append({
                        'domain': domain,
                        'cohens_d': contrast_row['cohens_d_mean'],
                        'task': contrast_row['task'],
                        'contrast': contrast_row['contrast'],
                        'source': 'contrast'
                    })
            return pd.DataFrame(violin_data)
        
        # Use subject-level data
        violin_data = []
        for _, domain_row in top_domains.iterrows():
            domain = domain_row['cognitive_domain']
            # Find all subject observations with this tag
            domain_subjects = subject_merged[
                subject_merged['parsed_tags'].apply(lambda x: domain in x if isinstance(x, list) else False)
            ]
            
            for _, subj_row in domain_subjects.iterrows():
                violin_data.append({
                    'domain': domain,
                    'cohens_d': subj_row['cohens_d'],
                    'subject': subj_row['subject'],
                    'task': subj_row['task'],
                    'contrast': subj_row['contrast'],
                    'source': 'subject'
                })
        
        return pd.DataFrame(violin_data)
    
    fpna_violin_data = prepare_violin_data(top_fpna)
    fpnb_violin_data = prepare_violin_data(top_fpnb)
    
    # Plot 1: FPN-A favored domains
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create violin plot
    parts = ax.violinplot(
        [fpna_violin_data[fpna_violin_data['domain'] == d]['cohens_d'].values 
         for d in top_fpna['cognitive_domain']],
        positions=range(len(top_fpna)),
        vert=False,
        widths=0.7,
        showmeans=False,
        showextrema=False
    )
    
    # Color violins
    for pc in parts['bodies']:
        pc.set_facecolor('#d62728')
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Overlay scatter points (subjects or contrasts)
    for i, domain in enumerate(top_fpna['cognitive_domain']):
        domain_data = fpna_violin_data[fpna_violin_data['domain'] == domain]
        y_jitter = np.random.normal(i, 0.04, size=len(domain_data))
        
        # Different appearance for subject vs contrast level
        if 'source' in domain_data.columns and (domain_data['source'] == 'subject').any():
            ax.scatter(domain_data['cohens_d'], y_jitter, 
                      alpha=0.5, s=40, color='darkred', edgecolors='black', linewidth=0.3, zorder=3)
        else:
            ax.scatter(domain_data['cohens_d'], y_jitter, 
                      alpha=0.7, s=60, color='darkred', edgecolors='black', linewidth=0.5, zorder=3)
    
    # Formatting
    ax.set_yticks(range(len(top_fpna)))
    ax.set_yticklabels(top_fpna['cognitive_domain'], fontsize=14, fontweight='bold')
    ax.set_xlabel("Cohen's d (FPN-A - FPN-B)", fontsize=14, fontweight='bold')
    
    data_source = "Subject-Level" if subject_merged is not None else "Contrast-Level"
    ax.set_title(f"Top {top_n} Cognitive Domains Favoring FPN-A\n({data_source} Variability)", 
                fontsize=16, fontweight='bold', pad=20)
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(left=min(0, fpna_violin_data['cohens_d'].min() * 0.9))
    
    plt.tight_layout()
    plot_fpna_path = os.path.join(output_dir, 'top_domains_fpna_violin.png')
    plt.savefig(plot_fpna_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: FPN-B favored domains
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create violin plot
    parts = ax.violinplot(
        [fpnb_violin_data[fpnb_violin_data['domain'] == d]['cohens_d'].values 
         for d in top_fpnb['cognitive_domain']],
        positions=range(len(top_fpnb)),
        vert=False,
        widths=0.7,
        showmeans=False,
        showextrema=False
    )
    
    # Color violins
    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    # Overlay scatter points (subjects or contrasts)
    for i, domain in enumerate(top_fpnb['cognitive_domain']):
        domain_data = fpnb_violin_data[fpnb_violin_data['domain'] == domain]
        y_jitter = np.random.normal(i, 0.04, size=len(domain_data))
        
        # Different appearance for subject vs contrast level
        if 'source' in domain_data.columns and (domain_data['source'] == 'subject').any():
            ax.scatter(domain_data['cohens_d'], y_jitter, 
                      alpha=0.5, s=40, color='darkblue', edgecolors='black', linewidth=0.3, zorder=3)
        else:
            ax.scatter(domain_data['cohens_d'], y_jitter, 
                      alpha=0.7, s=60, color='darkblue', edgecolors='black', linewidth=0.5, zorder=3)
    
    # Formatting
    ax.set_yticks(range(len(top_fpnb)))
    ax.set_yticklabels(top_fpnb['cognitive_domain'], fontsize=14, fontweight='bold')
    ax.set_xlabel("Cohen's d (FPN-A - FPN-B)", fontsize=14, fontweight='bold')
    
    data_source = "Subject-Level" if subject_merged is not None else "Contrast-Level"
    ax.set_title(f"Top {top_n} Cognitive Domains Favoring FPN-B\n({data_source} Variability)", 
                fontsize=16, fontweight='bold', pad=20)
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(right=max(0, fpnb_violin_data['cohens_d'].max() * 0.9))
    
    plt.tight_layout()
    plot_fpnb_path = os.path.join(output_dir, 'top_domains_fpnb_violin.png')
    plt.savefig(plot_fpnb_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_fpna_path, plot_fpnb_path

# Create violin plots based on composite score ranking
plot_fpna, plot_fpnb = create_violin_plots_separate(domain_df_composite, merged, output_dir)
print(f"  ✓ Saved FPN-A violin plot: {plot_fpna}")
print(f"  ✓ Saved FPN-B violin plot: {plot_fpnb}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print(f"Results saved to: {output_dir}")
print(f"\nCSV Files:")
print(f"  - {domain_csv_original}")
print(f"  - {domain_csv_composite}")
print(f"  - {domain_csv_highquality}")
print(f"  - {merged_csv}")
print(f"\nVisualization Plots:")
print(f"  - top_domains_effect_size.png")
print(f"  - top_domains_composite_score.png")
print(f"  - top_domains_high_quality.png (if applicable)")
print(f"  - domain_effect_vs_consistency.png")
print(f"  - ranking_method_comparison.png")
print(f"  - top_domains_fpna_violin.png")
print(f"  - top_domains_fpnb_violin.png")