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

# Merge hand response domains and cardinal direction domains
def merge_cognitive_tags(tag_list):
    """
    Merge related cognitive tags into unified categories:
    - left_hand/right_hand response execution -> hand_response_execution
    - north/south/east/west cardinal-direction judgment -> cardinal-direction_judgment
    """
    if not isinstance(tag_list, list):
        return tag_list
    
    merged_tags = []
    for tag in tag_list:
        # Merge hand response execution
        if tag in ['left_hand_response_execution', 'right_hand_response_execution']:
            if 'hand_response_execution' not in merged_tags:
                merged_tags.append('hand_response_execution')
        # Merge cardinal direction judgments
        elif tag in ['south_cardinal-direction_judgment', 'north_cardinal-direction_judgment', 
                     'east_cardinal-direction_judgment', 'west_cardinal-direction_judgment']:
            if 'cardinal-direction_judgment' not in merged_tags:
                merged_tags.append('cardinal-direction_judgment')
        else:
            merged_tags.append(tag)
    
    return merged_tags

merged['parsed_tags'] = merged['parsed_tags'].apply(merge_cognitive_tags)

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

# ========== 9. VISUALIZATIONS ==========
print("\n[8/9] Generating visualizations...")

# Plot 1: Scatter plot - Effect size vs. Consistency
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
plot_scatter_path = os.path.join(output_dir, 'domain_effect_vs_consistency.png')
plt.savefig(plot_scatter_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved scatter plot: {plot_scatter_path}")

def create_domain_violin_double_panel(df, merged_df, output_dir, top_n=10):
    """
    Create a double-panel figure with cognitive domains:
      Left: top-N domains favoring FPN-A (violins with subject-level dots)
      Right: top-N domains favoring FPN-B (violins with subject-level dots)
    """
    print("\n[EXTRA] Creating cognitive domain double-panel violin figure...")
    
    # Load subject-level data
    results_base = '/ptmp/hmueller2/Downloads/subnetwork_analysis_results'
    subject_files = []
    
    for subject_dir in os.listdir(results_base):
        if not subject_dir.startswith('sub-'):
            continue
        subj_path = os.path.join(results_base, subject_dir)
        if not os.path.isdir(subj_path):
            continue

        subj_file = os.path.join(subj_path, 'fpn_subnetwork_contrast_analysis.csv')
        if os.path.exists(subj_file):
            try:
                df_subj = pd.read_csv(subj_file)
                df_subj['subject'] = subject_dir.replace('sub-', '')
                subject_files.append(df_subj)
            except Exception as e:
                print(f"    ✗ Failed to load {subj_file}: {e}")

    if len(subject_files) == 0:
        print("    ⚠ No subject-level data found; cannot create domain violin figure.")
        return None

    subject_data = pd.concat(subject_files, ignore_index=True)
    print(f"    ✓ Loaded subject-level data: {len(subject_data)} rows from {subject_data['subject'].nunique()} subjects")
    
    # Merge with cognitive tags
    subject_merged = pd.merge(
        subject_data,
        contrast_metadata[['task', 'contrast', 'tags']],
        on=['task', 'contrast'],
        how='left'
    )
    subject_merged['parsed_tags'] = subject_merged['tags'].apply(parse_tags)
    
    # Get top domains for each network
    top_fpna = df[df['mean_diff_a_minus_b'] > 0].nlargest(top_n, 'rank_score')
    top_fpnb = df[df['mean_diff_a_minus_b'] < 0].nlargest(top_n, 'rank_score')
    
    # Extract subject-level data for each domain
    def get_domain_subject_data(domains):
        domain_data = []
        for _, domain_row in domains.iterrows():
            domain = domain_row['cognitive_domain']
            domain_subjects = subject_merged[
                subject_merged['parsed_tags'].apply(lambda x: domain in x if isinstance(x, list) else False)
            ]
            for _, subj_row in domain_subjects.iterrows():
                domain_data.append({
                    'domain': domain,
                    'cohens_d': subj_row['cohens_d'],
                    'subject': subj_row['subject']
                })
        return pd.DataFrame(domain_data)
    
    fpna_data = get_domain_subject_data(top_fpna)
    fpnb_data = get_domain_subject_data(top_fpnb)
    
    # Sort domains by mean Cohen's d
    fpna_domain_order = (
        fpna_data.groupby('domain')['cohens_d']
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
    fpnb_domain_order = (
        fpnb_data.groupby('domain')['cohens_d']
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    
    # Create double-panel figure
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=False)
    
    def plot_domain_side(ax, data, domain_order, title, color):
        positions = np.arange(len(domain_order))
        
        # Create violin plots
        violin_data = [data[data['domain'] == d]['cohens_d'].values for d in domain_order]
        parts = ax.violinplot(
            violin_data,
            positions=positions,
            vert=False,
            widths=0.6,
            showmeans=False,
            showextrema=False
        )
        
        # Color violins
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.3)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Add boxplot overlay
        bp = ax.boxplot(
            violin_data,
            positions=positions,
            vert=False,
            widths=0.3,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color='black', linewidth=2),
            boxprops=dict(facecolor='white', alpha=0.7, linewidth=1.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5)
        )
        
        # Overlay individual subject points
        for i, domain in enumerate(domain_order):
            vals = data[data['domain'] == domain]['cohens_d'].values
            y_jitter = np.random.normal(loc=positions[i], scale=0.08, size=len(vals))
            ax.scatter(
                vals,
                y_jitter,
                s=35,
                color=color,
                edgecolors='black',
                linewidth=0.5,
                alpha=0.6,
                zorder=3
            )
        
        # Formatting
        ax.set_yticks(positions)
        ax.set_yticklabels(domain_order, fontsize=11, fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=15, fontsize=18)
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.grid(axis='x', alpha=0.3)
    
    plot_domain_side(
        axes[0],
        fpna_data,
        fpna_domain_order,
        f"Top {top_n} Cognitive Domains\nFavoring FPN-A",
        color="#d62728"
    )
    plot_domain_side(
        axes[1],
        fpnb_data,
        fpnb_domain_order,
        f"Top {top_n} Cognitive Domains\nFavoring FPN-B",
        color="#1f77b4"
    )
    
    fig.text(0.5, 0.04, "Cohen's d (FPN-A − FPN-B)", 
         ha='center', fontsize=18, fontweight='bold')
    
    plt.tight_layout(rect=[0.05, 0.06, 0.98, 0.97])
    out_path = os.path.join(output_dir, "top_domains_fpna_fpnb_doublepanel.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved double-panel domain figure: {out_path}")
    return out_path

# Create the new cognitive domain double-panel violin plot
domain_doublepanel_path = create_domain_violin_double_panel(
    domain_df_composite, 
    merged, 
    output_dir, 
    top_n=10
)
print(f"  ✓ Saved cognitive domain double-panel: {domain_doublepanel_path}")

def create_contrast_violin_double_panel(results_base, output_dir, top_n=10):
    """
    Create a double-panel figure:
      Left: top-N contrasts favoring FPN-A
      Right: top-N contrasts favoring FPN-B
    Each point is one subject's Cohen's d (FPN-A - FPN-B).
    """
    print("\n[EXTRA] Creating contrast-level double-panel violin figure...")

    # ---- 1. Load subject-level data ----
    subject_files = []
    for subject_dir in os.listdir(results_base):
        if not subject_dir.startswith('sub-'):
            continue
        subj_path = os.path.join(results_base, subject_dir)
        if not os.path.isdir(subj_path):
            continue

        subj_file = os.path.join(subj_path, 'fpn_subnetwork_contrast_analysis.csv')
        if os.path.exists(subj_file):
            try:
                df_subj = pd.read_csv(subj_file)
                df_subj['subject'] = subject_dir.replace('sub-', '')
                subject_files.append(df_subj)
            except Exception as e:
                print(f"    ✗ Failed to load {subj_file}: {e}")

    if len(subject_files) == 0:
        print("    ⚠ No subject-level data found; cannot create contrast violin figure.")
        return None

    subj_all = pd.concat(subject_files, ignore_index=True)
    print(f"    ✓ Loaded subject-level data: {len(subj_all)} rows from {subj_all['subject'].nunique()} subjects")

    # ---- 2. Aggregate per contrast (mean d, consistency) ----
    def contrast_agg(group):
        d_vals = group['cohens_d'].values
        return pd.Series({
            'mean_d': np.mean(d_vals),
            'n_subj': len(d_vals),
            'consistency_pos': (d_vals > 0).mean(),
            'consistency_neg': (d_vals < 0).mean()
        })

    contrast_stats = subj_all.groupby(['task', 'contrast'], group_keys=False).apply(contrast_agg).reset_index()

    # Rank by mean_d
    top_fpna = contrast_stats.sort_values('mean_d', ascending=False).head(top_n).copy()
    top_fpnb = contrast_stats.sort_values('mean_d', ascending=True).head(top_n).copy()

    # ---- 3. Subset subject-level data to only those contrasts ----
    def subset_subjects(top_table):
        merged = pd.merge(
            subj_all,
            top_table[['task', 'contrast']],
            on=['task', 'contrast'],
            how='inner'
        )
        merged['label'] = merged['task'].astype(str) + " | " + merged['contrast'].astype(str)
        return merged

    fpna_subj = subset_subjects(top_fpna)
    fpnb_subj = subset_subjects(top_fpnb)

    # Order labels by mean_d within each side
    fpna_order = (
        fpna_subj.groupby('label')['cohens_d']
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
    fpnb_order = (
        fpnb_subj.groupby('label')['cohens_d']
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # ---- 4. Create double-panel figure ----
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14
    })

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=False)

    def plot_side(ax, data, order, title, color):
        positions = np.arange(len(order))
        plot_data = [data[data['label'] == lbl]['cohens_d'].values for lbl in order]

        bp = ax.boxplot(
            plot_data,
            vert=False,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color='black', linewidth=2),
            boxprops=dict(linewidth=1.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5)
        )

        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.25)

        # overlay all subjects
        for i, lbl in enumerate(order):
            vals = data[data['label'] == lbl]['cohens_d'].values
            y = np.random.normal(loc=positions[i], scale=0.08, size=len(vals))
            ax.scatter(
                vals,
                y,
                s=40,
                color=color,
                edgecolors='black',
                linewidth=0.5,
                alpha=0.7,
                zorder=3
            )

        ax.set_yticks(positions)
        ax.set_yticklabels(order)
        ax.set_title(title, fontweight='bold', pad=15)
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.grid(axis='x', alpha=0.3)

    plot_side(
        axes[0],
        fpna_subj,
        fpna_order,
        f"Top {top_n} Contrasts Favoring FPN-A",
        color="#008B8B"
    )
    plot_side(
        axes[1],
        fpnb_subj,
        fpnb_order,
        f"Top {top_n} Contrasts Favoring FPN-B",
        color="#1f77b4"
    )

    fig.text(0.5, 0.04, "Cohen's d (FPN-A − FPN-B)", 
         ha='center', fontsize=20, fontweight='bold')

    plt.tight_layout(rect=[0.05, 0.06, 0.98, 0.97])
    out_path = os.path.join(output_dir, "top_contrasts_fpna_fpnb_doublepanel.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved double-panel contrast figure: {out_path}")
    return out_path

# Create double-panel contrast-level violin plot
contrast_doublepanel_path = create_contrast_violin_double_panel(
    results_base='/ptmp/hmueller2/Downloads/subnetwork_analysis_results',
    output_dir=output_dir,
    top_n=10
)
print(f"  ✓ Saved contrast double-panel: {contrast_doublepanel_path}")

def create_combined_2x2_figure(results_base, merged_df, domain_df, output_dir, top_n=10):
    """
    Create a 2×2 figure combining:
      Top row: Task contrasts (left: FPN-A, right: FPN-B)
      Bottom row: Cognitive domains (left: FPN-A, right: FPN-B)
    """
    print("\n[EXTRA] Creating combined 2×2 abstract figure...")
    
    # ---- Load subject-level data ----
    subject_files = []
    for subject_dir in os.listdir(results_base):
        if not subject_dir.startswith('sub-'):
            continue
        subj_path = os.path.join(results_base, subject_dir)
        if not os.path.isdir(subj_path):
            continue
        subj_file = os.path.join(subj_path, 'fpn_subnetwork_contrast_analysis.csv')
        if os.path.exists(subj_file):
            try:
                df_subj = pd.read_csv(subj_file)
                df_subj['subject'] = subject_dir.replace('sub-', '')
                subject_files.append(df_subj)
            except Exception as e:
                print(f"    ✗ Failed to load {subj_file}: {e}")
    
    if len(subject_files) == 0:
        print("    ⚠ No subject-level data found.")
        return None
    
    subj_all = pd.concat(subject_files, ignore_index=True)
    print(f"    ✓ Loaded {len(subj_all)} rows from {subj_all['subject'].nunique()} subjects")
    
    # ---- Prepare contrast data with significance filtering ----
    # Merge with group-level p-values from merged_df
    contrast_pvals = merged_df[['task', 'contrast', 'p_value']].drop_duplicates()
    
    def contrast_agg(group):
        d_vals = group['cohens_d'].values
        return pd.Series({
            'mean_d': np.mean(d_vals),
            'n_subj': len(d_vals)
        })
    
    contrast_stats = subj_all.groupby(['task', 'contrast'], group_keys=False).apply(contrast_agg).reset_index()
    
    # Merge with p-values
    contrast_stats = pd.merge(contrast_stats, contrast_pvals, on=['task', 'contrast'], how='left')
    
    # Filter for p < 0.05
    contrast_stats_sig = contrast_stats[contrast_stats['p_value'] < 0.05].copy()
    print(f"    ✓ Found {len(contrast_stats_sig)} significant contrasts (p < 0.05) out of {len(contrast_stats)} total")
    
    # Select top N by Cohen's d (from significant contrasts only)
    top_fpna_contrasts = contrast_stats_sig.sort_values('mean_d', ascending=False).head(top_n).copy()
    top_fpnb_contrasts = contrast_stats_sig.sort_values('mean_d', ascending=True).head(top_n).copy()
    
    print(f"    ✓ Selected top {len(top_fpna_contrasts)} FPN-A contrasts (mean d: {top_fpna_contrasts['mean_d'].min():.3f} to {top_fpna_contrasts['mean_d'].max():.3f})")
    print(f"    ✓ Selected top {len(top_fpnb_contrasts)} FPN-B contrasts (mean d: {top_fpnb_contrasts['mean_d'].min():.3f} to {top_fpnb_contrasts['mean_d'].max():.3f})")
    
    def subset_contrast_subjects(top_table):
        merged = pd.merge(subj_all, top_table[['task', 'contrast']], on=['task', 'contrast'], how='inner')
        merged['label'] = merged['task'].astype(str) + " | " + merged['contrast'].astype(str)
        return merged
    
    fpna_contrast_subj = subset_contrast_subjects(top_fpna_contrasts)
    fpnb_contrast_subj = subset_contrast_subjects(top_fpnb_contrasts)
    
    fpna_contrast_order = (fpna_contrast_subj.groupby('label')['cohens_d'].mean()
                           .sort_values(ascending=True).index.tolist())
    fpnb_contrast_order = (fpnb_contrast_subj.groupby('label')['cohens_d'].mean()
                           .sort_values(ascending=False).index.tolist())
    
    # ---- Prepare domain data ----
    subject_merged = pd.merge(subj_all, contrast_metadata[['task', 'contrast', 'tags']], 
                              on=['task', 'contrast'], how='left')
    subject_merged['parsed_tags'] = subject_merged['tags'].apply(parse_tags)
    subject_merged['parsed_tags'] = subject_merged['parsed_tags'].apply(merge_cognitive_tags)
    
    top_fpna_domains = domain_df[domain_df['mean_diff_a_minus_b'] > 0].nlargest(top_n, 'rank_score')
    top_fpnb_domains = domain_df[domain_df['mean_diff_a_minus_b'] < 0].nlargest(top_n, 'rank_score')
    
    def get_domain_subject_data(domains):
        domain_data = []
        for _, domain_row in domains.iterrows():
            domain = domain_row['cognitive_domain']
            domain_subjects = subject_merged[
                subject_merged['parsed_tags'].apply(lambda x: domain in x if isinstance(x, list) else False)
            ]
            for _, subj_row in domain_subjects.iterrows():
                domain_data.append({
                    'domain': domain,
                    'cohens_d': subj_row['cohens_d'],
                    'subject': subj_row['subject']
                })
        return pd.DataFrame(domain_data)
    
    fpna_domain_data = get_domain_subject_data(top_fpna_domains)
    fpnb_domain_data = get_domain_subject_data(top_fpnb_domains)
    
    fpna_domain_order = (fpna_domain_data.groupby('domain')['cohens_d'].mean()
                         .sort_values(ascending=True).index.tolist())
    fpnb_domain_order = (fpnb_domain_data.groupby('domain')['cohens_d'].mean()
                         .sort_values(ascending=False).index.tolist())
    
    # ---- Create 2×2 figure ----
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.size': 15,
        'axes.titlesize': 18,
        'axes.labelsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15
    })
    
    fig = plt.figure(figsize=(24, 16))
    # Increased wspace from 0.4 to 0.5 for more horizontal spacing
    gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.5, left=0.05, right=0.98, top=0.96, bottom=0.05)
    
    ax_contrast_a = fig.add_subplot(gs[0, 0])
    ax_contrast_b = fig.add_subplot(gs[0, 1])
    ax_domain_a = fig.add_subplot(gs[1, 0])
    ax_domain_b = fig.add_subplot(gs[1, 1])
    
    # Colors
    color_fpna = "#008B8B"  # Teal
    color_fpnb = "#1f77b4"  # Blue
    
    # ---- Plot function ----
    def plot_panel(ax, data, order, color, is_contrast=True):
        positions = np.arange(len(order))
        label_col = 'label' if is_contrast else 'domain'
        
        plot_data = [data[data[label_col] == lbl]['cohens_d'].values for lbl in order]
        
        # Boxplot
        bp = ax.boxplot(
            plot_data,
            vert=False,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color='black', linewidth=2.5),
            boxprops=dict(linewidth=2),
            whiskerprops=dict(linewidth=2),
            capprops=dict(linewidth=2)
        )
        
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.25)
        
        # Scatter individual points
        for i, lbl in enumerate(order):
            vals = data[data[label_col] == lbl]['cohens_d'].values
            y = np.random.normal(loc=positions[i], scale=0.08, size=len(vals))
            ax.scatter(vals, y, s=45, color=color, edgecolors='black', 
                      linewidth=0.7, alpha=0.7, zorder=3)
        
        ax.set_yticks(positions)
        ax.set_yticklabels(order, fontsize=13)
        ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.8)
        ax.grid(axis='x', alpha=0.3, linewidth=1)
    
    # ---- Plot all panels ----
    # Top row: Contrasts
    plot_panel(ax_contrast_a, fpna_contrast_subj, fpna_contrast_order, color_fpna, is_contrast=True)
    plot_panel(ax_contrast_b, fpnb_contrast_subj, fpnb_contrast_order, color_fpnb, is_contrast=True)
    
    ax_contrast_a.set_title("FPN-A Dominant", fontsize=20, fontweight='bold', pad=12)
    ax_contrast_b.set_title("FPN-B Dominant", fontsize=20, fontweight='bold', pad=12)
    
    # Bottom row: Domains
    plot_panel(ax_domain_a, fpna_domain_data, fpna_domain_order, color_fpna, is_contrast=False)
    plot_panel(ax_domain_b, fpnb_domain_data, fpnb_domain_order, color_fpnb, is_contrast=False)
    
    # X-axis labels only on bottom row
    ax_domain_a.set_xlabel("Cohen's d (FPN-A − FPN-B)", fontsize=14, fontweight='bold')
    ax_domain_b.set_xlabel("Cohen's d (FPN-A − FPN-B)", fontsize=14, fontweight='bold')
    
    out_path = os.path.join(output_dir, "abstract_top_results.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved combined 2×2 figure: {out_path}")
    return out_path

# Create combined 2×2 figure (add this after the individual plots)
combined_2x2_path = create_combined_2x2_figure(
    results_base='/ptmp/hmueller2/Downloads/subnetwork_analysis_results',
    merged_df=merged,
    domain_df=domain_df_composite,
    output_dir=output_dir,
    top_n=10
)
print(f"  ✓ Saved combined 2×2 abstract figure: {combined_2x2_path}")

# ========== 10. DOMAIN SUMMARY TEXT FILES (REPLACES WORD CLOUDS) ==========
print("\n[9/9] Creating domain summary text files...")

# FPN-A: positive Cohen's d (mean_diff_a_minus_b > 0)
fpna_domains = domain_df_composite[domain_df_composite['mean_diff_a_minus_b'] > 0].copy()
fpnb_domains = domain_df_composite[domain_df_composite['mean_diff_a_minus_b'] < 0].copy()

top_fpna = fpna_domains.nlargest(25, 'rank_score')
top_fpnb = fpnb_domains.nlargest(25, 'rank_score')

# Save FPN-A domains
fpna_txt_path = os.path.join(output_dir, 'fpna_top_domains.txt')
with open(fpna_txt_path, 'w') as f:
    f.write("Top 25 FPN-A Dominant Cognitive Domains\n")
    f.write("=" * 60 + "\n\n")
    for i, (_, row) in enumerate(top_fpna.iterrows(), 1):
        domain_name = row['cognitive_domain'].replace('_', ' ').replace('-', ' ').title()
        f.write(f"{i}. {domain_name}\n")
        f.write(f"   Composite Score: {row['rank_score']:.2f}\n")
        f.write(f"   Cohen's d: {row['mean_cohens_d']:.3f}\n")
        f.write(f"   p-value (FDR): {row['p_fdr']:.4f}\n")
        f.write(f"   Consistency: {row['consistency']*100:.1f}%\n")
        f.write(f"   Number of contrasts: {row['n_contrasts']}\n\n")

print(f"  ✓ Saved FPN-A domain summary: {fpna_txt_path}")

# Save FPN-B domains
fpnb_txt_path = os.path.join(output_dir, 'fpnb_top_domains.txt')
with open(fpnb_txt_path, 'w') as f:
    f.write("Top 25 FPN-B Dominant Cognitive Domains\n")
    f.write("=" * 60 + "\n\n")
    for i, (_, row) in enumerate(top_fpnb.iterrows(), 1):
        domain_name = row['cognitive_domain'].replace('_', ' ').replace('-', ' ').title()
        f.write(f"{i}. {domain_name}\n")
        f.write(f"   Composite Score: {row['rank_score']:.2f}\n")
        f.write(f"   Cohen's d: {row['mean_cohens_d']:.3f}\n")
        f.write(f"   p-value (FDR): {row['p_fdr']:.4f}\n")
        f.write(f"   Consistency: {row['consistency']*100:.1f}%\n")
        f.write(f"   Number of contrasts: {row['n_contrasts']}\n\n")

print(f"  ✓ Saved FPN-B domain summary: {fpnb_txt_path}")

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
print(f"  - domain_effect_vs_consistency.png")
print(f"  - top_domains_fpna_fpnb_doublepanel.png")
print(f"  - top_contrasts_fpna_fpnb_doublepanel.png")
print(f"  - abstract_top_results.png (Combined 2×2)")
print(f"\nText Summaries:")
print(f"  - fpna_top_domains.txt")
print(f"  - fpnb_top_domains.txt")