"""
Map PPI contrast specificity results to cognitive domains using Cognitive Atlas tags.
Aggregates PPI metrics by cognitive concept to identify functional connectivity specialization.
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
print("PPI COGNITIVE DOMAIN MAPPING")
print("="*60)

# Load PPI contrast specificity results
ppi_results_file = '/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/ppi_results_dmn_dan/group_analysis/contrast_specificity_scores.csv'
ppi_results = pd.read_csv(ppi_results_file)
print(f"\n[1/6] Loaded PPI results: {len(ppi_results)} contrasts")
print(f"  Columns: {list(ppi_results.columns)}")

# Parse task_condition into task and contrast
print("\n[DEBUG] Parsing task_condition column...")
print(f"  Sample values: {ppi_results['task_condition'].head(3).tolist()}")

# Split task_condition by underscore or other delimiter
# Assuming format is "task_contrast" or similar
def parse_task_condition(task_condition):
    """Extract task and contrast from task_condition string."""
    parts = str(task_condition).rsplit('_', 1)  # Split from right to get last part as contrast
    if len(parts) == 2:
        return parts[0], parts[1]
    else:
        return str(task_condition), ''

ppi_results[['task', 'contrast']] = ppi_results['task_condition'].apply(
    lambda x: pd.Series(parse_task_condition(x))
)

print(f"  Example: task_condition='{ppi_results['task_condition'].iloc[0]}' → task='{ppi_results['task'].iloc[0]}', contrast='{ppi_results['contrast'].iloc[0]}'")

# Load contrast metadata with cognitive tags
contrast_metadata = pd.read_csv('/home/hmueller2/ibc_code/ibc_latent/Data Info/all_contrasts.tsv', sep='\t')
print(f"\n[1/6] Loaded contrast metadata: {len(contrast_metadata)} contrasts")
print(f"  Columns: {list(contrast_metadata.columns)}")

# ========== 2. MERGE PPI RESULTS WITH COGNITIVE TAGS ==========
print("\n[2/6] Merging PPI results with cognitive tags...")

# Merge on task and contrast
merged = pd.merge(
    ppi_results,
    contrast_metadata[['task', 'contrast', 'tags', 'pretty name']],
    on=['task', 'contrast'],
    how='left'
)

print(f"  Matched {merged['tags'].notna().sum()} / {len(merged)} contrasts")

# Handle missing tags
print("\n[DEBUG] Investigating missing tags...")
missing_tags = merged[merged['tags'].isna()]
if len(missing_tags) > 0:
    print(f"  WARNING: {len(missing_tags)} contrasts have no tags:")
    for _, row in missing_tags.head(10).iterrows():
        print(f"    PPI results: task='{row['task']}', contrast='{row['contrast']}'")
        
        # Check if similar entries exist in metadata
        task_matches = contrast_metadata[contrast_metadata['task'].str.contains(row['task'], case=False, na=False)]
        print(f"      → Found {len(task_matches)} tasks matching '{row['task']}' in metadata")

# ========== 3. PARSE COGNITIVE TAGS ==========
print("\n[3/6] Parsing cognitive tags...")

def parse_tags(tag_str):
    """Convert string representation of list to actual list."""
    if pd.isna(tag_str):
        return []
    try:
        tags = ast.literal_eval(tag_str)
        return tags if isinstance(tags, list) else []
    except:
        return [t.strip().strip("'\"") for t in str(tag_str).strip('[]').split(',') if t.strip()]

merged['parsed_tags'] = merged['tags'].apply(parse_tags)
merged['n_tags'] = merged['parsed_tags'].apply(len)

print(f"  Average tags per contrast: {merged['n_tags'].mean():.1f}")
print(f"  Contrasts with tags: {(merged['n_tags'] > 0).sum()}")

if (merged['n_tags'] > 0).any():
    sample_with_tags = merged[merged['n_tags'] > 0].iloc[0]
    print(f"  Example: {sample_with_tags['task']} - {sample_with_tags['contrast']}")
    print(f"    Tags: {sample_with_tags['parsed_tags']}")

# ========== 4. AGGREGATE BY COGNITIVE DOMAIN ==========
print("\n[4/6] Aggregating PPI metrics by cognitive domain...")

# Identify PPI metric columns (exclude task, contrast, tags, task_condition)
exclude_cols = ['task', 'contrast', 'tags', 'pretty name', 'task_condition']
ppi_metrics = [col for col in ppi_results.columns if col not in exclude_cols]
print(f"  PPI metrics found: {ppi_metrics}")

# Create domain-level statistics
domain_data = defaultdict(lambda: {
    metric: [] for metric in ppi_metrics
})

for _, row in merged.iterrows():
    if row['n_tags'] == 0:
        continue
    
    for tag in row['parsed_tags']:
        for metric in ppi_metrics:
            if pd.notna(row[metric]):
                domain_data[tag][metric].append(row[metric])

print(f"  Collected data for {len(domain_data)} unique cognitive tags")

# Compute domain-level statistics
domain_results = []

for domain, data in domain_data.items():
    # Get sample size from first metric
    first_metric = ppi_metrics[0]
    n_contrasts = len(data[first_metric])
    
    if n_contrasts < 2:  # Need at least 2 contrasts for meaningful stats
        continue
    
    result = {
        'cognitive_domain': domain,
        'n_contrasts': n_contrasts,
    }
    
    # Compute statistics for each PPI metric
    for metric in ppi_metrics:
        values = np.array(data[metric])
        
        result[f'{metric}_mean'] = np.mean(values)
        result[f'{metric}_std'] = np.std(values, ddof=1) if n_contrasts > 1 else np.nan
        result[f'{metric}_se'] = np.std(values, ddof=1) / np.sqrt(n_contrasts) if n_contrasts > 1 else np.nan
        
        # One-sample t-test (testing if mean differs from zero)
        if n_contrasts >= 2:
            t_stat, p_val = stats.ttest_1samp(values, 0)
            result[f'{metric}_t'] = t_stat
            result[f'{metric}_p'] = p_val
        else:
            result[f'{metric}_t'] = np.nan
            result[f'{metric}_p'] = np.nan
    
    domain_results.append(result)

domain_df = pd.DataFrame(domain_results)

# FDR correction for each metric
from statsmodels.stats.multitest import multipletests

for metric in ppi_metrics:
    p_col = f'{metric}_p'
    if p_col in domain_df.columns and domain_df[p_col].notna().any():
        _, domain_df[f'{metric}_p_fdr'], _, _ = multipletests(
            domain_df[p_col].fillna(1), method='fdr_bh'
        )

print(f"  Found {len(domain_df)} cognitive domains with ≥2 contrasts")

# ========== 5. CREATE RANKING SCHEMES ==========
print("\n[5/6] Creating ranking schemes...")

# Use first PPI metric for ranking
primary_metric = ppi_metrics[0] if ppi_metrics else None

if primary_metric:
    # Add absolute value column to base dataframe (needed for filtering)
    domain_df[f'{primary_metric}_abs'] = domain_df[f'{primary_metric}_mean'].abs()
    
    # ORIGINAL: Sort by absolute effect
    domain_df_original = domain_df.copy()
    domain_df_original = domain_df_original.sort_values(f'{primary_metric}_abs', ascending=False)
    
    # COMPOSITE: Balance significance + effect + sample size
    domain_df_composite = domain_df.copy()
    p_col = f'{primary_metric}_p_fdr'
    domain_df_composite['rank_score'] = (
        -np.log10(domain_df_composite[p_col].fillna(1) + 1e-10) * 0.4 +
        domain_df_composite[f'{primary_metric}_abs'] * 100 * 0.4 +
        domain_df_composite['n_contrasts'] * 0.2
    )
    domain_df_composite = domain_df_composite.sort_values('rank_score', ascending=False)
    
    print(f"  Composite score - Top domain: {domain_df_composite.iloc[0]['cognitive_domain']} (score={domain_df_composite.iloc[0]['rank_score']:.2f})")
    
    # HIGH-QUALITY: Stringent filtering (now domain_df has the _abs column)
    domain_df_highquality = domain_df[
        (domain_df[p_col] < 0.05) &
        (domain_df[f'{primary_metric}_abs'] > domain_df[f'{primary_metric}_abs'].quantile(0.25)) &
        (domain_df['n_contrasts'] >= 5)
    ].copy()
    domain_df_highquality = domain_df_highquality.sort_values(f'{primary_metric}_abs', ascending=False)
    
    print(f"  High-quality domains (filtered): {len(domain_df_highquality)}")

# ========== 6. STATISTICAL SUMMARY ==========
print("\n[6/9] Statistical summary:")
if primary_metric:
    p_col = f'{primary_metric}_p_fdr'
    print(f"  Domains with FDR < 0.05: {(domain_df[p_col] < 0.05).sum()}")
    print(f"  Domains with |{primary_metric}| > median: {(domain_df[f'{primary_metric}_abs'] > domain_df[f'{primary_metric}_abs'].median()).sum()}")

# ========== 7. SAVE RESULTS ==========
output_dir = '/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/ppi_results_dmn_dan/cognitive_atlas'
os.makedirs(output_dir, exist_ok=True)

# Save rankings
domain_csv_original = os.path.join(output_dir, 'ppi_cognitive_domains_by_effect.csv')
domain_df_original.to_csv(domain_csv_original, index=False)
print(f"\n[7/9] ✓ Saved to: {domain_csv_original}")

domain_csv_composite = os.path.join(output_dir, 'ppi_cognitive_domains_composite_score.csv')
domain_df_composite.to_csv(domain_csv_composite, index=False)
print(f"[7/9] ✓ Saved to: {domain_csv_composite}")

domain_csv_highquality = os.path.join(output_dir, 'ppi_cognitive_domains_high_quality.csv')
domain_df_highquality.to_csv(domain_csv_highquality, index=False)
print(f"[7/9] ✓ Saved to: {domain_csv_highquality}")

# Save merged data
merged_csv = os.path.join(output_dir, 'contrasts_with_cognitive_tags_ppi.csv')
merged.to_csv(merged_csv, index=False)
print(f"[7/9] ✓ Saved to: {merged_csv}")

# ========== 8. PRINT TOP FINDINGS ==========
if primary_metric:
    print("\n" + "="*60)
    print(f"TOP 10 DOMAINS (BY {primary_metric.upper()})")
    print("="*60)
    top_domains = domain_df_original.head(10)
    cols_to_print = ['cognitive_domain', 'n_contrasts', f'{primary_metric}_mean', f'{primary_metric}_p_fdr']
    print(top_domains[cols_to_print].to_string(index=False))
    
    print("\n" + "="*60)
    print("COMPOSITE SCORE RANKING (BALANCED)")
    print("="*60)
    top_composite = domain_df_composite.head(10)
    cols_to_print = ['cognitive_domain', 'rank_score', f'{primary_metric}_mean', f'{primary_metric}_p_fdr', 'n_contrasts']
    print(top_composite[cols_to_print].to_string(index=False))
    
    print("\n" + "="*60)
    print("HIGH-QUALITY DOMAINS (STRINGENT FILTERING)")
    print("="*60)
    print(f"Total domains meeting criteria: {len(domain_df_highquality)}")
    if len(domain_df_highquality) > 0:
        print("\nTOP 10:")
        cols_to_print = ['cognitive_domain', 'n_contrasts', f'{primary_metric}_mean', f'{primary_metric}_p_fdr']
        print(domain_df_highquality.head(10)[cols_to_print].to_string(index=False))

# ========== 9. VISUALIZATIONS ==========
print("\n[8/9] Generating visualizations...")

if primary_metric:
    def create_ppi_bar_plot(df, title_suffix, filename_suffix, top_n=20):
        """Create horizontal bar plot of top domains."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        top_domains = df.head(top_n)
        metric_col = f'{primary_metric}_mean'
        colors = ['#d62728' if x > 0 else '#1f77b4' for x in top_domains[metric_col]]
        
        ax.barh(range(len(top_domains)), top_domains[metric_col], color=colors)
        ax.set_yticks(range(len(top_domains)))
        ax.set_yticklabels(top_domains['cognitive_domain'], fontsize=9)
        ax.set_xlabel(f'{primary_metric} (mean)', fontsize=11)
        ax.set_title(f"Top {top_n} Cognitive Domains - {title_suffix}", fontsize=13, fontweight='bold')
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
        
        # Add significance markers
        p_col = f'{primary_metric}_p_fdr'
        for i, (_, row) in enumerate(top_domains.iterrows()):
            if row[p_col] < 0.001:
                marker = '***'
            elif row[p_col] < 0.01:
                marker = '**'
            elif row[p_col] < 0.05:
                marker = '*'
            else:
                marker = ''
            
            if marker:
                x_pos = row[metric_col] + (0.02 if row[metric_col] > 0 else -0.02)
                ax.text(x_pos, i, marker, va='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'ppi_domains_{filename_suffix}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path
    
    # Create plots
    plot1 = create_ppi_bar_plot(domain_df_original, "Sorted by Effect", "effect")
    print(f"  ✓ Saved: {plot1}")
    
    plot2 = create_ppi_bar_plot(domain_df_composite, "Composite Score Ranking", "composite_score")
    print(f"  ✓ Saved: {plot2}")
    
    if len(domain_df_highquality) >= 5:
        plot3 = create_ppi_bar_plot(domain_df_highquality, "High-Quality Domains", "high_quality", top_n=min(20, len(domain_df_highquality)))
        print(f"  ✓ Saved: {plot3}")
    
    # Scatter plot: Effect vs. Sample Size
    fig, ax = plt.subplots(figsize=(10, 8))
    
    metric_col = f'{primary_metric}_mean'
    p_col = f'{primary_metric}_p_fdr'
    sig_mask = domain_df_composite[p_col] < 0.05
    colors_sig = ['#d62728' if sig else '#999999' for sig in sig_mask]
    
    ax.scatter(domain_df_composite[metric_col], domain_df_composite['n_contrasts'],
               s=200, c=colors_sig, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(f'{primary_metric} (mean)', fontsize=11)
    ax.set_ylabel('Number of Contrasts', fontsize=11)
    ax.set_title(f"PPI {primary_metric.capitalize()} vs. Sample Size\n(by Cognitive Domain)", 
                 fontsize=12, fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(alpha=0.3)
    
    # Label top domains
    top_labels = domain_df_composite.nlargest(8, 'rank_score')
    for _, row in top_labels.iterrows():
        ax.annotate(row['cognitive_domain'], 
                    xy=(row[metric_col], row['n_contrasts']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, 'ppi_effect_vs_sample_size.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {scatter_path}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print(f"Results saved to: {output_dir}")
print(f"  - ppi_cognitive_domains_by_effect.csv")
print(f"  - ppi_cognitive_domains_composite_score.csv")
print(f"  - ppi_cognitive_domains_high_quality.csv")
print(f"  - contrasts_with_cognitive_tags_ppi.csv")
print(f"  - Visualization plots")