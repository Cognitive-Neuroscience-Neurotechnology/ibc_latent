"""
Map FPN subnetwork PPI connectivity to cognitive domains using Cognitive Atlas tags.
Analyzes differential coupling of FPNA vs FPNB to DMN and DAN across cognitive tasks.
"""
import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
from statsmodels.stats.multitest import multipletests

# ========== 1. LOAD DATA ==========
print("="*60)
print("COGNITIVE DOMAIN MAPPING FOR FPN-DMN/DAN CONNECTIVITY")
print("="*60)

# Load PPI results (contrast-level group statistics)
ppi_base = '/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/ppi_results_dmn_dan/group_analysis'
ppi_results = pd.read_csv(os.path.join(ppi_base, 'seed_target_contrast_ranking.csv'))
print(f"\n[1/6] Loaded PPI results: {len(ppi_results)} seed-target-contrast combinations")
print(f"  Columns: {list(ppi_results.columns)}")

# Load subject-level PPI data for variance estimation (optional - skip if not found)
subject_ppi_files = []
ppi_subject_base = '/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/ppi_results_dmn_dan'

print("\n[DEBUG] Searching for subject-level PPI data...")
print(f"  Base directory: {ppi_subject_base}")
print(f"  Directory exists: {os.path.exists(ppi_subject_base)}")

if os.path.exists(ppi_subject_base):
    all_items = sorted(os.listdir(ppi_subject_base))
    print(f"  Total items in base directory: {len(all_items)}")
    print(f"  First 20 items: {all_items[:20]}")
    
    subject_dirs = [d for d in all_items if d.startswith('sub-') and os.path.isdir(os.path.join(ppi_subject_base, d))]
    print(f"  Found {len(subject_dirs)} subject directories: {subject_dirs[:10]}")
    
    for subject_dir in subject_dirs:
        subject = subject_dir.replace('sub-', '')
        subject_path = os.path.join(ppi_subject_base, subject_dir)
        
        task_items = sorted(os.listdir(subject_path))
        print(f"\n  {subject_dir}/ contains: {task_items[:10]}")
        
        for item in task_items:
            item_path = os.path.join(subject_path, item)
            
            if os.path.isdir(item_path):
                ppi_file = os.path.join(item_path, 'ppi_results.csv')
                print(f"    Checking {item}/ppi_results.csv... exists={os.path.exists(ppi_file)}")
                
                if os.path.exists(ppi_file):
                    try:
                        df = pd.read_csv(ppi_file)
                        df['subject'] = subject
                        subject_ppi_files.append(df)
                        print(f"      ✓ Loaded {subject}/{item} ({len(df)} rows)")
                    except Exception as e:
                        print(f"      ✗ Failed to load {item}: {e}")

if len(subject_ppi_files) > 0:
    subject_ppi_data = pd.concat(subject_ppi_files, ignore_index=True)
    print(f"\n[0/6] Loaded subject-level PPI data: {len(subject_ppi_data)} run-level observations")
    print(f"  Found data for {subject_ppi_data['subject'].nunique()} subjects")
else:
    print("\n[0/6] WARNING: No subject-level PPI data found - proceeding with group-level only")
    subject_ppi_data = None

# Load contrast metadata with cognitive tags
contrast_metadata = pd.read_csv('/home/hmueller2/ibc_code/ibc_latent/Data Info/all_contrasts.tsv', sep='\t')
print(f"[1/6] Loaded contrast metadata: {len(contrast_metadata)} contrasts")

# ========== 2. MERGE PPI RESULTS WITH COGNITIVE TAGS ==========
print("\n[2/6] Merging PPI results with cognitive tags...")

# Merge on task and condition (condition = contrast in PPI analysis)
merged = pd.merge(
    ppi_results,
    contrast_metadata[['task', 'contrast', 'tags', 'pretty name']],
    left_on=['task', 'condition'],
    right_on=['task', 'contrast'],
    how='left'
)

print(f"  Matched {merged['tags'].notna().sum()} / {len(merged)} combinations")

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

# ========== 4. COMPUTE DIFFERENTIAL CONNECTIVITY ==========
print("\n[4/6] Computing differential connectivity patterns...")

# For each contrast, compute FPNA-DMN vs FPNA-DAN difference
merged_fpna = merged[merged['seed'] == 'FPNA'].copy()
fpna_dmn = merged_fpna[merged_fpna['target'] == 'DMN'].set_index(['task', 'condition'])
fpna_dan = merged_fpna[merged_fpna['target'] == 'DAN'].set_index(['task', 'condition'])

# For each contrast, compute FPNB-DMN vs FPNB-DAN difference
merged_fpnb = merged[merged['seed'] == 'FPNB'].copy()
fpnb_dmn = merged_fpnb[merged_fpnb['target'] == 'DMN'].set_index(['task', 'condition'])
fpnb_dan = merged_fpnb[merged_fpnb['target'] == 'DAN'].set_index(['task', 'condition'])

# Create differential connectivity dataframes
fpna_differential = pd.DataFrame({
    'task': fpna_dmn.index.get_level_values('task'),
    'contrast': fpna_dmn.index.get_level_values('condition'),
    'diff_dmn_minus_dan_effect': fpna_dmn['mean_effect'].values - fpna_dan['mean_effect'].values,
    'diff_dmn_minus_dan_cohens_d': fpna_dmn['cohens_d'].values - fpna_dan['cohens_d'].values,
    'dmn_effect': fpna_dmn['mean_effect'].values,
    'dan_effect': fpna_dan['mean_effect'].values,
    'dmn_cohens_d': fpna_dmn['cohens_d'].values,
    'dan_cohens_d': fpna_dan['cohens_d'].values,
    'tags': fpna_dmn['tags'].values,
    'parsed_tags': fpna_dmn['parsed_tags'].values
})

fpnb_differential = pd.DataFrame({
    'task': fpnb_dmn.index.get_level_values('task'),
    'contrast': fpnb_dmn.index.get_level_values('condition'),
    'diff_dmn_minus_dan_effect': fpnb_dmn['mean_effect'].values - fpnb_dan['mean_effect'].values,
    'diff_dmn_minus_dan_cohens_d': fpnb_dmn['cohens_d'].values - fpnb_dan['cohens_d'].values,
    'dmn_effect': fpnb_dmn['mean_effect'].values,
    'dan_effect': fpnb_dan['mean_effect'].values,
    'dmn_cohens_d': fpnb_dmn['cohens_d'].values,
    'dan_cohens_d': fpnb_dan['cohens_d'].values,
    'tags': fpnb_dmn['tags'].values,
    'parsed_tags': fpnb_dmn['parsed_tags'].values
})

print(f"  FPNA differential connectivity: {len(fpna_differential)} contrasts")
print(f"  FPNB differential connectivity: {len(fpnb_differential)} contrasts")

# ========== 5. AGGREGATE BY COGNITIVE DOMAIN ==========
print("\n[5/6] Aggregating connectivity by cognitive domain...")

def aggregate_by_domain(diff_df, metric='effect'):
    """Aggregate differential connectivity by cognitive domain."""
    domain_data = defaultdict(lambda: {
        'diff_values': [],
        'dmn_values': [],
        'dan_values': [],
        'contrasts': [],
        'tasks': []
    })
    
    diff_col = f'diff_dmn_minus_dan_{metric}'
    dmn_col = f'dmn_{metric}'
    dan_col = f'dan_{metric}'
    
    for _, row in diff_df.iterrows():
        # Check if tags exist and are not empty
        if not isinstance(row['parsed_tags'], list) or len(row['parsed_tags']) == 0:
            continue
        
        for tag in row['parsed_tags']:
            domain_data[tag]['diff_values'].append(row[diff_col])
            domain_data[tag]['dmn_values'].append(row[dmn_col])
            domain_data[tag]['dan_values'].append(row[dan_col])
            domain_data[tag]['contrasts'].append(row['contrast'])
            domain_data[tag]['tasks'].append(row['task'])
    
    # Compute domain-level statistics
    domain_results = []
    for domain, data in domain_data.items():
        diff = np.array(data['diff_values'])
        dmn = np.array(data['dmn_values'])
        dan = np.array(data['dan_values'])
        
        n_contrasts = len(diff)
        if n_contrasts < 2:
            continue
        
        # Statistics
        mean_diff = np.mean(diff)
        se_diff = np.std(diff) / np.sqrt(n_contrasts)
        t_stat, p_val = stats.ttest_1samp(diff, 0)
        
        # Target preference
        prop_dmn_favored = (diff > 0).sum() / n_contrasts
        prop_dan_favored = (diff < 0).sum() / n_contrasts
        consistency = max(prop_dmn_favored, prop_dan_favored)
        
        domain_results.append({
            'cognitive_domain': domain,
            'n_contrasts': n_contrasts,
            'n_tasks': len(set(data['tasks'])),
            'mean_diff_dmn_minus_dan': mean_diff,
            'se_diff': se_diff,
            't_statistic': t_stat,
            'p_value': p_val,
            'mean_dmn': np.mean(dmn),
            'mean_dan': np.mean(dan),
            'consistency': consistency,
            'dmn_favored_pct': prop_dmn_favored * 100,
            'dan_favored_pct': prop_dan_favored * 100
        })
    
    domain_df = pd.DataFrame(domain_results)
    
    # FDR correction
    if len(domain_df) > 0:
        _, domain_df['p_fdr'], _, _ = multipletests(domain_df['p_value'], method='fdr_bh')
        domain_df['abs_diff'] = domain_df['mean_diff_dmn_minus_dan'].abs()
    
    return domain_df, domain_data

# Aggregate for FPNA (effect size)
fpna_domains_effect, fpna_domain_data_effect = aggregate_by_domain(fpna_differential, metric='effect')
print(f"  FPNA (effect): {len(fpna_domains_effect)} domains with ≥2 contrasts")

# Aggregate for FPNA (Cohen's d)
fpna_domains_cohens, fpna_domain_data_cohens = aggregate_by_domain(fpna_differential, metric='cohens_d')
print(f"  FPNA (Cohen's d): {len(fpna_domains_cohens)} domains with ≥2 contrasts")

# Aggregate for FPNB (effect size)
fpnb_domains_effect, fpnb_domain_data_effect = aggregate_by_domain(fpnb_differential, metric='effect')
print(f"  FPNB (effect): {len(fpnb_domains_effect)} domains with ≥2 contrasts")

# Aggregate for FPNB (Cohen's d)
fpnb_domains_cohens, fpnb_domain_data_cohens = aggregate_by_domain(fpnb_differential, metric='cohens_d')
print(f"  FPNB (Cohen's d): {len(fpnb_domains_cohens)} domains with ≥2 contrasts")

# ========== 6. RANK BY DIFFERENTIAL EFFECT ==========
print("\n[6/6] Ranking domains by differential connectivity...")

# Sort by absolute differential effect
fpna_domains_effect = fpna_domains_effect.sort_values('abs_diff', ascending=False)
fpna_domains_cohens = fpna_domains_cohens.sort_values('abs_diff', ascending=False)
fpnb_domains_effect = fpnb_domains_effect.sort_values('abs_diff', ascending=False)
fpnb_domains_cohens = fpnb_domains_cohens.sort_values('abs_diff', ascending=False)

print(f"  Top FPNA domain (effect): {fpna_domains_effect.iloc[0]['cognitive_domain']} (diff={fpna_domains_effect.iloc[0]['mean_diff_dmn_minus_dan']:.3f})")
print(f"  Top FPNB domain (effect): {fpnb_domains_effect.iloc[0]['cognitive_domain']} (diff={fpnb_domains_effect.iloc[0]['mean_diff_dmn_minus_dan']:.3f})")

# ========== 7. SAVE RESULTS ==========
output_dir = '/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/ppi_results_dmn_dan/cognitive_atlas'
os.makedirs(output_dir, exist_ok=True)

fpna_domains_effect.to_csv(os.path.join(output_dir, 'fpna_target_specificity_effect.csv'), index=False)
fpna_domains_cohens.to_csv(os.path.join(output_dir, 'fpna_target_specificity_cohens_d.csv'), index=False)
fpnb_domains_effect.to_csv(os.path.join(output_dir, 'fpnb_target_specificity_effect.csv'), index=False)
fpnb_domains_cohens.to_csv(os.path.join(output_dir, 'fpnb_target_specificity_cohens_d.csv'), index=False)

print(f"\n[7/9] ✓ Saved domain rankings to: {output_dir}")

# ========== 8. CREATE VIOLIN PLOTS ==========
print("\n[8/9] Generating violin plots...")

def create_violin_plots_ppi(domain_df, domain_data, diff_df, seed_name, metric='effect', output_dir='', top_n=10):
    """Create separate violin plots for top DMN and DAN-preferring domains with subject-level variability."""
    
    diff_col = f'diff_dmn_minus_dan_{metric}'
    metric_label = 'PPI Effect (β)' if metric == 'effect' else "Cohen's d"
    
    # Load subject-level data if available
    if subject_ppi_data is not None:
        print(f"    Using subject-level data for {seed_name} ({metric})")
        
        # Merge subject data with contrast metadata to get tags
        subject_merged = pd.merge(
            subject_ppi_data,
            contrast_metadata[['task', 'contrast', 'tags']],
            left_on=['task', 'condition'],
            right_on=['task', 'contrast'],
            how='left'
        )
        subject_merged['parsed_tags'] = subject_merged['tags'].apply(parse_tags)
    else:
        subject_merged = None
    
    # Get top domains for each target
    top_dmn = domain_df[domain_df['mean_diff_dmn_minus_dan'] > 0].nlargest(top_n, 'abs_diff')
    top_dan = domain_df[domain_df['mean_diff_dmn_minus_dan'] < 0].nlargest(top_n, 'abs_diff')
    
    # Prepare violin data
    def prepare_violin_data_subjects(top_domains, seed_filter):
        """Extract individual subject values for each domain."""
        if subject_merged is None:
            # Fallback to contrast-level data
            violin_data = []
            for _, domain_row in top_domains.iterrows():
                domain = domain_row['cognitive_domain']
                domain_contrasts = diff_df[diff_df['parsed_tags'].apply(lambda x: domain in x if isinstance(x, list) else False)]
                for _, contrast_row in domain_contrasts.iterrows():
                    violin_data.append({
                        'domain': domain,
                        'diff_value': contrast_row[diff_col],
                        'task': contrast_row['task'],
                        'contrast': contrast_row['contrast'],
                        'source': 'contrast'
                    })
            return pd.DataFrame(violin_data)
        
        # Use subject-level data
        violin_data = []
        for _, domain_row in top_domains.iterrows():
            domain = domain_row['cognitive_domain']
            
            # Find all subject×run observations for this domain and seed
            for _, subj_row in subject_merged[
                (subject_merged['seed'] == seed_filter) &
                (subject_merged['parsed_tags'].apply(lambda x: domain in x if isinstance(x, list) else False))
            ].iterrows():
                # Calculate differential connectivity for this subject×run
                target_dmn_mask = (subject_merged['subject'] == subj_row['subject']) & \
                                 (subject_merged['task'] == subj_row['task']) & \
                                 (subject_merged['condition'] == subj_row['condition']) & \
                                 (subject_merged['seed'] == seed_filter) & \
                                 (subject_merged['target'] == 'DMN')
                target_dan_mask = (subject_merged['subject'] == subj_row['subject']) & \
                                 (subject_merged['task'] == subj_row['task']) & \
                                 (subject_merged['condition'] == subj_row['condition']) & \
                                 (subject_merged['seed'] == seed_filter) & \
                                 (subject_merged['target'] == 'DAN')
                
                dmn_data = subject_merged[target_dmn_mask]
                dan_data = subject_merged[target_dan_mask]
                
                if len(dmn_data) > 0 and len(dan_data) > 0:
                    if metric == 'effect':
                        diff_value = dmn_data.iloc[0]['ppi_beta'] - dan_data.iloc[0]['ppi_beta']
                    else:  # cohens_d - need to compute from betas
                        dmn_beta = dmn_data.iloc[0]['ppi_beta']
                        dan_beta = dan_data.iloc[0]['ppi_beta']
                        # Approximate Cohen's d (assuming similar variance)
                        diff_value = (dmn_beta - dan_beta) / (np.std([dmn_beta, dan_beta]) + 1e-10)
                    
                    violin_data.append({
                        'domain': domain,
                        'diff_value': diff_value,
                        'subject': subj_row['subject'],
                        'task': subj_row['task'],
                        'contrast': subj_row['condition'],
                        'source': 'subject'
                    })
        
        return pd.DataFrame(violin_data)
    
    dmn_violin_data = prepare_violin_data_subjects(top_dmn, seed_name)
    dan_violin_data = prepare_violin_data_subjects(top_dan, seed_name)
    
    # Plot 1: DMN-preferring domains
    if len(top_dmn) > 0 and len(dmn_violin_data) > 0:
        fig, ax = plt.subplots(figsize=(14, 10))
        
        parts = ax.violinplot(
            [dmn_violin_data[dmn_violin_data['domain'] == d]['diff_value'].values 
             for d in top_dmn['cognitive_domain']],
            positions=range(len(top_dmn)),
            vert=False,
            widths=0.7,
            showmeans=False,
            showextrema=False
        )
        
        for pc in parts['bodies']:
            pc.set_facecolor('#2ca02c')
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Overlay individual subject/run points
        for i, domain in enumerate(top_dmn['cognitive_domain']):
            domain_data = dmn_violin_data[dmn_violin_data['domain'] == domain]
            y_jitter = np.random.normal(i, 0.04, size=len(domain_data))
            
            # Different colors for subject vs contrast level
            if 'source' in domain_data.columns and (domain_data['source'] == 'subject').any():
                ax.scatter(domain_data['diff_value'], y_jitter, 
                          alpha=0.5, s=40, color='darkgreen', edgecolors='black', linewidth=0.3, zorder=3)
            else:
                ax.scatter(domain_data['diff_value'], y_jitter, 
                          alpha=0.7, s=60, color='darkgreen', edgecolors='black', linewidth=0.5, zorder=3)
        
        ax.set_yticks(range(len(top_dmn)))
        ax.set_yticklabels(top_dmn['cognitive_domain'], fontsize=14, fontweight='bold')
        ax.set_xlabel(f"DMN - DAN ({metric_label})", fontsize=14, fontweight='bold')
        
        data_source = "Subject-Level" if subject_merged is not None else "Contrast-Level"
        ax.set_title(f"{seed_name}: Top {top_n} Domains Preferring DMN Coupling\n({data_source} Variability)", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(left=min(0, dmn_violin_data['diff_value'].min() * 0.9))
        
        plt.tight_layout()
        plot_dmn_path = os.path.join(output_dir, f'{seed_name.lower()}_dmn_domains_{metric}_violin.png')
        plt.savefig(plot_dmn_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved {seed_name} DMN plot: {plot_dmn_path}")
    
    # Plot 2: DAN-preferring domains
    if len(top_dan) > 0 and len(dan_violin_data) > 0:
        fig, ax = plt.subplots(figsize=(14, 10))
        
        parts = ax.violinplot(
            [dan_violin_data[dan_violin_data['domain'] == d]['diff_value'].values 
             for d in top_dan['cognitive_domain']],
            positions=range(len(top_dan)),
            vert=False,
            widths=0.7,
            showmeans=False,
            showextrema=False
        )
        
        for pc in parts['bodies']:
            pc.set_facecolor('#ff7f0e')
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Overlay individual subject/run points
        for i, domain in enumerate(top_dan['cognitive_domain']):
            domain_data = dan_violin_data[dan_violin_data['domain'] == domain]
            y_jitter = np.random.normal(i, 0.04, size=len(domain_data))
            
            if 'source' in domain_data.columns and (domain_data['source'] == 'subject').any():
                ax.scatter(domain_data['diff_value'], y_jitter, 
                          alpha=0.5, s=40, color='darkorange', edgecolors='black', linewidth=0.3, zorder=3)
            else:
                ax.scatter(domain_data['diff_value'], y_jitter, 
                          alpha=0.7, s=60, color='darkorange', edgecolors='black', linewidth=0.5, zorder=3)
        
        ax.set_yticks(range(len(top_dan)))
        ax.set_yticklabels(top_dan['cognitive_domain'], fontsize=14, fontweight='bold')
        ax.set_xlabel(f"DMN - DAN ({metric_label})", fontsize=14, fontweight='bold')
        
        data_source = "Subject-Level" if subject_merged is not None else "Contrast-Level"
        ax.set_title(f"{seed_name}: Top {top_n} Domains Preferring DAN Coupling\n({data_source} Variability)", 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(right=max(0, dan_violin_data['diff_value'].max() * 0.9))
        
        plt.tight_layout()
        plot_dan_path = os.path.join(output_dir, f'{seed_name.lower()}_dan_domains_{metric}_violin.png')
        plt.savefig(plot_dan_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved {seed_name} DAN plot: {plot_dan_path}")

# Generate all violin plots
create_violin_plots_ppi(fpna_domains_effect, fpna_domain_data_effect, fpna_differential, 
                        'FPNA', metric='effect', output_dir=output_dir)
create_violin_plots_ppi(fpna_domains_cohens, fpna_domain_data_cohens, fpna_differential, 
                        'FPNA', metric='cohens_d', output_dir=output_dir)
create_violin_plots_ppi(fpnb_domains_effect, fpnb_domain_data_effect, fpnb_differential, 
                        'FPNB', metric='effect', output_dir=output_dir)
create_violin_plots_ppi(fpnb_domains_cohens, fpnb_domain_data_cohens, fpnb_differential, 
                        'FPNB', metric='cohens_d', output_dir=output_dir)

print("\n[9/9] ✓ Analysis complete!")
print(f"  Output directory: {output_dir}")