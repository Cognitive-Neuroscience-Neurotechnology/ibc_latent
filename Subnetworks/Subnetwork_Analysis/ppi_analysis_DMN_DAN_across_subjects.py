'''
Script to analyze PPI connectivity from FPN subnetworks to DMN/DAN across subjects and contrasts.
Generates statistical summaries, rankings, and visualizations for seed-target-contrast combinations.
Output files: seed_target_contrast_ranking.csv, seed_target_overview.png, contrast_heatmaps.png, 
              seed_comparison.png, analysis_summary.txt
Step 2/3 -> PPI per Contrast on Group-level
'''

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import sys
from statsmodels.stats.multitest import multipletests

# Configuration - read subjects from command line or process all
if len(sys.argv) > 1:
    subjects = sys.argv[1:]
else:
    # Auto-detect all subjects in ppi_results_dmn_dan
    ppi_base = "/ptmp/hmueller2/Downloads/ppi_results_dmn_dan"
    subjects = sorted([d.replace("sub-", "") for d in os.listdir(ppi_base) 
                      if os.path.isdir(os.path.join(ppi_base, d)) and d.startswith("sub-")])

PPI_BASE = "/ptmp/hmueller2/Downloads/ppi_results_dmn_dan"
OUTPUT_DIR = os.path.join(PPI_BASE, "group_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_subject_results(subjects):
    """Load per-run PPI results for all subjects."""
    results = []
    
    for subject in subjects:
        subject_dir = os.path.join(PPI_BASE, f"sub-{subject}")
        
        # Look for ppi_dmn_dan_results.csv (per-run, per-condition)
        results_file = os.path.join(subject_dir, "ppi_dmn_dan_results.csv")
        
        if os.path.exists(results_file):
            try:
                df = pd.read_csv(results_file)
                df['subject'] = subject  # Add this line to include subject ID
                results.append(df)
                print(f"Loaded {len(df)} run×condition×seed×target combinations from sub-{subject}")
            except Exception as e:
                print(f"Error loading {results_file}: {e}")
        else:
            print(f"Warning: {results_file} not found")
    
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def compute_subject_level_ffx(df):
    """Compute fixed-effects estimates for each subject×task×condition×seed×target."""
    ffx_results = []
    
    for (subject, task, condition, seed, target), group in df.groupby(['subject', 'task', 'condition', 'seed', 'target']):
        if len(group) < 1:
            continue
        
        # Fixed-effects across runs
        betas = group['beta'].values
        variances = group['variance'].values
        
        # Remove invalid values
        valid_mask = np.isfinite(betas) & np.isfinite(variances) & (variances > 0)
        betas = betas[valid_mask]
        variances = variances[valid_mask]
        
        if len(betas) == 0:
            continue
        
        if len(betas) == 1:
            beta_ffx = betas[0]
            var_ffx = variances[0]
        else:
            # Inverse-variance weighted average
            weights = 1.0 / variances
            beta_ffx = np.sum(betas * weights) / np.sum(weights)
            var_ffx = 1.0 / np.sum(weights)
        
        se_ffx = np.sqrt(var_ffx)
        t_ffx = beta_ffx / se_ffx if se_ffx > 0 else 0
        
        ffx_results.append({
            'subject': subject,
            'task': task,
            'condition': condition,
            'seed': seed,
            'target': target,
            'n_runs': len(betas),
            'beta_ffx': beta_ffx,
            'se_ffx': se_ffx,
            'variance_ffx': var_ffx,
            't_ffx': t_ffx
        })
    
    return pd.DataFrame(ffx_results)

def calculate_cohens_d(values):
    """Calculate Cohen's d as effect size relative to zero."""
    if len(values) < 2:
        return np.nan
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    return mean / std if std > 0 else 0

def perform_contrast_analysis(df):
    """Perform statistical analysis for each task×condition×seed×target combination across subjects."""
    combo_stats = []
    
    for (task, condition, seed, target), group in df.groupby(['task', 'condition', 'seed', 'target']):
        effects = group['beta_ffx'].dropna().values
        
        if len(effects) < 1:
            continue
        
        n_subjects = len(effects)
        
        if n_subjects == 1:
            # Single subject: report descriptive stats only
            t_stat = np.nan
            p_value = np.nan
            cohens_d = np.nan
            se_effect = np.nan
        else:
            # Multiple subjects: perform statistical tests
            t_stat, p_value = stats.ttest_1samp(effects, 0)
            cohens_d = calculate_cohens_d(effects)
            se_effect = np.std(effects, ddof=1) / np.sqrt(n_subjects)
        
        # Consistency (% subjects showing same direction)
        pos_count = np.sum(effects > 0)
        consistency = max(pos_count, n_subjects - pos_count) / n_subjects
        
        # Rename FPN1 -> FPNA, FPN2 -> FPNB
        seed_display = seed.replace('FPN1', 'FPNA').replace('FPN2', 'FPNB')
        
        combo_stats.append({
            'task': task,
            'condition': condition,
            'task_condition': f"{task}_{condition}",
            'seed': seed_display,
            'target': target,
            'seed_target': f"{seed_display}→{target}",
            'n_subjects': n_subjects,
            'mean_effect': np.mean(effects),
            'median_effect': np.median(effects),
            'std_effect': np.std(effects, ddof=1) if n_subjects > 1 else np.nan,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'consistency': consistency,
            'n_positive': pos_count,
            'n_negative': n_subjects - pos_count,
            'se_effect': se_effect,
            'min_effect': np.min(effects),
            'max_effect': np.max(effects)
        })
    
    return pd.DataFrame(combo_stats)

def apply_fdr_correction(combo_stats):
    """Apply FDR correction for multiple comparisons."""
    
    if combo_stats['p_value'].isna().all():
        print("⚠️  No p-values available for FDR correction")
        return combo_stats
    
    # Apply FDR within each seed→target pair (recommended for connectivity analyses)
    combo_stats['p_fdr'] = np.nan
    combo_stats['significant_fdr'] = False
    
    for pair in combo_stats['seed_target'].unique():
        mask = combo_stats['seed_target'] == pair
        p_vals = combo_stats.loc[mask, 'p_value'].values
        
        if np.isnan(p_vals).all():
            continue
        
        # Remove NaN values for FDR
        valid_mask = ~np.isnan(p_vals)
        valid_p = p_vals[valid_mask]
        
        if len(valid_p) == 0:
            continue
        
        # BH FDR correction
        reject, p_corrected, _, _ = multipletests(valid_p, alpha=0.05, method='fdr_bh')
        
        # Map back to original indices
        p_corrected_full = np.full(len(p_vals), np.nan)
        reject_full = np.full(len(p_vals), False)
        p_corrected_full[valid_mask] = p_corrected
        reject_full[valid_mask] = reject
        
        combo_stats.loc[mask, 'p_fdr'] = p_corrected_full
        combo_stats.loc[mask, 'significant_fdr'] = reject_full
    
    return combo_stats

def rank_combinations(combo_stats):
    """Rank task×condition×seed×target combinations by statistical strength."""
    combo_stats['abs_t_stat'] = combo_stats['t_statistic'].abs()
    combo_stats['abs_cohens_d'] = combo_stats['cohens_d'].abs()
    combo_stats['abs_mean_effect'] = combo_stats['mean_effect'].abs()
    
    # If we have p-values (multi-subject), use them for ranking
    if combo_stats['p_value'].notna().any():
        combo_stats['rank_score'] = (
            combo_stats['abs_t_stat'].fillna(0).rank(ascending=False) * 0.4 +
            combo_stats['abs_cohens_d'].fillna(0).rank(ascending=False) * 0.3 +
            combo_stats['consistency'].rank(ascending=False) * 0.3
        )
    else:
        # Single subject: rank by absolute effect size and consistency
        combo_stats['rank_score'] = (
            combo_stats['abs_mean_effect'].rank(ascending=False) * 0.6 +
            combo_stats['consistency'].rank(ascending=False) * 0.4
        )
    
    combo_stats = combo_stats.sort_values('rank_score')
    combo_stats['overall_rank'] = range(1, len(combo_stats) + 1)
    
    return combo_stats

def create_overview_visualizations(combo_stats, output_dir):
    """Create overview visualizations comparing seeds and targets."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Effect size by seed and target (averaged across task×conditions)
    ax = axes[0, 0]
    seed_target_summary = combo_stats.groupby(['seed', 'target'])['mean_effect'].mean().unstack()
    sns.heatmap(seed_target_summary, annot=True, fmt='.3f', cmap='RdBu_r', center=0, 
                ax=ax, cbar_kws={'label': 'Mean Effect'})
    ax.set_title('Average Effect Size by Seed→Target\n(Across All Task Contrasts)')
    ax.set_xlabel('Target Network')
    ax.set_ylabel('Seed Subnetwork')
    
    # 2. Number of significant effects by seed×target
    ax = axes[0, 1]
    if combo_stats['p_value'].notna().any():
        sig_counts = combo_stats[combo_stats['p_value'] < 0.05].groupby(['seed', 'target']).size().unstack(fill_value=0)
        sns.heatmap(sig_counts, annot=True, fmt='d', cmap='YlOrRd', ax=ax, 
                   cbar_kws={'label': 'N Significant Contrasts'})
        ax.set_title('Number of Significant Contrast Effects (p<0.05)\nby Seed→Target')
    else:
        total_counts = combo_stats.groupby(['seed', 'target']).size().unstack(fill_value=0)
        sns.heatmap(total_counts, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar_kws={'label': 'N Contrasts'})
        ax.set_title('Number of Contrasts Analyzed\nby Seed→Target')
    ax.set_xlabel('Target Network')
    ax.set_ylabel('Seed Subnetwork')
    
    # 3. Distribution of effects by seed
    ax = axes[1, 0]
    seed_data = []
    seed_labels = []
    for seed in sorted(combo_stats['seed'].unique()):
        seed_effects = combo_stats[combo_stats['seed'] == seed]['mean_effect'].values
        seed_data.append(seed_effects)
        seed_labels.append(f"{seed}\n(n={len(seed_effects)})")
    
    bp = ax.boxplot(seed_data, labels=seed_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_ylabel('PPI Effect (β)')
    ax.set_title('Effect Distribution by FPN Seed Subnetwork')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Distribution of effects by target
    ax = axes[1, 1]
    target_data = []
    target_labels = []
    for target in sorted(combo_stats['target'].unique()):
        target_effects = combo_stats[combo_stats['target'] == target]['mean_effect'].values
        target_data.append(target_effects)
        target_labels.append(f"{target}\n(n={len(target_effects)})")
    
    bp = ax.boxplot(target_data, labels=target_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_ylabel('PPI Effect (β)')
    ax.set_title('Effect Distribution by Target Network')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seed_target_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: seed_target_overview.png")

def create_contrast_heatmaps(combo_stats, output_dir):
    """Create separate heatmaps for positive and negative effects."""
    seed_target_pairs = sorted(combo_stats['seed_target'].unique())
    
    n_pairs = len(seed_target_pairs)
    
    # Create figure with 2 columns: positive effects (left), negative effects (right)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(20, 5*n_pairs))
    
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    
    for idx, pair in enumerate(seed_target_pairs):
        pair_data = combo_stats[combo_stats['seed_target'] == pair]
        
        # LEFT: Top 15 POSITIVE effects (increased connectivity)
        ax_pos = axes[idx, 0]
        pos_effects = pair_data[pair_data['mean_effect'] > 0].nlargest(15, 'mean_effect')
        
        if len(pos_effects) > 0:
            y_labels = [f"{row['task']}_{row['condition']}" for _, row in pos_effects.iterrows()]
            colors = ['darkgreen'] * len(pos_effects)
            
            ax_pos.barh(range(len(pos_effects)), pos_effects['mean_effect'], 
                       color=colors, alpha=0.7, edgecolor='black')
            ax_pos.set_yticks(range(len(pos_effects)))
            ax_pos.set_yticklabels(y_labels, fontsize=9)
            
            # Add error bars if available
            if 'se_effect' in pos_effects.columns and pos_effects['se_effect'].notna().any():
                ax_pos.errorbar(pos_effects['mean_effect'], range(len(pos_effects)), 
                               xerr=pos_effects['se_effect']*1.96, fmt='none', 
                               ecolor='black', capsize=3, alpha=0.5, linewidth=1.5)
            
            # Add significance markers
            if pos_effects['p_value'].notna().any():
                for i, (_, row) in enumerate(pos_effects.iterrows()):
                    if row['p_value'] < 0.001:
                        ax_pos.text(row['mean_effect'], i, ' ***', va='center', fontsize=10, fontweight='bold')
                    elif row['p_value'] < 0.01:
                        ax_pos.text(row['mean_effect'], i, ' **', va='center', fontsize=10, fontweight='bold')
                    elif row['p_value'] < 0.05:
                        ax_pos.text(row['mean_effect'], i, ' *', va='center', fontsize=10, fontweight='bold')
            
            ax_pos.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax_pos.set_xlabel('PPI Effect (β)', fontsize=11)
            ax_pos.set_title(f'{pair}: INCREASED Connectivity (Top 15)', fontsize=12, fontweight='bold')
            ax_pos.grid(axis='x', alpha=0.3)
            ax_pos.set_xlim(left=0)
        else:
            ax_pos.text(0.5, 0.5, 'No positive effects', transform=ax_pos.transAxes, 
                       ha='center', va='center', fontsize=14)
            ax_pos.set_title(f'{pair}: INCREASED Connectivity', fontsize=12)
        
        # RIGHT: Top 15 NEGATIVE effects (decreased connectivity)
        ax_neg = axes[idx, 1]
        neg_effects = pair_data[pair_data['mean_effect'] < 0].nsmallest(15, 'mean_effect')
        
        if len(neg_effects) > 0:
            y_labels = [f"{row['task']}_{row['condition']}" for _, row in neg_effects.iterrows()]
            colors = ['darkred'] * len(neg_effects)
            
            ax_neg.barh(range(len(neg_effects)), neg_effects['mean_effect'], 
                       color=colors, alpha=0.7, edgecolor='black')
            ax_neg.set_yticks(range(len(neg_effects)))
            ax_neg.set_yticklabels(y_labels, fontsize=9)
            
            if 'se_effect' in neg_effects.columns and neg_effects['se_effect'].notna().any():
                ax_neg.errorbar(neg_effects['mean_effect'], range(len(neg_effects)), 
                               xerr=neg_effects['se_effect']*1.96, fmt='none', 
                               ecolor='black', capsize=3, alpha=0.5, linewidth=1.5)
            
            if neg_effects['p_value'].notna().any():
                for i, (_, row) in enumerate(neg_effects.iterrows()):
                    if row['p_value'] < 0.001:
                        ax_neg.text(row['mean_effect'], i, ' ***', va='center', fontsize=10, fontweight='bold')
                    elif row['p_value'] < 0.01:
                        ax_neg.text(row['mean_effect'], i, ' **', va='center', fontsize=10, fontweight='bold')
                    elif row['p_value'] < 0.05:
                        ax_neg.text(row['mean_effect'], i, ' *', va='center', fontsize=10, fontweight='bold')
            
            ax_neg.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax_neg.set_xlabel('PPI Effect (β)', fontsize=11)
            ax_neg.set_title(f'{pair}: DECREASED Connectivity (Top 15)', fontsize=12, fontweight='bold')
            ax_neg.grid(axis='x', alpha=0.3)
            ax_neg.set_xlim(right=0)
        else:
            ax_neg.text(0.5, 0.5, 'No negative effects', transform=ax_neg.transAxes, 
                       ha='center', va='center', fontsize=14)
            ax_neg.set_title(f'{pair}: DECREASED Connectivity', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'contrast_heatmaps_polarity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: contrast_heatmaps_polarity.png (separate positive/negative)")

def create_comparison_plots(combo_stats, output_dir):
    """Create plots comparing FPNA vs FPNB connectivity to DMN vs DAN."""
    
    # Extract FPNA and FPNB effects for each target
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # DMN comparison
    ax = axes[0]
    dmn_data = combo_stats[combo_stats['target'] == 'DMN'].pivot(
        index='task_condition', columns='seed', values='mean_effect'
    )
    
    if 'FPNA' in dmn_data.columns and 'FPNB' in dmn_data.columns:
        dmn_data_clean = dmn_data.dropna()
        ax.scatter(dmn_data_clean['FPNA'], dmn_data_clean['FPNB'], s=100, alpha=0.6, color='steelblue')
        
        # Add diagonal line (y=x)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0, label='y=x')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.3)
        ax.set_xlabel('FPNA→DMN Effect')
        ax.set_ylabel('FPNB→DMN Effect')
        ax.set_title(f'FPN Subnetwork Connectivity to DMN\n(n={len(dmn_data_clean)} contrasts)')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # DAN comparison
    ax = axes[1]
    dan_data = combo_stats[combo_stats['target'] == 'DAN'].pivot(
        index='task_condition', columns='seed', values='mean_effect'
    )
    
    if 'FPNA' in dan_data.columns and 'FPNB' in dan_data.columns:
        dan_data_clean = dan_data.dropna()
        ax.scatter(dan_data_clean['FPNA'], dan_data_clean['FPNB'], s=100, alpha=0.6, color='coral')
        
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0, label='y=x')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.3)
        ax.set_xlabel('FPNA→DAN Effect')
        ax.set_ylabel('FPNB→DAN Effect')
        ax.set_title(f'FPN Subnetwork Connectivity to DAN\n(n={len(dan_data_clean)} contrasts)')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seed_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: seed_comparison.png")

def create_specificity_plots(combo_stats, output_dir):
    """
    Identify contrasts that are SPECIFIC to particular seed→target combinations.
    Shows which contrasts distinguish FPNA-DMN from FPNB-DMN, etc.
    """
    
    # Calculate differential effects for each contrast
    contrast_diffs = []
    
    for contrast in combo_stats['task_condition'].unique():
        contrast_data = combo_stats[combo_stats['task_condition'] == contrast]
        
        # Get effects for each seed→target combination
        effects_dict = {}
        for _, row in contrast_data.iterrows():
            effects_dict[f"{row['seed']}_{row['target']}"] = row['mean_effect']
        
        # Only proceed if we have all 4 combinations
        if len(effects_dict) == 4:
            fpna_dmn = effects_dict.get('FPNA_DMN', np.nan)
            fpnb_dmn = effects_dict.get('FPNB_DMN', np.nan)
            fpna_dan = effects_dict.get('FPNA_DAN', np.nan)
            fpnb_dan = effects_dict.get('FPNB_DAN', np.nan)
            
            # Calculate seed specificity (FPNA vs FPNB) for each target
            dmn_seed_diff = fpna_dmn - fpnb_dmn  # Positive = FPNA-specific for DMN
            dan_seed_diff = fpna_dan - fpnb_dan  # Positive = FPNA-specific for DAN
            
            # Calculate target specificity (DMN vs DAN) for each seed
            fpna_target_diff = fpna_dmn - fpna_dan  # Positive = DMN-specific for FPNA
            fpnb_target_diff = fpnb_dmn - fpnb_dan  # Positive = DMN-specific for FPNB
            
            # Calculate overall "uniqueness" - how different is one combination from others?
            fpna_dmn_unique = abs(fpna_dmn - fpnb_dmn) + abs(fpna_dmn - fpna_dan)
            fpnb_dmn_unique = abs(fpnb_dmn - fpna_dmn) + abs(fpnb_dmn - fpnb_dan)
            fpna_dan_unique = abs(fpna_dan - fpnb_dan) + abs(fpna_dan - fpna_dmn)
            fpnb_dan_unique = abs(fpnb_dan - fpna_dan) + abs(fpnb_dan - fpnb_dmn)
            
            contrast_diffs.append({
                'task_condition': contrast,
                'FPNA_DMN': fpna_dmn,
                'FPNB_DMN': fpnb_dmn,
                'FPNA_DAN': fpna_dan,
                'FPNB_DAN': fpnb_dan,
                'DMN_seed_specificity': dmn_seed_diff,  # FPNA>FPNB for DMN
                'DAN_seed_specificity': dan_seed_diff,  # FPNA>FPNB for DAN
                'FPNA_target_specificity': fpna_target_diff,  # DMN>DAN for FPNA
                'FPNB_target_specificity': fpnb_target_diff,  # DMN>DAN for FPNB
                'FPNA_DMN_uniqueness': fpna_dmn_unique,
                'FPNB_DMN_uniqueness': fpnb_dmn_unique,
                'FPNA_DAN_uniqueness': fpna_dan_unique,
                'FPNB_DAN_uniqueness': fpnb_dan_unique
            })
    
    df_diffs = pd.DataFrame(contrast_diffs)
    
    if df_diffs.empty:
        print("⚠️  Not enough data for specificity analysis (need all 4 seed×target combinations)")
        return
    
    # Create 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # ===== PLOT 1: FPNA-DMN vs FPNB-DMN Specificity =====
    ax = axes[0, 0]
    top_fpna_dmn = df_diffs.nlargest(10, 'DMN_seed_specificity')  # Most FPNA>FPNB for DMN
    top_fpnb_dmn = df_diffs.nsmallest(10, 'DMN_seed_specificity')  # Most FPNB>FPNA for DMN
    
    combined_dmn = pd.concat([top_fpna_dmn, top_fpnb_dmn]).sort_values('DMN_seed_specificity')
    
    colors = ['steelblue' if x > 0 else 'coral' for x in combined_dmn['DMN_seed_specificity']]
    ax.barh(range(len(combined_dmn)), combined_dmn['DMN_seed_specificity'], 
           color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(combined_dmn)))
    ax.set_yticklabels(combined_dmn['task_condition'], fontsize=8)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax.set_xlabel('FPNA−FPNB Difference (β)', fontsize=11)
    ax.set_title('Contrasts Distinguishing FPNA vs FPNB for DMN Connectivity\n' + 
                '(Blue = FPNA-specific, Orange = FPNB-specific)', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # ===== PLOT 2: FPNA-DAN vs FPNB-DAN Specificity =====
    ax = axes[0, 1]
    top_fpna_dan = df_diffs.nlargest(10, 'DAN_seed_specificity')
    top_fpnb_dan = df_diffs.nsmallest(10, 'DAN_seed_specificity')
    
    combined_dan = pd.concat([top_fpna_dan, top_fpnb_dan]).sort_values('DAN_seed_specificity')
    
    colors = ['steelblue' if x > 0 else 'coral' for x in combined_dan['DAN_seed_specificity']]
    ax.barh(range(len(combined_dan)), combined_dan['DAN_seed_specificity'], 
           color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(combined_dan)))
    ax.set_yticklabels(combined_dan['task_condition'], fontsize=8)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax.set_xlabel('FPNA−FPNB Difference (β)', fontsize=11)
    ax.set_title('Contrasts Distinguishing FPNA vs FPNB for DAN Connectivity\n' + 
                '(Blue = FPNA-specific, Orange = FPNB-specific)', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # ===== PLOT 3: Unique FPNA-DMN Contrasts =====
    ax = axes[1, 0]
    top_unique_fpna_dmn = df_diffs.nlargest(15, 'FPNA_DMN_uniqueness')
    
    # Color by the actual FPNA-DMN effect direction
    colors = ['darkgreen' if x > 0 else 'darkred' for x in top_unique_fpna_dmn['FPNA_DMN']]
    ax.barh(range(len(top_unique_fpna_dmn)), top_unique_fpna_dmn['FPNA_DMN_uniqueness'], 
           color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(top_unique_fpna_dmn)))
    ax.set_yticklabels(top_unique_fpna_dmn['task_condition'], fontsize=8)
    ax.set_xlabel('Uniqueness Score (|diff from others|)', fontsize=11)
    ax.set_title('Top 15 FPNA→DMN Specific Contrasts\n' +
                '(Green = positive effect, Red = negative effect)', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # ===== PLOT 4: Unique FPNB-DAN Contrasts =====
    ax = axes[1, 1]
    top_unique_fpnb_dan = df_diffs.nlargest(15, 'FPNB_DAN_uniqueness')
    
    colors = ['darkgreen' if x > 0 else 'darkred' for x in top_unique_fpnb_dan['FPNB_DAN']]
    ax.barh(range(len(top_unique_fpnb_dan)), top_unique_fpnb_dan['FPNB_DAN_uniqueness'], 
           color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(top_unique_fpnb_dan)))
    ax.set_yticklabels(top_unique_fpnb_dan['task_condition'], fontsize=8)
    ax.set_xlabel('Uniqueness Score (|diff from others|)', fontsize=11)
    ax.set_title('Top 15 FPNB→DAN Specific Contrasts\n' +
                '(Green = positive effect, Red = negative effect)', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'contrast_specificity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the differential scores as CSV
    df_diffs.to_csv(os.path.join(output_dir, 'contrast_specificity_scores.csv'), index=False)
    
    print(f"✓ Saved: contrast_specificity.png")
    print(f"✓ Saved: contrast_specificity_scores.csv")

def generate_summary_report(combo_stats, df_ffx, output_dir):
    """Generate interpretable summary report."""
    report = []
    report.append("="*80)
    report.append("FPN→DMN/DAN PPI CONNECTIVITY ANALYSIS - SUMMARY (BY CONTRAST)")
    report.append("="*80)
    report.append("")
    
    n_subjects = df_ffx['subject'].nunique()
    n_contrasts = combo_stats['task_condition'].nunique()
    n_tasks = combo_stats['task'].nunique()
    
    report.append(f"Total Subjects: {n_subjects}")
    report.append(f"Total Tasks: {n_tasks}")
    report.append(f"Total Task Contrasts: {n_contrasts}")
    report.append(f"Total Seed×Target×Contrast Combinations: {len(combo_stats)}")
    
    if n_subjects > 1 and combo_stats['p_value'].notna().any():
        sig_combos = sum(combo_stats['p_value'] < 0.05)
        report.append(f"Significant Effects (p<0.05): {sig_combos}")
    report.append("")
    
    # Summary by seed→target pair
    for pair in sorted(combo_stats['seed_target'].unique()):
        report.append("-" * 80)
        report.append(f"{pair.upper()} CONNECTIVITY")
        report.append("-" * 80)
        
        pair_data = combo_stats[combo_stats['seed_target'] == pair]
        
        # Overall statistics
        mean_across_contrasts = pair_data['mean_effect'].mean()
        std_across_contrasts = pair_data['mean_effect'].std()
        
        report.append(f"\nOverall Statistics:")
        report.append(f"  Mean effect across contrasts: {mean_across_contrasts:.4f} ± {std_across_contrasts:.4f}")
        report.append(f"  Number of contrasts: {len(pair_data)}")
        
        if pair_data['p_value'].notna().any():
            sig_contrasts = sum(pair_data['p_value'] < 0.05)
            report.append(f"  Significant contrasts (p<0.05): {sig_contrasts}")
        
        # Top positive effects
        report.append(f"\nTop 5 Contrasts with INCREASED Connectivity:")
        top_pos = pair_data[pair_data['mean_effect'] > 0].nlargest(5, 'mean_effect')
        for idx, row in top_pos.iterrows():
            if n_subjects > 1 and np.isfinite(row['t_statistic']):
                report.append(f"  • {row['task']}_{row['condition']}: β={row['mean_effect']:.4f}, t={row['t_statistic']:.2f}, p={row['p_value']:.4f}")
            else:
                report.append(f"  • {row['task']}_{row['condition']}: β={row['mean_effect']:.4f}")
        
        # Top negative effects
        report.append(f"\nTop 5 Contrasts with DECREASED Connectivity:")
        top_neg = pair_data[pair_data['mean_effect'] < 0].nsmallest(5, 'mean_effect')
        for idx, row in top_neg.iterrows():
            if n_subjects > 1 and np.isfinite(row['t_statistic']):
                report.append(f"  • {row['task']}_{row['condition']}: β={row['mean_effect']:.4f}, t={row['t_statistic']:.2f}, p={row['p_value']:.4f}")
            else:
                report.append(f"  • {row['task']}_{row['condition']}: β={row['mean_effect']:.4f}")
        
        report.append("")
    
    # Cross-seed comparison
    report.append("-" * 80)
    report.append("COMPARING FPNA vs FPNB")
    report.append("-" * 80)
    
    for target in ['DMN', 'DAN']:
        report.append(f"\n{target}:")
        fpna_effects = combo_stats[(combo_stats['seed'] == 'FPNA') & (combo_stats['target'] == target)]['mean_effect']
        fpnb_effects = combo_stats[(combo_stats['seed'] == 'FPNB') & (combo_stats['target'] == target)]['mean_effect']
        
        if len(fpna_effects) > 0 and len(fpnb_effects) > 0:
            report.append(f"  FPNA→{target}: mean={fpna_effects.mean():.4f}, std={fpna_effects.std():.4f}")
            report.append(f"  FPNB→{target}: mean={fpnb_effects.mean():.4f}, std={fpnb_effects.std():.4f}")
            
            # Paired comparison if same contrasts
            matched_data = combo_stats[combo_stats['target'] == target].pivot(
                index='task_condition', columns='seed', values='mean_effect'
            )
            if 'FPNA' in matched_data.columns and 'FPNB' in matched_data.columns:
                matched_clean = matched_data.dropna()
                if len(matched_clean) > 1:
                    t_stat, p_val = stats.ttest_rel(matched_clean['FPNA'], matched_clean['FPNB'])
                    report.append(f"  Paired t-test (n={len(matched_clean)} contrasts): t={t_stat:.3f}, p={p_val:.4f}")
    
    report.append("\n" + "="*80)
    
    report_text = "\n".join(report)
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
        f.write(report_text)
    
    print(report_text)

def assess_statistical_power(combo_stats, df_ffx):
    """Assess overall statistical power and effect size benchmarks."""
    
    power_report = []
    power_report.append("="*80)
    power_report.append("STATISTICAL POWER AND EFFECT SIZE ASSESSMENT")
    power_report.append("="*80)
    power_report.append("")
    
    # Cohen's d benchmarks: 0.2=small, 0.5=medium, 0.8=large
    # For connectivity, effects are typically smaller, so adjusted thresholds:
    # 0.1=small, 0.3=medium, 0.5=large
    
    for pair in sorted(combo_stats['seed_target'].unique()):
        pair_data = combo_stats[combo_stats['seed_target'] == pair]
        
        power_report.append(f"\n{pair}:")
        power_report.append("-" * 40)
        
        # Count by effect size category
        small_effects = sum((pair_data['cohens_d'].abs() >= 0.1) & (pair_data['cohens_d'].abs() < 0.3))
        medium_effects = sum((pair_data['cohens_d'].abs() >= 0.3) & (pair_data['cohens_d'].abs() < 0.5))
        large_effects = sum(pair_data['cohens_d'].abs() >= 0.5)
        
        power_report.append(f"  Small effects (|d|≥0.1): {small_effects}")
        power_report.append(f"  Medium effects (|d|≥0.3): {medium_effects}")
        power_report.append(f"  Large effects (|d|≥0.5): {large_effects}")
        
        # Statistical significance
        if pair_data['p_value'].notna().any():
            sig_001 = sum(pair_data['p_value'] < 0.001)
            sig_01 = sum((pair_data['p_value'] >= 0.001) & (pair_data['p_value'] < 0.01))
            sig_05 = sum((pair_data['p_value'] >= 0.01) & (pair_data['p_value'] < 0.05))
            
            power_report.append(f"\n  p<0.001: {sig_001}")
            power_report.append(f"  p<0.01: {sig_01}")
            power_report.append(f"  p<0.05: {sig_05}")
        
        # Replicability: consistency across subjects
        consistent_effects = sum(pair_data['consistency'] >= 0.75)
        power_report.append(f"\n  Consistent across ≥75% subjects: {consistent_effects}")
        
        # "Significant AND meaningful" effects
        if pair_data['p_value'].notna().any():
            meaningful = sum((pair_data['p_value'] < 0.05) & (pair_data['cohens_d'].abs() >= 0.3))
            power_report.append(f"  Significant (p<0.05) AND medium+ effect: {meaningful}")
    
    return "\n".join(power_report)

def permutation_test_contrast(df_ffx, n_permutations=1000):
    """
    Test if observed contrast effects are beyond chance using permutation testing.
    Randomly shuffles subject labels to create null distribution.
    """
    
    perm_results = []
    
    for (task, condition, seed, target), group in df_ffx.groupby(['task', 'condition', 'seed', 'target']):
        if len(group) < 3:  # Need at least 3 subjects
            continue
        
        observed_mean = group['beta_ffx'].mean()
        
        # Generate null distribution
        null_distribution = []
        for _ in range(n_permutations):
            # Randomly shuffle signs (flip connectivity direction)
            shuffled = group['beta_ffx'].values * np.random.choice([-1, 1], size=len(group))
            null_distribution.append(shuffled.mean())
        
        null_distribution = np.array(null_distribution)
        
        # Compute p-value
        if observed_mean >= 0:
            p_perm = np.mean(null_distribution >= observed_mean)
        else:
            p_perm = np.mean(null_distribution <= observed_mean)
        
        # Two-tailed
        p_perm_twotail = 2 * min(p_perm, 1 - p_perm)
        
        perm_results.append({
            'task': task,
            'condition': condition,
            'seed': seed,
            'target': target,
            'observed_mean': observed_mean,
            'p_permutation': p_perm_twotail,
            'n_subjects': len(group)
        })
    
    return pd.DataFrame(perm_results)

def main():
    """Main analysis pipeline."""
    print(f"Found subjects: {subjects}")
    print(f"\nLoading per-run results from {PPI_BASE}...")
    df = load_subject_results(subjects)
    
    if df.empty:
        print("No data loaded. Check subject directories for ppi_dmn_dan_results.csv files.")
        return
    
    print(f"Loaded {len(df)} run-level results")
    print(f"Computing subject-level fixed-effects...")
    df_ffx = compute_subject_level_ffx(df)
    
    print(f"Computed FFX for {df_ffx['subject'].nunique()} subjects, {df_ffx['task'].nunique()} tasks, {df_ffx['condition'].nunique()} conditions")
    print(f"Seed×Target combinations: {df_ffx.groupby(['seed', 'target']).size()}")
    
    print("\nPerforming permutation tests...")
    perm_results = permutation_test_contrast(df_ffx, n_permutations=1000)
    perm_results.to_csv(os.path.join(OUTPUT_DIR, 'permutation_test_results.csv'), index=False)

    print("\nPerforming contrast-level statistical analysis...")
    combo_stats = perform_contrast_analysis(df_ffx)
    
    if combo_stats.empty:
        print("No valid combinations found for analysis.")
        return
    
    # Merge permutation results
    combo_stats = combo_stats.merge(
        perm_results[['task', 'condition', 'seed', 'target', 'p_permutation']], 
        on=['task', 'condition', 'seed', 'target'],
        how='left'
    )

    print("Ranking combinations...")
    combo_stats = rank_combinations(combo_stats)
    
    print("Applying FDR correction for multiple comparisons...")
    combo_stats = apply_fdr_correction(combo_stats)
    
    # Report FDR-corrected significant effects
    if 'significant_fdr' in combo_stats.columns:
        n_sig_fdr = combo_stats['significant_fdr'].sum()
        print(f"  Significant after FDR correction (q<0.05): {n_sig_fdr}/{len(combo_stats)}")
    
    # Save ranking
    combo_stats.to_csv(os.path.join(OUTPUT_DIR, 'seed_target_contrast_ranking.csv'), index=False)
    print(f"\n✓ Saved seed_target_contrast_ranking.csv with {len(combo_stats)} contrast combinations")
    
    print("\nCreating visualizations...")
    create_overview_visualizations(combo_stats, OUTPUT_DIR)
    create_contrast_heatmaps(combo_stats, OUTPUT_DIR)
    create_comparison_plots(combo_stats, OUTPUT_DIR)
    create_specificity_plots(combo_stats, OUTPUT_DIR)
    
    print("\nGenerating summary report...")
    generate_summary_report(combo_stats, df_ffx, OUTPUT_DIR)
    
    print("\nAssessing statistical power...")
    power_report = assess_statistical_power(combo_stats, df_ffx)
    with open(os.path.join(OUTPUT_DIR, 'statistical_power_report.txt'), 'w') as f:
        f.write(power_report)
    print(power_report)
    
    print(f"\n✓ Analysis complete! Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()