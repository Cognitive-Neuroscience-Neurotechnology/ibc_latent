'''
Script to analyze PPI subnetwork distinction across subjects and tasks.
Generates statistical summaries, rankings, and visualizations.
OLD!! - Now using _DMN_DAN
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

# Configuration - read subjects from command line or process all
if len(sys.argv) > 1:
    subjects = sys.argv[1:]
else:
    # Auto-detect all subjects in ppi_results
    ppi_base = "/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/old_versions/ppi_results"
    subjects = sorted([d.replace("sub-", "") for d in os.listdir(ppi_base) 
                      if os.path.isdir(os.path.join(ppi_base, d)) and d.startswith("sub-")])

PPI_BASE = "/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/old_versions/ppi_results"
OUTPUT_DIR = os.path.join(PPI_BASE, "group_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_subject_ffx_results(subjects):
    """Load FFX PPI results for all subjects."""
    results = []
    
    for subject in subjects:
        subject_dir = os.path.join(PPI_BASE, f"sub-{subject}")
        
        # Look for task_level_ppi_ffx_summary.csv
        ffx_file = os.path.join(subject_dir, "task_level_ppi_ffx_summary.csv")
        
        if os.path.exists(ffx_file):
            try:
                df = pd.read_csv(ffx_file)
                results.append(df)
            except Exception as e:
                print(f"Error loading {ffx_file}: {e}")
    
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def calculate_cohens_d(values):
    """Calculate Cohen's d as effect size relative to zero."""
    if len(values) < 2:
        return np.nan
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    return mean / std if std > 0 else 0

def perform_task_analysis(df):
    """Perform statistical analysis for each task across subjects."""
    task_stats = []
    
    for task in df['task'].unique():
        task_data = df[df['task'] == task]
        
        # Get beta_ffx (PPI effect) values
        if 'beta_ffx' not in task_data.columns:
            continue
        
        effects = task_data['beta_ffx'].dropna().values
        
        # Allow single-subject analysis OR group analysis with >=2 subjects
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
        
        # Separate positive and negative effects
        pos_effects = effects[effects > 0]
        neg_effects = effects[effects < 0]
        
        task_stats.append({
            'task': task,
            'n_subjects': n_subjects,
            'mean_effect': np.mean(effects),
            'std_effect': np.std(effects, ddof=1) if n_subjects > 1 else np.nan,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'consistency': consistency,
            'n_positive': len(pos_effects),
            'n_negative': len(neg_effects),
            'mean_pos_effect': np.mean(pos_effects) if len(pos_effects) > 0 else np.nan,
            'mean_neg_effect': np.mean(neg_effects) if len(neg_effects) > 0 else np.nan,
            'se_effect': se_effect,
        })
    
    return pd.DataFrame(task_stats)

def rank_tasks(task_stats):
    """Rank tasks by statistical strength or effect size (for single subjects)."""
    task_stats['abs_t_stat'] = task_stats['t_statistic'].abs()
    task_stats['abs_cohens_d'] = task_stats['cohens_d'].abs()
    task_stats['abs_mean_effect'] = task_stats['mean_effect'].abs()
    
    # If we have p-values (multi-subject), use them for ranking
    if task_stats['p_value'].notna().any():
        # Combined ranking (weighted by t-stat, cohens_d, and consistency)
        task_stats['rank_score'] = (
            task_stats['abs_t_stat'].fillna(0).rank(ascending=False) * 0.4 +
            task_stats['abs_cohens_d'].fillna(0).rank(ascending=False) * 0.3 +
            task_stats['consistency'].rank(ascending=False) * 0.3
        )
    else:
        # Single subject: rank by absolute effect size and consistency
        task_stats['rank_score'] = (
            task_stats['abs_mean_effect'].rank(ascending=False) * 0.6 +
            task_stats['consistency'].rank(ascending=False) * 0.4
        )
    
    task_stats = task_stats.sort_values('rank_score')
    task_stats['overall_rank'] = range(1, len(task_stats) + 1)
    
    return task_stats

def create_visualizations(task_stats, df, output_dir):
    """Create comprehensive visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Task ranking by effect size or t-statistic
    if task_stats['t_statistic'].notna().any():
        sorted_tasks = task_stats.sort_values('t_statistic', key=abs)
        x_col = 't_statistic'
        x_label = 't-statistic'
    else:
        sorted_tasks = task_stats.sort_values('mean_effect', key=abs)
        x_col = 'mean_effect'
        x_label = 'Mean Effect'
    
    ax = axes[0, 0]
    colors = ['green' if x > 0 else 'red' for x in sorted_tasks[x_col]]
    ax.barh(sorted_tasks['task'], sorted_tasks[x_col], color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel(x_label)
    ax.set_title(f'Tasks Ranked by {x_label}')
    ax.grid(axis='x', alpha=0.3)
    
    # 2. Effect sizes
    ax = axes[0, 1]
    sorted_tasks_e = task_stats.sort_values('mean_effect', key=abs)
    colors = ['green' if x > 0 else 'red' for x in sorted_tasks_e['mean_effect']]
    ax.barh(sorted_tasks_e['task'], sorted_tasks_e['mean_effect'], color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Mean Effect")
    ax.set_title("Effect Sizes by Task")
    ax.grid(axis='x', alpha=0.3)
    
    # 3. Consistency
    ax = axes[1, 0]
    sorted_tasks_cons = task_stats.sort_values('consistency', ascending=True)
    ax.barh(sorted_tasks_cons['task'], sorted_tasks_cons['consistency'], color='steelblue', alpha=0.7)
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1, label='Chance (50%)')
    ax.set_xlabel('Consistency')
    ax.set_title('Effect Direction Consistency')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # 4. Mean effect vs number of subjects
    ax = axes[1, 1]
    ax.scatter(task_stats['n_subjects'], task_stats['mean_effect'], 
              s=100, alpha=0.6, c=task_stats['consistency'], cmap='RdYlGn')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Number of Subjects')
    ax.set_ylabel('Mean Effect')
    ax.set_title('Effect Size vs Sample Size')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'task_ranking_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization: task_ranking_overview.png")

def create_violin_plots(df, task_stats, output_dir):
    """Create violin plots for top and bottom tasks."""
    top_tasks = task_stats.nlargest(3, 'mean_effect')['task'].values
    bottom_tasks = task_stats.nsmallest(3, 'mean_effect')['task'].values
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Top tasks
    top_data = df[df['task'].isin(top_tasks)]
    if len(top_data) > 0:
        sns.violinplot(data=top_data, x='task', y='beta_ffx', ax=axes[0])
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[0].set_title('Top 3 Tasks (Strongest Positive Effects)')
        axes[0].set_ylabel('PPI Effect (β)')
    
    # Bottom tasks
    bottom_data = df[df['task'].isin(bottom_tasks)]
    if len(bottom_data) > 0:
        sns.violinplot(data=bottom_data, x='task', y='beta_ffx', ax=axes[1])
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[1].set_title('Bottom 3 Tasks (Strongest Negative Effects)')
        axes[1].set_ylabel('PPI Effect (β)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'task_distributions_top_bottom.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization: task_distributions_top_bottom.png")

def create_scatter_plot(df, task_stats, output_dir):
    """Create scatter plot showing individual subject effects by task."""
    top_tasks = task_stats.nlargest(6, 'mean_effect')['task'].values
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, task in enumerate(top_tasks):
        ax = axes[idx]
        task_data = df[df['task'] == task]
        
        # Scatter plot with task-level mean
        y_vals = task_data['beta_ffx'].values
        x_vals = np.random.normal(0, 0.04, size[len(y_vals)])
        
        ax.scatter(x_vals, y_vals, alpha=0.6, s=80, color='steelblue')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=y_vals.mean(), color='green', linestyle='-', linewidth=2, label=f'Mean={y_vals.mean():.4f}')
        
        ax.set_xlim(-0.3, 0.3)
        ax.set_xticks([])
        ax.set_ylabel('PPI Effect (β)')
        ax.set_title(f'{task} (n={len(y_vals)})')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'task_scatter_individual_subjects.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization: task_scatter_individual_subjects.png")

def generate_summary_report(task_stats, df, output_dir):
    """Generate interpretable summary report."""
    report = []
    report.append("="*70)
    report.append("PPI SUBNETWORK DISTINCTION ANALYSIS - SUMMARY")
    report.append("="*70)
    report.append("")
    
    n_subjects = df['subject'].nunique()
    n_tasks = len(task_stats)
    
    # Overall statistics
    report.append(f"Total Subjects: {n_subjects}")
    report.append(f"Total Tasks Analyzed: {n_tasks}")
    
    if n_subjects > 1 and task_stats['p_value'].notna().any():
        sig_tasks = sum(task_stats['p_value'] < 0.05)
        report.append(f"Tasks with Significant Effects (p<0.05): {sig_tasks}")
    report.append("")
    
    # Top tasks for positive effects
    report.append("-" * 70)
    report.append("TOP TASKS WITH POSITIVE EFFECTS (Subnetwork Coupling)")
    report.append("-" * 70)
    top_pos = task_stats[task_stats['mean_effect'] > 0].nlargest(5, 'mean_effect')
    for idx, row in top_pos.iterrows():
        report.append(f"\n{row['task'].upper()}")
        report.append(f"  Mean Effect: {row['mean_effect']:.4f}")
        if row['n_subjects'] > 1:
            report.append(f"  Std Dev: {row['std_effect']:.4f}")
            if np.isfinite(row['se_effect']):
                report.append(f"  95% CI: [{row['mean_effect'] - 1.96*row['se_effect']:.4f}, {row['mean_effect'] + 1.96*row['se_effect']:.4f}]")
            if np.isfinite(row['t_statistic']):
                report.append(f"  t-statistic: {row['t_statistic']:.3f}, p-value: {row['p_value']:.4f}")
            if np.isfinite(row['cohens_d']):
                report.append(f"  Cohen's d: {row['cohens_d']:.3f}")
        report.append(f"  Consistency: {row['consistency']*100:.1f}% ({row['n_positive']}/{row['n_subjects']} subjects)")
    
    report.append("\n" + "-" * 70)
    report.append("TOP TASKS WITH NEGATIVE EFFECTS (Subnetwork Decoupling)")
    report.append("-" * 70)
    top_neg = task_stats[task_stats['mean_effect'] < 0].nsmallest(5, 'mean_effect')
    for idx, row in top_neg.iterrows():
        report.append(f"\n{row['task'].upper()}")
        report.append(f"  Mean Effect: {row['mean_effect']:.4f}")
        if row['n_subjects'] > 1:
            report.append(f"  Std Dev: {row['std_effect']:.4f}")
            if np.isfinite(row['se_effect']):
                report.append(f"  95% CI: [{row['mean_effect'] - 1.96*row['se_effect']:.4f}, {row['mean_effect'] + 1.96*row['se_effect']:.4f}]")
            if np.isfinite(row['t_statistic']):
                report.append(f"  t-statistic: {row['t_statistic']:.3f}, p-value: {row['p_value']:.4f}")
            if np.isfinite(row['cohens_d']):
                report.append(f"  Cohen's d: {row['cohens_d']:.3f}")
        report.append(f"  Consistency: {row['consistency']*100:.1f}% ({row['n_negative']}/{row['n_subjects']} subjects)")
    
    # Most reliable tasks
    report.append("\n" + "-" * 70)
    report.append("MOST RELIABLE TASKS (Highest Consistency)")
    report.append("-" * 70)
    most_reliable = task_stats.nlargest(5, 'consistency')
    for idx, row in most_reliable.iterrows():
        report.append(f"\n{row['task'].upper()}")
        report.append(f"  Consistency: {row['consistency']*100:.1f}%")
        report.append(f"  Mean Effect: {row['mean_effect']:.4f}")
        report.append(f"  n_subjects: {row['n_subjects']}")
    
    report.append("\n" + "="*70)
    
    report_text = "\n".join(report)
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
        f.write(report_text)
    
    print(report_text)

def create_anchor_panels(df, task_stats, output_dir):
    """Create violin plots for anchor tasks with individual subjects visible."""
    # Define anchor tasks
    positive_anchors = ['Checkerboard', 'HCPMotor']
    negative_anchors = ['PreferencePaintings', 'ContRing']
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    all_anchors = positive_anchors + negative_anchors
    colors = ['green', 'green', 'red', 'red']
    
    for idx, (task, color) in enumerate(zip(all_anchors, colors)):
        ax = axes[idx]
        task_data = df[df['task'] == task]
        
        if len(task_data) == 0:
            ax.text(0.5, 0.5, f'{task}\nNo data', ha='center', va='center')
            ax.set_xlim(-0.5, 0.5)
            continue
        
        effects = task_data['beta_ffx'].values
        
        # Violin plot
        parts = ax.violinplot([effects], positions=[0], widths=0.7, 
                              showmeans=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.3)
        
        # Individual subject dots with jitter
        x_jitter = np.random.normal(0, 0.04, size=len(effects))
        ax.scatter(x_jitter, effects, alpha=0.7, s=60, color=color, 
                  edgecolors='black', linewidths=0.5, zorder=3)
        
        # Mean and CI
        mean_eff = np.mean(effects)
        if len(effects) > 1:
            se = np.std(effects, ddof=1) / np.sqrt(len(effects))
            ci_lower = mean_eff - 1.96 * se
            ci_upper = mean_eff + 1.96 * se
            
            # Mean line
            ax.hlines(mean_eff, -0.3, 0.3, colors='black', linewidth=2.5, zorder=4)
            # CI error bar
            ax.vlines(0, ci_lower, ci_upper, colors='black', linewidth=2, zorder=4)
            ax.plot([0], [ci_lower], marker='_', color='black', markersize=10, zorder=4)
            ax.plot([0], [ci_upper], marker='_', color='black', markersize=10, zorder=4)
        else:
            ax.hlines(mean_eff, -0.3, 0.3, colors='black', linewidth=2.5, zorder=4)
        
        # Zero line
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Formatting
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylabel('PPI Effect (β)' if idx == 0 else '')
        ax.set_title(f'{task}\n(n={len(effects)})', fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Anchor Tasks: Strong Positive and Negative Effects', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anchor_tasks_panel.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization: anchor_tasks_panel.png")

def create_effect_precision_scatter(task_stats, output_dir):
    """Create scatter plot of effect size vs precision (SE)."""
    # Filter out tasks with missing SE
    valid_stats = task_stats[task_stats['se_effect'].notna() & (task_stats['n_subjects'] > 1)].copy()
    
    if len(valid_stats) == 0:
        print("Skipping effect-precision scatter (insufficient multi-subject data)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by effect direction
    colors = ['green' if x > 0 else 'red' for x in valid_stats['mean_effect']]
    
    # Scatter plot
    scatter = ax.scatter(valid_stats['mean_effect'], valid_stats['se_effect'],
                        s=100, alpha=0.6, c=colors, edgecolors='black', linewidths=0.5)
    
    # Annotate anchor tasks
    anchors = ['Checkerboard', 'HCPMotor', 'PreferencePaintings', 'ContRing']
    for task in anchors:
        task_row = valid_stats[valid_stats['task'] == task]
        if len(task_row) > 0:
            x = task_row['mean_effect'].values[0]
            y = task_row['se_effect'].values[0]
            ax.annotate(task, (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Reference lines
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(xlim[1]*0.7, ylim[1]*0.9, 'Large &\nUncertain', 
           ha='center', va='top', fontsize=10, style='italic', alpha=0.5)
    ax.text(xlim[1]*0.7, ylim[0]*0.1, 'Large &\nPrecise', 
           ha='center', va='bottom', fontsize=10, style='italic', alpha=0.5)
    
    ax.set_xlabel('Mean Effect (β)', fontsize=12)
    ax.set_ylabel('Standard Error (SE)', fontsize=12)
    ax.set_title('Effect Size vs Precision Across Tasks', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'effect_precision_scatter.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization: effect_precision_scatter.png")

def create_network_context_radar(df, task_stats, output_dir):
    """Create radar plots showing network connectivity context for representative tasks."""
    import sys
    sys.path.insert(1, '/home/hmueller2/ibc_code/ibc_latent/Preprocessing/Aradia')
    try:
        import RR_utils as RR
    except ImportError:
        print("Cannot import RR_utils - skipping network context radar plot")
        return
    
    # Select one strong positive and one strong negative task
    pos_task = task_stats[task_stats['mean_effect'] > 0].nlargest(1, 'abs_mean_effect')
    neg_task = task_stats[task_stats['mean_effect'] < 0].nsmallest(1, 'mean_effect')
    
    if len(pos_task) == 0 or len(neg_task) == 0:
        print("Skipping network-context radar (insufficient tasks)")
        return
    
    pos_task_name = pos_task['task'].values[0]
    neg_task_name = neg_task['task'].values[0]
    
    # Define LSN indices based on spider_plots_infomap_kmeans.py network_names
    network_indices = {
        'DMN': [0, 1, 2, 3],        # 4 DMN subnetworks
        'DAN': [8, 9],              # 2 DAN subnetworks
        'Salience': [11],           # Salience network
        'SMN': [14, 15, 16],        # Somatomotor (Hand, Face, Foot)
        'Language': [10]            # Language network
    }
    
    networks = list(network_indices.keys())
    working_dir = '/ptmp/hmueller2/2025_ibc_latent/outputs'
    
    # Compute connectivity for each task
    connectivity_data = {}
    
    for task_name in [pos_task_name, neg_task_name]:
        task_data = df[df['task'] == task_name]
        
        # Initialize arrays for subnetwork A and B connectivity
        conn_A_list = []
        conn_B_list = []
        
        for idx, row in task_data.iterrows():
            subject = row['subject'].zfill(2) if isinstance(row['subject'], str) else f"{int(row['subject']):02d}"
            sub_str = f"sub-{subject}"
            
            try:
                # Load LSN ptseries (19 networks after removing FPN and Noise)
                parc_filename = os.path.join(
                    working_dir, 'individual_networks', sub_str, 'resting_state', 
                    f'{sub_str}_individual_nets_concat.ptseries.nii'
                )
                
                if not os.path.exists(parc_filename):
                    print(f"  Skipping {subject} - ptseries not found")
                    continue
                
                all_data_concat = RR.load_data(parc_filename)
                all_data_concat = np.delete(all_data_concat, [8, -1], axis=1)  # Remove FPN and Noise
                
                # Load vertex-level dtseries
                dtseries_path = os.path.join(
                    working_dir, 'individual_networks', sub_str, 'resting_state',
                    f'{sub_str}_all-tasks_concatenated_cleaned_fsLR_cortexOnly.dtseries.nii'
                )
                
                if not os.path.exists(dtseries_path):
                    print(f"  Skipping {subject} - dtseries not found")
                    continue
                
                dtseries_concat = RR.load_data(dtseries_path)
                
                # Load subnetwork masks (k=2 for A and B)
                kmeans_dir = os.path.join(working_dir, 'subnetworks', 'infomap', sub_str)
                kmeans_dlabel = os.path.join(kmeans_dir, f'{subject}_FPN_infomap_communities_kmeans.dlabel.nii')
                kmeans_dscalar = os.path.join(kmeans_dir, f'{subject}_FPN_infomap_communities_kmeans.dscalar.nii')
                
                filename = kmeans_dlabel if os.path.exists(kmeans_dlabel) else kmeans_dscalar
                
                if not os.path.exists(filename):
                    print(f"  Skipping {subject} - kmeans file not found")
                    continue
                
                subnetworks = RR.load_data(filename)
                
                # Get k=2 subnetworks (index 0 in the file, since k_values starts at 2)
                current_sns = subnetworks[0, :]  # k=2 is at index 0
                labels = RR.get_labels(filename, n_map=0)
                
                # Debug: print what labels look like
                print(f"  Subject {subject}: labels type={type(labels)}, labels={labels}")
                
                # Compute connectivity for subnetwork A (label 1) and B (label 2)
                conn_A_subject = np.zeros(len(networks))
                conn_B_subject = np.zeros(len(networks))
                
                # Find unique values in current_sns (should be 1 and 2 for k=2)
                unique_labels = np.unique(current_sns)
                print(f"  Subject {subject}: unique labels in k=2={unique_labels}")
                
                for subnetwork_id in [1, 2]:
                    # Create mask for this subnetwork
                    mask = (current_sns == subnetwork_id).astype(int)
                    
                    if np.sum(mask) == 0:
                        print(f"  Warning: Empty mask for {subject} subnetwork {subnetwork_id}")
                        continue
                    
                    # Get time series for this subnetwork
                    subnetwork_tseries = RR.get_network(dtseries_concat, mask, remove_rest=True)
                    
                    if subnetwork_tseries.shape[1] == 0:
                        print(f"  Warning: Empty subnetwork timeseries for {subject} subnetwork {subnetwork_id}")
                        continue
                    
                    average_tseries = np.mean(subnetwork_tseries, axis=1)
                    
                    # Compute correlation with all LSNs
                    corr_matrix = np.corrcoef(all_data_concat.T, average_tseries)
                    correlations_with_lsns = corr_matrix[-1, :-1]
                    
                    # Aggregate by network groups
                    for net_idx, (net_name, indices) in enumerate(network_indices.items()):
                        mean_corr = np.mean(correlations_with_lsns[indices])
                        
                        if subnetwork_id == 1:
                            conn_A_subject[net_idx] = mean_corr
                        else:
                            conn_B_subject[net_idx] = mean_corr
                
                conn_A_list.append(conn_A_subject)
                conn_B_list.append(conn_B_subject)
                
                print(f"  ✓ Computed connectivity for {subject}")
                
            except Exception as e:
                print(f"  Error computing connectivity for {subject}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Average across subjects
        if len(conn_A_list) > 0:
            conn_A = np.mean(conn_A_list, axis=0)
            conn_B = np.mean(conn_B_list, axis=0)
            connectivity_data[task_name] = {'A': conn_A, 'B': conn_B}
            print(f"✓ Computed connectivity for {task_name} (n={len(conn_A_list)} subjects)")
        else:
            print(f"✗ No connectivity data computed for {task_name}")
    
    # Only create radar plot if we have valid connectivity data for BOTH tasks
    if len(connectivity_data) < 2:
        print(f"✗ Skipping radar plot - insufficient connectivity data (got {len(connectivity_data)}/2 tasks)")
        return
    
    # Create radar plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(networks), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for idx, (task_name, ax) in enumerate([
        (pos_task_name, axes[0]),
        (neg_task_name, axes[1])
    ]):
        if task_name not in connectivity_data:
            continue
            
        conn_A = connectivity_data[task_name]['A']
        conn_B = connectivity_data[task_name]['B']
        
        # Close the plot
        conn_A_plot = np.concatenate([conn_A, [conn_A[0]]])
        conn_B_plot = np.concatenate([conn_B, [conn_B[0]]])
        
        ax.plot(angles, conn_A_plot, 'o-', linewidth=2, label='Subnetwork A', color='blue')
        ax.fill(angles, conn_A_plot, alpha=0.15, color='blue')
        ax.plot(angles, conn_B_plot, 'o-', linewidth=2, label='Subnetwork B', color='orange')
        ax.fill(angles, conn_B_plot, alpha=0.15, color='orange')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(networks, fontsize=10)
        ax.set_ylim(-0.5, 1)  # Adjust based on your correlation range
        ax.set_title(f'{task_name}\nNetwork Context', fontsize=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
    
    plt.suptitle('Network Connectivity Context for Representative Tasks', 
                 fontsize=14, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'network_context_radar.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved visualization: network_context_radar.png")

def compute_network_connectivity_from_raw(subject, task, subnetwork_id):
    """
    Helper function to compute network connectivity from raw data
    following the approach in spider_plots_infomap_kmeans.py
    
    This requires access to:
    - ptseries file with LSN time series
    - dtseries file with vertex-level data
    - subnetwork masks
    """
    import sys
    sys.path.insert(1, '/home/hmueller2/ibc_code/ibc_latent/Preprocessing/Aradia')
    import RR_utils as RR
    
    working_dir = '/ptmp/hmueller2/2025_ibc_latent/outputs'
    sub_str = f"sub-{subject}"
    
    # Load LSN ptseries (19 networks after removing FPN and Noise)
    parc_filename = os.path.join(
        working_dir, 'individual_networks', sub_str, 'resting_state', 
        f'{sub_str}_individual_nets_concat.ptseries.nii'
    )
    all_data_concat = RR.load_data(parc_filename)
    all_data_concat = np.delete(all_data_concat, [8, -1], axis=1)  # Remove FPN and Noise
    
    # Load vertex-level dtseries
    dtseries_path = os.path.join(
        working_dir, 'individual_networks', sub_str, 'resting_state',
        f'{sub_str}_all-tasks_concatenated_cleaned_fsLR_cortexOnly.dtseries.nii'
    )
    dtseries_concat = RR.load_data(dtseries_path)
    
    # Load subnetwork mask (adjust path based on your structure)
    kmeans_dir = os.path.join(working_dir, 'subnetworks', 'infomap', sub_str)
    # ... load mask for subnetwork_id ...
    
    # Compute average time series for subnetwork
    # subnetwork_tseries = RR.get_network(dtseries_concat, mask, remove_rest=True)
    # average_tseries = np.mean(subnetwork_tseries, axis=1)
    
    # Compute correlation with LSNs
    # corr_matrix = np.corrcoef(all_data_concat.T, average_tseries)
    # correlations_with_column_vector = corr_matrix[-1, :-1]
    
    # Return connectivity values for each network group
    # network_connectivity = {}
    # for net_name, indices in network_indices.items():
    #     network_connectivity[net_name] = np.mean(correlations_with_column_vector[indices])
    
    # return network_connectivity
    pass

def main():
    """Main analysis pipeline."""
    print(f"Found subjects: {subjects}")
    print(f"\nLoading FFX results from {PPI_BASE}...")
    df = load_subject_ffx_results(subjects)
    
    if df.empty:
        print("No data loaded. Check subject directories for task_level_ppi_ffx_summary.csv files.")
        return
    
    print(f"Loaded data for {df['subject'].nunique()} subjects and {df['task'].nunique()} tasks")
    
    print("\nPerforming statistical analysis...")
    task_stats = perform_task_analysis(df)
    
    if task_stats.empty:
        print("No valid tasks found for analysis.")
        return
    
    print("Ranking tasks...")
    task_stats = rank_tasks(task_stats)
    
    # Save task ranking
    task_stats.to_csv(os.path.join(OUTPUT_DIR, 'task_ranking.csv'), index=False)
    print(f"\nSaved task_ranking.csv with {len(task_stats)} tasks")
    
    print("\nCreating visualizations...")
    create_anchor_panels(df, task_stats, OUTPUT_DIR)
    create_effect_precision_scatter(task_stats, OUTPUT_DIR)
    create_network_context_radar(df, task_stats, OUTPUT_DIR)
    
    print("\nGenerating summary report...")
    generate_summary_report(task_stats, df, OUTPUT_DIR)
    
    print(f"\n✓ Analysis complete! Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
