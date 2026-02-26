'''
Script to analyze PPI subnetwork distinction across subjects and tasks.
Generates statistical summaries, rankings, and visualizations.
Output files: task_ranking_overview.png, task_distributions_top_bottom.png, task_scatter_individual_subjects.png, analysis_summary.txt
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
        x_vals = np.random.normal(0, 0.04, size=len(y_vals))
        
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
    create_visualizations(task_stats, df, OUTPUT_DIR)
    create_violin_plots(df, task_stats, OUTPUT_DIR)
    create_scatter_plot(df, task_stats, OUTPUT_DIR)
    
    print("\nGenerating summary report...")
    generate_summary_report(task_stats, df, OUTPUT_DIR)
    
    print(f"\n✓ Analysis complete! Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
