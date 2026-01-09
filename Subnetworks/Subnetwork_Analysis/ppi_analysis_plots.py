"""
Compare task-level PPI across subjects and visualize which tasks show strongest connectivity changes.
Output files: ppi_by_task.png, ppi_tstat_by_task.png, effect_size_vs_se.png
OLD!! - Now using _DMN_DAN
"""
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

subject = sys.argv[1]
base_dir = '/ptmp/hmueller2/Downloads/ppi_results'
subject_dir = os.path.join(base_dir, f'sub-{subject}')

# Check if results exist
ffx_csv = os.path.join(subject_dir, 'task_level_ppi_ffx_summary.csv')
per_run_csv = os.path.join(subject_dir, 'task_level_ppi_results.csv')

if not os.path.exists(ffx_csv):
    if os.path.exists(per_run_csv):
        print(f"Warning: FFX summary not found for sub-{subject}. Using per-run results.")
        df = pd.read_csv(per_run_csv)
    else:
        print(f"ERROR: No results found for sub-{subject}")
        sys.exit(1)
else:
    df = pd.read_csv(ffx_csv)

if df.empty:
    print(f"Warning: Empty results for sub-{subject}. Skipping plots.")
    sys.exit(0)

# Create output directory
plot_dir = os.path.join(subject_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

# Plot 1: Beta values by task
plt.figure(figsize=(12, 6))
if 'beta_ffx' in df.columns:
    sns.barplot(data=df, x='task', y='beta_ffx', hue='task', palette='coolwarm')
    plt.ylabel('Beta (FFX)')
else:
    sns.barplot(data=df, x='task', y='beta', hue='task', palette='coolwarm')
    plt.ylabel('Beta')
plt.title(f'PPI Effects by Task - sub-{subject}')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'ppi_by_task.png'), dpi=300)
plt.close()

# Plot 2: T-statistics
plt.figure(figsize=(12, 6))
if 't_ffx' in df.columns:
    sns.barplot(data=df, x='task', y='t_ffx', hue='task', palette='RdBu_r')
    plt.ylabel('t-statistic (FFX)')
else:
    sns.barplot(data=df, x='task', y='tstat', hue='task', palette='RdBu_r')
    plt.ylabel('t-statistic')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.title(f'PPI t-statistics by Task - sub-{subject}')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'ppi_tstat_by_task.png'), dpi=300)
plt.close()

# Plot 3: Effect size vs Standard Error (if available)
if 'se_ffx' in df.columns and 'beta_ffx' in df.columns:
    plt.figure(figsize=(10, 8))
    plt.scatter(df['se_ffx'], df['beta_ffx'], s=100, alpha=0.6)
    for idx, row in df.iterrows():
        plt.annotate(row['task'], (row['se_ffx'], row['beta_ffx']), fontsize=8, alpha=0.7)
    plt.xlabel('Standard Error')
    plt.ylabel('Beta (FFX)')
    plt.title(f'Effect Size vs Precision - sub-{subject}')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'effect_size_vs_se.png'), dpi=300)
    plt.close()

print(f"✓ Plots saved to: {plot_dir}")