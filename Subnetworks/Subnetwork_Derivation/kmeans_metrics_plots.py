'''
This script collects k-means clustering metrics from multiple subjects,
plots them, and saves the plots to an output directory. It assumes that
the metrics CSV files are stored in a specific directory structure.
'''

import os
import re
import csv
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----- CLUSTERING METHOD -----
clustering_method = "kmeans"  # either "infomap" or "kmeans"

# ---- SET YOUR PATHS HERE ----
base_dir = "/ptmp/hmueller2/2025_ibc_latent/outputs"  # parent of subnetworks/
out_dir = os.path.join(base_dir, "subnetworks", clustering_method)

METRICS = [
    'Entropy',
    'BIC',
    'Smallest Cluster Size',
    'Inertia',
    'Silhouette Score', 
]

def parse_metrics_csv(csv_path):
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return {}, []

    # Header: ['metric', 'k=1', 'k=2', ...]
    header = rows[0]
    k_labels = header[1:]
    ks = []
    for lab in k_labels:
        m = re.search(r'k=(\d+)', lab)
        if m:
            ks.append(int(m.group(1)))
        else:
            ks.append(None)

    metrics = {}
    for row in rows[1:]:
        if not row:
            continue
        metric_name = row[0]
        vals = []
        for v in row[1:]:
            if v is None or v == '':
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(v))
                except ValueError:
                    # handle strings like 'inf', '-inf', 'nan'
                    try:
                        vals.append(float(v.lower()))
                    except Exception:
                        vals.append(np.nan)
        # Map k -> value
        metrics[metric_name] = {k: val for k, val in zip(ks, vals) if k is not None}
    return metrics, sorted([k for k in ks if k is not None])

def collect_all_metrics(base_dir):
    # Look under: base_dir/subnetworks/<clustering_method>/sub-*/<csv>
    pattern = os.path.join(out_dir, 'sub-*', f'sub-*_clustering_of_{clustering_method}_metrics.csv')
    files = sorted(glob(pattern))
    return files

def aggregate(files):
    # metrics_agg[metric][k] = list of values across subjects
    metrics_agg = {m: {} for m in METRICS}
    subjects = []
    for fp in files:
        m, ks = parse_metrics_csv(fp)
        subjects.append(os.path.basename(os.path.dirname(fp)))
        for metric in METRICS:
            if metric not in m:
                continue
            for k, val in m[metric].items():
                metrics_agg[metric].setdefault(k, []).append(val)
    return metrics_agg, subjects

def compute_mean_min_max_std(values):
    arr = np.array(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.nan, np.nan, np.nan, np.nan
    arr = arr[finite]
    return float(np.nanmean(arr)), float(np.nanstd(arr)), float(np.nanmin(arr)), float(np.nanmax(arr))

def save_aggregated_csv(metrics_agg, out_dir):
    """Save aggregated metrics (mean, std, min, max) to CSV files"""
    
    # Get all k values
    all_ks = set()
    for metric_data in metrics_agg.values():
        all_ks.update(metric_data.keys())
    ks_sorted = sorted(all_ks)
    
    # Create summary dataframe for all metrics
    summary_data = []
    for metric in METRICS:
        data_by_k = metrics_agg.get(metric, {})
        for k in ks_sorted:
            if k in data_by_k:
                mean_v, std_v, min_v, max_v = compute_mean_min_max_std(data_by_k[k])
                summary_data.append({
                    'Metric': metric,
                    'k': k,
                    'Mean': mean_v,
                    'Std': std_v,
                    'Min': min_v,
                    'Max': max_v,
                    'N_subjects': len([v for v in data_by_k[k] if np.isfinite(v)])
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save wide-format summary (one row per metric, columns for k values)
    wide_data = []
    for metric in METRICS:
        row = {'Metric': metric}
        data_by_k = metrics_agg.get(metric, {})
        for k in ks_sorted:
            if k in data_by_k:
                mean_v, std_v, _, _ = compute_mean_min_max_std(data_by_k[k])
                row[f'k={k}_mean'] = mean_v
                row[f'k={k}_std'] = std_v
        wide_data.append(row)
    
    wide_df = pd.DataFrame(wide_data)
    
    # Save both formats
    os.makedirs(out_dir, exist_ok=True)
    
    summary_path = os.path.join(out_dir, f'group_{clustering_method}_metrics_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary (long format): {summary_path}")
    
    wide_path = os.path.join(out_dir, f'group_{clustering_method}_metrics_wide.csv')
    wide_df.to_csv(wide_path, index=False)
    print(f"Saved summary (wide format): {wide_path}")
    
    return summary_df, wide_df

def plot_metric(metric, data_by_k, out_dir):
    ks_sorted = sorted(data_by_k.keys())
    
    # Filter to only include k=2 to k=10
    ks_sorted = [k for k in ks_sorted if 2 <= k <= 10]
    means, stds, mins, maxs, xs = [], [], [], [], []
    for k in ks_sorted:
        mean_v, std_v, min_v, max_v = compute_mean_min_max_std(data_by_k[k])
        if np.isfinite(mean_v):
            xs.append(k)
            means.append(mean_v)
            stds.append(std_v)
            mins.append(min_v)
            maxs.append(max_v)

    if not xs:
        print(f"[WARN] No finite data to plot for metric: {metric}")
        return

    xs = np.array(xs, dtype=int)
    means = np.array(means, dtype=float)
    stds = np.array(stds, dtype=float)
    mins = np.array(mins, dtype=float)
    maxs = np.array(maxs, dtype=float)

    # Choose color based on clustering method
    plot_color = 'firebrick' if clustering_method == "infomap" else 'indigo' # else 'C0'

    plt.figure(figsize=(8, 5))
    
    # Plot mean with error bars (std)
    plt.errorbar(xs, means, yerr=stds, fmt='-o', color=plot_color, 
                 capsize=5, capthick=2, label='Mean ± SD')
    
    # Add min/max range as shaded area
    plt.fill_between(xs, mins, maxs, color=plot_color, alpha=0.15, label='Range (min–max)')
    
    plt.xlabel('Number of clusters (k)', fontsize=11)
    plt.ylabel(metric, fontsize=11)
    plt.title(f'{metric} across subjects ({clustering_method})', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(xs)
    plt.legend(loc='best', frameon=True)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'group_kmeans_on_{clustering_method}_{metric.replace(" ", "_").lower()}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    plt.close()
    print(f"Saved plot: {out_path}")

def main():
    files = collect_all_metrics(base_dir)
    if not files:
        print(f"No metrics CSV files found. Expecting them under subnetworks/{clustering_method}/sub-*/")
        return

    print(f"Found {len(files)} metrics files.")
    metrics_agg, subjects = aggregate(files)
    
    print(f"\nSubjects included: {', '.join(subjects)}")
    
    # Save aggregated metrics to CSV
    print("\n=== Saving aggregated metrics ===")
    summary_df, wide_df = save_aggregated_csv(metrics_agg, out_dir)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for metric in METRICS:
        print(f"\n{metric}:")
        metric_data = summary_df[summary_df['Metric'] == metric]
        if not metric_data.empty:
            print(metric_data[['k', 'Mean', 'Std', 'Min', 'Max']].to_string(index=False))
    
    # Generate plots
    print("\n=== Generating plots ===")
    for metric in METRICS:
        plot_metric(metric, metrics_agg.get(metric, {}), out_dir)
    
    print("\n=== Complete ===")

if __name__ == '__main__':
    main()