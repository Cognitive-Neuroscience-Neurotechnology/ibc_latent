import os
import re
import csv
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

# ----- CLUSTERING METHOD -----
clustering_method = "kmeans"  # either "infomap" or "kmeans"

# ---- SET YOUR PATHS HERE ----
base_dir = "/ptmp/hmueller2/Downloads"  # parent of subnetworks/
out_dir = os.path.join(base_dir, "subnetworks", clustering_method)  # or set to any output folder you want

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
    pattern = os.path.join(base_dir, 'subnetworks', clustering_method, 'sub-*', f'sub-*_clustering_of_{clustering_method}_metrics.csv')
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

def compute_mean_min_max(values):
    arr = np.array(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.nan, np.nan, np.nan
    arr = arr[finite]
    return float(np.nanmean(arr)), float(np.nanmin(arr)), float(np.nanmax(arr))

def plot_metric(metric, data_by_k, out_dir):
    ks_sorted = sorted(data_by_k.keys())
    means, mins, maxs, xs = [], [], [], []
    for k in ks_sorted:
        mean_v, min_v, max_v = compute_mean_min_max(data_by_k[k])
        if np.isfinite(mean_v):
            xs.append(k)
            means.append(mean_v)
            mins.append(min_v)
            maxs.append(max_v)

    if not xs:
        print(f"[WARN] No finite data to plot for metric: {metric}")
        return

    xs = np.array(xs, dtype=int)
    means = np.array(means, dtype=float)
    mins = np.array(mins, dtype=float)
    maxs = np.array(maxs, dtype=float)

    plt.figure(figsize=(6, 4))
    plt.plot(xs, means, '-o', color='C0', label='Mean')
    plt.fill_between(xs, mins, maxs, color='C0', alpha=0.2, label='Range (min–max)')
    plt.xlabel('k')
    plt.ylabel(metric)
    plt.title(f'{metric} across subjects')
    plt.grid(True, alpha=0.3)
    plt.xticks(xs)
    plt.legend(loc='best', frameon=False)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'group_kmeans_on_{clustering_method}_{metric.replace(" ", "_").lower()}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

def main():
    files = collect_all_metrics(base_dir)
    if not files:
        print(f"No metrics CSV files found. Expecting them under subnetworks/{clustering_method}/sub-*/")
        return

    print(f"Found {len(files)} metrics files.")
    metrics_agg, subjects = aggregate(files)

    for metric in METRICS:
        plot_metric(metric, metrics_agg.get(metric, {}), out_dir)

if __name__ == '__main__':
    main()