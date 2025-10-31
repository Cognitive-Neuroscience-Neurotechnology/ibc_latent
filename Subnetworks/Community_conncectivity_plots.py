'''
This script analyzes the connectivity patterns of FPN communities derived from Infomap
and their relationship to k=2 clustering results obtained via k-means clustering. 
It generates radar plots comparing the connectivity profiles
of two FPN communities across standard brain networks and saves the results to CSV files.
'''

import os
import re
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib

# ---------------------------
# Paths expected:
# - {base}/individual_networks/sub-{sub}/resting_state/
#     Bipartite_PhysicalCommunities+AlgorithmicLabeling_FC_btwn_InfoMapCommunities.dtseries.nii
#     Bipartite_PhysicalCommunities+AlgorithmicLabeling_NetworkLabels.xls
# - {base}/subnetworks/infomap/sub-{sub}/
#     FPN_communities.dscalar.nii
#     {sub}_FPN_infomap_communities_kmeans.dlabel.nii  (contains k maps; we take k=2)
# ---------------------------

DMN_NAMES = {
    "DEFAULT_RETROSPLENIAL",
    "DEFAULT_DORSOLATERAL",
    "DEFAULT_ANTEROLATERAL",
    "DEFAULT_PARIETAL",
}
DAN_NAMES = {
    "DORSALATTENTION",
    "PREMOTOR/DORSALATTENTIONII",
    "PREMOTOR/DORSALATTENTIONLL",  # tolerate ll vs II
}
FPN_NAME = "FRONTOPARIETAL"

def norm_label(s):
    if s is None:
        return ""
    return str(s).strip().replace(" ", "").upper()

def group_network_name(raw):
    n = norm_label(raw)
    if n in DMN_NAMES or n.startswith("DEFAULT_"):
        return "DMN"
    if n in DAN_NAMES or n.startswith("DORSALATTENTION") or n.startswith("PREMOTOR/DORSALATTENTION"):
        return "DAN"
    if FPN_NAME in n or n == "FPN" or n.startswith("FRONTOPARIETAL"):
        return "FPN"
    return raw  # keep original for other networks

#   Load CIFTI file and return data and axes
def load_cifti(path):
    img = nib.load(path)
    data = img.get_fdata()
    ax0 = img.header.get_axis(0)  # time or scalar axis
    ax1 = img.header.get_axis(1)  # BrainModel or parcels/scalars axis
    return img, data, ax0, ax1

# Read community->network mapping from Excel or CSV
def read_network_labels_xls(xls_path):
# Prefer pandas if available; otherwise simple csv fallback (user can export .csv from Excel)
    if not os.path.exists(xls_path):
        csv_guess = os.path.splitext(xls_path)[0] + ".csv"
        if os.path.exists(csv_guess):
            return read_network_labels_csv(csv_guess)
        raise FileNotFoundError(f"Mapping file not found: {xls_path}")
    try:
        import pandas as pd
        df = pd.read_excel(xls_path)
# Expect columns like "Community", "Network"
        col_comm = [c for c in df.columns if "comm" in c.lower()][0]
        col_net = [c for c in df.columns if "net" in c.lower()][0]
        mapping = {}
        for _, row in df.iterrows():
            try:
                cid = int(row[col_comm])
            except Exception:
                continue
            mapping[cid] = str(row[col_net])
        return mapping
    except Exception as e:
        raise RuntimeError(f"Failed reading {xls_path}. If pandas/xlrd missing, export as CSV. Error: {e}")

# Fallback CSV reader
def read_network_labels_csv(csv_path):
    mapping = {}
    with open(csv_path, "r", newline="") as f:
        import csv as _csv
        reader = _csv.DictReader(f, delimiter=",")
# handle generic headers
        cols = [c.lower() for c in reader.fieldnames]
        i_comm = cols.index("community") if "community" in cols else 0
        i_net = cols.index("network") if "network" in cols else 1
        for row in reader:
            try:
                cid = int(list(row.values())[i_comm])
            except Exception:
                continue
            net = list(row.values())[i_net]
            mapping[cid] = str(net)
# Find index of k=2 map in dlabel
    return mapping

# Find index of k=2 map in dlabel
def find_k2_index(ax0):
# If map names exist (e.g., "k=2"), pick that, else fallback to index 1
    names = []
    try:
        names = [ax0.name(i) for i in range(len(ax0))]
    except Exception:
        names = []
    if names:
        for i, n in enumerate(names):
            if n and "k=2" in str(n):
                return i
    return 1 if len(ax0) > 1 else 0

# Determine majority k=2 cluster for a given community
def majority_cluster_for_community(k2_vertex_labels, community_vertex_labels, comm_id):
    mask = (community_vertex_labels == comm_id)
    if not np.any(mask):
        return 0
    vals, counts = np.unique(k2_vertex_labels[mask], return_counts=True)
# ignore background 0
    if 0 in vals:
        m = vals != 0
        vals = vals[m]
        counts = counts[m]
    if len(vals) == 0:
        return 0
    return int(vals[np.argmax(counts)])  # 1 or 2

def main_for_subject(subject, base_dir):
    indiv_dir = os.path.join(base_dir, "individual_networks", f"sub-{subject}", "resting_state")
    sub_dir = os.path.join(base_dir, "subnetworks", "infomap", f"sub-{subject}")
    os.makedirs(sub_dir, exist_ok=True)

    ts_path = os.path.join(indiv_dir, "Bipartite_PhysicalCommunities+AlgorithmicLabeling_FC_btwn_InfoMapCommunities.dtseries.nii")
    xls_path = os.path.join(indiv_dir, "Bipartite_PhysicalCommunities+AlgorithmicLabeling_NetworkLabels.xls")
    fpn_comm_path = os.path.join(sub_dir, "FPN_communities.dscalar.nii")
    kmeans_dlabel_path = os.path.join(sub_dir, f"{subject}_FPN_infomap_communities_kmeans.dlabel.nii")

    print(f"Loading community time series: {ts_path}")
    ts_img, ts_data, ts_ax0, ts_ax1 = load_cifti(ts_path)  # (T, N_comm)
    T, N_comm = ts_data.shape
    print(f"TS shape: {ts_data.shape}")

    print(f"Loading community->network mapping: {xls_path}")
    comm2net_raw = read_network_labels_xls(xls_path)
    all_cids = sorted(comm2net_raw.keys())
    if not all_cids:
        raise RuntimeError("No community->network assignments read from the labels file.")

# Build network groups
    comm2net_group = {cid: group_network_name(name) for cid,# Load vertex-level FPN communities and k=2 cluster labels
name in comm2net_raw.items()}

# Load vertex-level FPN communities and k=2 cluster labels
    print(f"Loading FPN communities (vertex-level): {fpn_comm_path}")
    fpn_img = nib.load(fpn_comm_path)
    fpn_comm = np.squeeze(fpn_img.get_fdata()).astype(int)  # (grayordinates,)

    print(f"Loading kmeans clusters (dlabel): {kmeans_dlabel_path}")
    km_img = nib.load(kmeans_dlabel_path)
    km_data = km_img.get_fdata()
    km_ax0 = km_img.header.get_axis(0)
    k2_idx = find_k2_index(km_ax0)
    k2_labels = km_data[k2_idx].astype(int)

# Determine which community columns map to which community ids
    # We assume column index i corresponds to community id (i+1)
    # If parcel names include IDs, we try to parse them; else fall back to i+1.
    col_cid = np.arange(1, N_comm + 1, dtype=int)
    try:
        names = [ts_ax1.name(i) for i in range(len(ts_ax1))]
        parsed = []
        for nm in names:
            m = re.search(r'(\d+)\s*$', str(nm))
            parsed.append(int(m.group(1)) if m else None)
        if all(p is not None for p in parsed):
            col_cid = np.array(parsed, dtype=int)
            print("Parsed community IDs from dtseries column names.")
    except Exception:
        pass

# Get list of FPN community ids
    fpn_cids = sorted([cid for cid, net in comm2net_group.items() if group_network_name(net) == "FPN"])
    if not fpn_cids:
        raise RuntimeError("No FPN communities found in the mapping file.")

# Map each FPN community id to cluster (1 or 2) via vertex majority vote
    comm2cluster = {}
    for cid in fpn_cids:
        cl = majority_cluster_for_community(k2_labels, fpn_comm, cid)
        if cl in (1, 2):
            comm2cluster[cid] = cl
    if not comm2cluster:
        raise RuntimeError("No FPN communities received a valid cluster label from k=2 map.")

    # Precompute rank-transformed matrix for Spearman
    ranks = ts_data.argsort(axis=0).argsort(axis=0).astype(float)
    ranks -= ranks.mean(axis=0, keepdims=True)
    ranks_std = np.linalg.norm(ranks, axis=0, keepdims=True)
    ranks_std[ranks_std == 0] = 1.0
    R = ranks / ranks_std

    cid2col = {}
    for j, cid in enumerate(col_cid):
        cid2col[int(cid)] = j

# For each FPN community, compute correlation vector to all communities; then average per network grou# We will exclude FPN from the output "other networks"
    networks_all = sorted({group_network_name(n) for n in comm2net_raw.values()})
# We will exclude FPN from the output "other network# Per-community DMN/DAN means for scatter
    scatter_points = []  # list of (cid, cluster, r_DMN, r_DAN)

    networks_all = sorted({group_network_name(n) for n in comm2net_raw.values()})
    networks_other = [n for n in networks_all if group_network_name(n) != "FPN"]

    scatter_points = []  # list of (cid, cluster, r_DMN, r_DAN)

    # Accumulate per-network lists for cluster A/B
    net2vals_A = {net: [] for net in networks_other}
    net2vals_B = {net: [] for net in networks_other}
    for cid, cl in comm2cluster.items():
        col = cid2col.get(cid, None)
        if col is None or col < 0 or col >= N_comm:
            print(f"[warn] No column found for community id {cid}; skipping.")
            continue
        v = R[:, col]
        r_all = v @ R / R.shape[0]

# aggregate per network
        per_net = {}
        for net in networks_other:
# gather columns whose community id maps to this network
            cols = [cid2col[c2] for c2, n2 in comm2net_group.items()
                    if group_network_name(n2) == net and c2 in cid2col]
            if not cols:
# push to A/B pools
                per_net[net] = np.nan
            else:
                per_net[net] = float(np.nanmean(r_all[cols]))

# push to A/B pools
        for net, val in per_net.items():
            if np.isnan(val):
                continue
            if cl == 1:  # DMN/DAN for scatter
                net2vals_A[net].append(val)
            elif cl == 2:
                net2vals_B[net].append(val)

# DMN/DAN for scatter
        r_dmn = per_net.get("DMN", np.nan)
        r_dan = per_net.get("DAN", np.nan)
        scatter_points.append((cid, cl, r_dmn, r_dan))

# Compute per-network means for CSV
    rows = []
    for net in networks_other:
        valsA = np.array(net2vals_A.get(net, []), dtype=float)
        valsB = np.array(net2vals_B.get(net, []), dtype=float)
        meanA = float(np.nanmean(valsA)) if valsA.size else np.nan
        meanB = float(np.nanmean(valsB)) if valsB.size else np.nan
# Save CSV
        diff = meanB - meanA if (np.isfinite(meanA) and np.isfinite(meanB)) else np.nan
        rows.append([net, meanA, meanB, diff])

# Save CSV
    csv_out = os.path.join(sub_dir, f"sub-{subject}_FPN_k2_connectivity_by_network.csv")
    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Network", "mean_r_FPN_A", "mean_r_FPN_B", "diff_B_minus_A"])
        for r in rows:
            writer.writerow(r)
    print(f"Saved CSV: {csv_out}")

# Scatter plot: each FPN community, x=DMN, y=DAN, color by cluster
    plt.figure(figsize=(6, 5))
    for cid, cl, x, y in scatter_points:
        color = "C0" if cl == 1 else "C1"
        plt.scatter(x, y, c=color, s=40, alpha=0.8, edgecolor="k", linewidths=0.3)
        plt.text(x, y, str(cid), fontsize=7, ha="left", va="bottom")
    plt.axvline(0, color="k", lw=0.5, alpha=0.3)
    plt.axhline(0, color="k", lw=0.5, alpha=0.3)
    plt.xlabel("Connectivity to DMN (mean r across DMN communities)")
    plt.ylabel("Connectivity to DAN (mean r across DAN communities)")
    plt.title(f"sub-{subject}: FPN communities (color = cluster A/B)")
    plt.grid(True, alpha=0.2)
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', label='FPN_A', markerfacecolor='C0', markersize=8, markeredgecolor='k', markeredgewidth=0.3),
        plt.Line2D([0], [0], marker='o', color='w', label='FPN_B', markerfacecolor='C1', markersize=8, markeredgecolor='k', markeredgewidth=0.3),
    ], loc="best", frameon=False)
    scatter_out = os.path.join(sub_dir, f"sub-{subject}_FPN_k2_scatter_DMN_vs_DAN.png")
    plt.tight_layout()
    plt.savefig(scatter_out, dpi=150)
    plt.close()
    print(f"Saved scatter: {scatter_out}")

# Radar (spider) plot comparing cluster means across all networks (excluding FPN)
    categories = [n for n in networks_other]
    theta = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    theta = np.concatenate([theta, theta[:1]])
    def pad(vals):
        return np.concatenate([np.asarray(vals, dtype=float), [vals[0] if len(vals) else np.nan]])

    meansA = [np.nan] * len(categories)
    meansB = [np.nan] * len(categories)
    for i, net in enumerate(categories):
        vA = net2vals_A.get(net, [])
        vB = net2vals_B.get(net, [])
        meansA[i] = float(np.nanmean(vA)) if len(vA) else np.nan
        meansB[i] = float(np.nanmean(vB)) if len(vB) else np.nan

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(7, 6))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(theta[:-1]), categories)
    A = pad(meansA)
    B = pad(meansB)
    ax.plot(theta, A, color="C0", lw=2, label="FPN_A")
    ax.fill(theta, A, color="C0", alpha=0.15)
    ax.plot(theta, B, color="C1", lw=2, label="FPN_B")
    ax.fill(theta, B, color="C1", alpha=0.15)
    ax.set_title(f"sub-{subject}: mean connectivity of FPN_A vs FPN_B")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1), frameon=False)
    plt.tight_layout()
    radar_out = os.path.join(sub_dir, f"sub-{subject}_FPN_k2_radar_cluster_means.png")
    plt.savefig(radar_out, dpi=150)
    plt.close(fig)
    print(f"Saved radar: {radar_out}")

if __name__ == "__main__":
    base_dir = "/ptmp/hmueller2/Downloads"
    subjects_file = os.path.join(base_dir, "subjects_resting.txt")
    with open(subjects_file, "r") as f:
        subjects = [line.strip() for line in f if line.strip()]
    for subject in subjects:
        print(f"\n=== Processing subject {subject} ===")
        try:
            main_for_subject(subject.zfill(2), base_dir)
        except Exception as e:
            print(f"Error processing subject {subject}: {e}")
