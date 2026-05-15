"""Compute resting-state functional connectivity for Glasser 360 or split network parcels.

Steps:
1) Load resting-state fMRI timeseries for the subject (multiple runs concatenated).
2) Extract average timecourse for each parcel (either Glasser 360 or split network parcels).
3) Compute Pearson correlation matrix between all parcels (FC matrix).
4) Compute hub metrics: participation coefficient and strength (mean positive FC).
5) For split parcels: create spring-embedded network plot with FC-based edges (threshold-filtered).
6) Save FC matrix, metrics, and visualizations to subject-specific directories.

Outputs:
- subject_fc_360x360.npz or subject_fc_network_parcels.npz: FC correlation matrix + metadata
- subject_fc_network_collapsed.npz (optional): network-by-network FC collapsed from parcel FC
- static_hub_metrics_subject.csv: PC, strength, network assignment per parcel
- spring_plot_<analysis>_edgesXXX_nodeYYY_<metric>.png: Spring-embedded visualization (split parcels mode only)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

HUBNESS_DIR = Path(__file__).resolve().parents[1]
if str(HUBNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HUBNESS_DIR))

from hubness_utils import (
    compute_participation_coefficient,
    discover_subjects_from_subdirs,
    ensure_dir,
    extract_network_colors_from_dlabel,
    find_rest_runs,
    load_glasser_parcellation,
    load_split_parcel_manifest_retained,
    load_split_parcel_masks,
    map_to_cortex,
    subject_network_label_path,
    zscore_columns,
)

DEFAULT_FMRIPREP_BASE = "/ptmp/hmueller2/2025_ibc_latent/outputs/preprocessing/fmriprep_out"
DEFAULT_ASSIGNMENT_DIR = "/ptmp/hmueller2/2025_ibc_latent/outputs/hubness"
DEFAULT_OUTPUT_DIR = "/ptmp/hmueller2/2025_ibc_latent/outputs/hubness"
DEFAULT_NETWORK_LABEL_BASE = "/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks/derived_networks"
DEFAULT_OVERLAP_THRESHOLD = 0.30
ANALYSIS_LEVEL_NETWORK = "network"
ANALYSIS_LEVEL_NETWORK_PARCEL = "network_parcel"
FPN_MODE_UNIFIED = "unified"
FPN_MODE_SPLIT = "split"


def load_assignment_for_subject(subject: str, assignment_dir: str) -> pd.DataFrame:
    subject_file = Path(assignment_dir) / f"sub-{subject}" / "parcel_network_assignment_subject.csv"
    if subject_file.exists():
        return pd.read_csv(subject_file)

    raise FileNotFoundError("No assignment table found. Run define_networks.py first.")


def parcellate_run(func_path: Path, cortical_indices: np.ndarray, parcel_masks: Dict[int, np.ndarray]) -> np.ndarray:
    img = nib.load(str(func_path))
    data = img.get_fdata()

    if data.ndim != 2:
        raise ValueError(f"Unexpected dtseries shape for {func_path}: {data.shape}")

    if data.shape[1] == len(cortical_indices):
        cortex_data = data
    else:
        max_idx = int(cortical_indices.max())
        if data.shape[1] <= max_idx:
            raise ValueError(f"Could not map data shape={data.shape} to cortical indices")
        cortex_data = data[:, cortical_indices]

    ts = np.zeros((cortex_data.shape[0], len(parcel_masks)), dtype=float)
    parcel_ids = sorted(parcel_masks.keys())
    for i, parcel_id in enumerate(parcel_ids):
        mask = parcel_masks[parcel_id]
        if not np.any(mask):
            ts[:, i] = np.nan
            continue
        ts[:, i] = np.nanmean(cortex_data[:, mask], axis=1)

    return zscore_columns(ts)


def compute_subject_fc(subject: str, fmriprep_base: str, parcellation_path: Optional[str]) -> Tuple[np.ndarray, np.ndarray, List[Path]]:
    parcellation_cortex, unique_parcels, _, cortical_indices, _ = load_glasser_parcellation(parcellation_path)
    parcel_masks = {int(pid): (parcellation_cortex == int(pid)) for pid in unique_parcels}

    runs = find_rest_runs(subject, fmriprep_base)
    if not runs:
        raise FileNotFoundError(f"No resting-state runs found for sub-{subject}")

    run_ts = []
    for run in runs:
        run_ts.append(parcellate_run(run, cortical_indices, parcel_masks))

    all_ts = np.concatenate(run_ts, axis=0)
    corr = np.corrcoef(all_ts, rowvar=False)
    corr = np.asarray(corr, dtype=float)
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 0.0)
    return corr, unique_parcels, runs


def compute_hub_metrics(fc: np.ndarray, modules: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pc = compute_participation_coefficient(fc, modules)

    fc_pos = np.array(fc, dtype=float)
    fc_pos[fc_pos < 0] = 0.0
    np.fill_diagonal(fc_pos, 0.0)
    strength = np.sum(fc_pos, axis=1)

    return pc, strength


def compute_abs_strength(fc: np.ndarray) -> np.ndarray:
    fc_abs = np.abs(np.asarray(fc, dtype=float))
    np.fill_diagonal(fc_abs, 0.0)
    return np.sum(fc_abs, axis=1)


def normalize_to_percentile_rank(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return arr
    if n == 1:
        return np.array([1.0], dtype=float)

    order = np.argsort(arr)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    return ranks / float(n - 1)


def color_from_name(name: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    # Stable deterministic fallback color from network name.
    hue = (sum(ord(ch) for ch in str(name)) % 360) / 360.0
    rgb = tuple(float(c) for c in plt.cm.hsv(hue)[:3])
    return rgb[0], rgb[1], rgb[2], float(alpha)


def resolve_split_mask_path(subject: str, assignment_dir: str, split_mask_path: str) -> Path:
    path_text = str(split_mask_path).strip()
    if not path_text:
        raise ValueError("Missing split_mask_path for a retained parcel row")

    candidates = [
        Path(path_text),
        Path(assignment_dir) / path_text,
        Path(assignment_dir) / f"sub-{subject}" / path_text,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not resolve split mask path: {split_mask_path}")


def build_spring_plot_filename(
    analysis_level: str,
    edge_threshold_percentile: int,
    top_hubs_k: int,
    hub_selection_metric: str,
) -> str:
    edge_pct = int(np.clip(edge_threshold_percentile, 0, 100))
    # File naming uses kept-edge percentage for readability: edges005 means top 5% retained globally.
    kept_edge_pct = 100 - edge_pct
    node_k = max(1, int(top_hubs_k))
    metric = str(hub_selection_metric)
    return (
        f"spring_plot_{analysis_level}_"
        f"edges{kept_edge_pct:03d}_"
        f"node{node_k:03d}_"
        f"{metric}.png"
    )


def build_circular_plot_filename(
    analysis_level: str,
    edge_threshold_percentile: int,
    hub_selection_metric: str,
) -> str:
    edge_pct = int(np.clip(edge_threshold_percentile, 0, 100))
    kept_edge_pct = 100 - edge_pct
    metric = str(hub_selection_metric)
    return (
        f"circular_plot_{analysis_level}_"
        f"edges{kept_edge_pct:03d}_"
        f"{metric}.png"
    )


def write_top_hubs_split_dlabel(
    subject: str,
    manifest: pd.DataFrame,
    hub_indices_desc: np.ndarray,
    output_path: Path,
    assignment_dir: str,
    network_label_base: str,
    parcellation_path: Optional[str],
    infomap_color_map: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
) -> None:
    network_path = subject_network_label_path(network_label_base, subject)
    if network_path.exists():
        template_img = nib.load(str(network_path))
    else:
        _, _, _, _, template_img = load_glasser_parcellation(parcellation_path)

    out_data = np.zeros(template_img.shape, dtype=np.int32)
    label_table: Dict[int, Tuple[str, Tuple[float, float, float, float]]] = {
        0: ("background", (0.0, 0.0, 0.0, 0.0))
    }

    if "network_name" not in manifest.columns:
        raise ValueError("manifest missing 'network_name' column")

    for label_id, parcel_idx in enumerate(hub_indices_desc, start=1):
        row = manifest.iloc[int(parcel_idx)]
        network_name = str(row.get("network_name", "unknown"))
        parcel_name = str(row.get("parcel_name", f"parcel_{int(parcel_idx) + 1}"))
        mask_path = resolve_split_mask_path(subject, assignment_dir, str(row.get("split_mask_path", "")))

        mask_img = nib.load(str(mask_path))
        mask_data = mask_img.get_fdata()
        if mask_data.ndim == 1:
            split_mask = mask_data > 0
        elif mask_data.ndim == 2:
            split_mask = (mask_data > 0).any(axis=0)
        else:
            raise ValueError(f"Unexpected split mask shape for {mask_path}: {mask_data.shape}")

        if split_mask.shape[0] != out_data.shape[1]:
            raise ValueError(
                f"Split mask grayordinate length mismatch for {mask_path}: "
                f"mask={split_mask.shape[0]}, template={out_data.shape[1]}"
            )

        rgba = color_from_name(network_name, alpha=1.0)
        if infomap_color_map and network_name in infomap_color_map:
            rgba = infomap_color_map[network_name]

        label_name = f"hub_{label_id:02d}_{network_name}_{parcel_name}"
        label_table[label_id] = (label_name, rgba)
        out_data[0, split_mask] = label_id

    brain_axis = template_img.header.get_axis(1)
    label_axis = nib.cifti2.LabelAxis(name=np.array(["top_hubs"]), label=label_table)
    header = nib.cifti2.Cifti2Header.from_axes((label_axis, brain_axis))
    out_img = nib.Cifti2Image(out_data.astype(np.int32), header=header)
    nib.save(out_img, str(output_path))


def collapse_parcel_fc_to_network_fc(fc: np.ndarray, network_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse parcel-level FC into network-level FC by averaging block connections.

    Within-network values are computed from off-diagonal parcel pairs only.
    Between-network values are computed from all cross-network parcel pairs.
    """
    labels = np.asarray(network_labels).astype(str)
    network_names = np.array(list(dict.fromkeys(labels.tolist())))
    n_networks = len(network_names)

    network_fc = np.zeros((n_networks, n_networks), dtype=float)
    pair_counts = np.zeros((n_networks, n_networks), dtype=int)
    network_indices = [np.where(labels == net)[0] for net in network_names]

    for i in range(n_networks):
        idx_i = network_indices[i]
        for j in range(i, n_networks):
            idx_j = network_indices[j]

            if i == j:
                if len(idx_i) < 2:
                    value = 0.0
                    count = 0
                else:
                    block = fc[np.ix_(idx_i, idx_i)]
                    tri = np.triu_indices(len(idx_i), k=1)
                    values = block[tri]
                    values = values[np.isfinite(values)]
                    count = int(values.size)
                    value = float(np.mean(values)) if count > 0 else 0.0
            else:
                block = fc[np.ix_(idx_i, idx_j)]
                values = block[np.isfinite(block)]
                count = int(values.size)
                value = float(np.mean(values)) if count > 0 else 0.0

            network_fc[i, j] = value
            network_fc[j, i] = value
            pair_counts[i, j] = count
            pair_counts[j, i] = count

    np.fill_diagonal(network_fc, 0.0)
    return network_fc, network_names, pair_counts


def compute_subject_fc_network_parcels(
    subject: str, fmriprep_base: str, assignment_dir: str, overlap_threshold: float = 0.30
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[Path]]:
    """Compute FC matrix for split network parcels."""
    manifest = load_split_parcel_manifest_retained(subject, assignment_dir, overlap_threshold)
    
    if manifest.empty:
        raise ValueError(f"No retained split parcels found for sub-{subject}")
    
    _, _, _, cortical_indices, _ = load_glasser_parcellation(None)
    parcel_masks = load_split_parcel_masks(subject, assignment_dir, overlap_threshold, cortical_indices)
    
    # Sort manifest by split_mask_path to match the order of loaded masks
    manifest = manifest.sort_values('split_mask_path').reset_index(drop=True)
    
    runs = find_rest_runs(subject, fmriprep_base)
    if not runs:
        raise FileNotFoundError(f"No resting-state runs found for sub-{subject}")
    
    run_ts = []
    for run in runs:
        run_ts.append(parcellate_run(run, cortical_indices, parcel_masks))
    
    all_ts = np.concatenate(run_ts, axis=0)
    corr = np.corrcoef(all_ts, rowvar=False)
    corr = np.asarray(corr, dtype=float)
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 0.0)
    
    # Parcel IDs should match manifest order (1, 2, 3, ...)
    parcel_ids = np.arange(1, len(manifest) + 1)
    manifest['parcel_id'] = parcel_ids

    if len(parcel_masks) != len(parcel_ids):
        raise ValueError(
            f"Mismatch between retained manifest rows ({len(parcel_ids)}) and loaded split masks ({len(parcel_masks)})"
        )
    
    return corr, parcel_ids, manifest, runs


def create_spring_network_plot(
    fc_matrix: np.ndarray,
    manifest: pd.DataFrame,
    subject: str,
    output_path: Path,
    infomap_color_map: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    edge_threshold_percentile: int = 95,
    hub_indices: Optional[np.ndarray] = None,
    node_size_metric: Optional[np.ndarray] = None,
    hub_metric_name: str = "strength",
) -> None:
    """Advanced network visualization with sophisticated edge thresholding and styling.
    
    Features:
    - Top 1-10% edges globally + top k=10 per node locally
    - Intra-network edges: light grey, low alpha
    - Inter-network edges: red/blue by sign, alpha scaled by weight^3
    - Edge width: weight^2
    - Node size: strength or participation coefficient
    - Hub highlighting: top 10 nodes by strength with labels
    """
    G = nx.Graph()
    
    # Extract network names, colors, and metrics
    network_names = manifest["network_name"].values if "network_name" in manifest.columns else ["unknown"] * len(manifest)
    
    # Compute node metric for sizing. Rank normalization keeps visible variation even in narrow ranges.
    if node_size_metric is None:
        node_size_metric = compute_abs_strength(fc_matrix)
    node_size_metric = np.asarray(node_size_metric, dtype=float)
    size_rank = normalize_to_percentile_rank(node_size_metric)
    node_sizes = 45.0 + 320.0 * (size_rank ** 1.7)
    
    # Create unique color map for networks
    unique_networks = sorted(set(network_names))
    colors_tab = plt.cm.tab20(np.linspace(0, 1, len(unique_networks)))
    network_color_map = {net: colors_tab[i] for i, net in enumerate(unique_networks)}
    if infomap_color_map:
        for net in unique_networks:
            if net in infomap_color_map:
                network_color_map[net] = infomap_color_map[net]
    
    # Add nodes with network assignment
    for i, network in enumerate(network_names):
        G.add_node(i, network=str(network), strength=float(node_size_metric[i]))
    
    # Edge thresholding: keep top 1-10% globally AND top k=10 per node locally
    fc_upper = np.triu_indices_from(fc_matrix, k=1)
    # Global threshold: top 1-10% edges
    edge_values_flat = np.abs(fc_matrix[fc_upper])
    pct = float(np.clip(edge_threshold_percentile, 0, 100))
    global_threshold = np.percentile(edge_values_flat, pct)
    
    # Per-node threshold: top k=10 edges per node
    k_per_node = 10
    per_node_edges = set()
    for i in range(len(fc_matrix)):
        # Get top k connections from this node
        top_k_indices = np.argsort(np.abs(fc_matrix[i, :]))[-k_per_node:]
        for j in top_k_indices:
            if i < j:
                per_node_edges.add((i, j))
            elif i > j:
                per_node_edges.add((j, i))
    
    # Keep edges that pass either criterion
    edges_to_add = []
    for idx, (i, j) in enumerate(zip(*fc_upper)):
        fc_val = fc_matrix[i, j]
        # Keep if: globally top 10% OR top 5 per node
        if np.abs(fc_val) >= global_threshold or (i, j) in per_node_edges:
            is_intra = network_names[i] == network_names[j]
            edges_to_add.append({
                'i': i, 'j': j, 'fc_val': fc_val, 
                'is_intra': is_intra, 'weight': np.abs(fc_val)
            })
    
    # Add edges to graph with metadata
    for edge in edges_to_add:
        G.add_edge(edge['i'], edge['j'], 
                   fc_val=edge['fc_val'],
                   is_intra=edge['is_intra'],
                   weight=edge['weight'])
    
    # Pre-position nodes in network clusters, then run spring layout
    pos = {}
    n_networks = len(unique_networks)
    radius = 5.0
    
    for net_idx, network in enumerate(unique_networks):
        angle = 2 * np.pi * net_idx / n_networks
        center_x = radius * np.cos(angle)
        center_y = radius * np.sin(angle)
        
        node_indices = [i for i, net in enumerate(network_names) if net == network]
        n_nodes = len(node_indices)
        
        for local_idx, node_i in enumerate(node_indices):
            scatter_radius = 0.4
            scatter_angle = 2 * np.pi * local_idx / max(n_nodes, 1) + (local_idx % 3) * 0.3
            radial_offset = scatter_radius * (1 + 0.2 * (local_idx % 2))
            pos[node_i] = (
                center_x + radial_offset * np.cos(scatter_angle),
                center_y + radial_offset * np.sin(scatter_angle)
            )
    
    # Spring layout with higher repulsion (k parameter)
    pos = nx.spring_layout(G, pos=pos, k=3.0, iterations=200, seed=42)
    
    # Identify hub nodes (provided by caller, or top 10 by node-size metric)
    if hub_indices is None:
        k_hubs = min(10, len(network_names))
        hub_indices = np.argsort(node_size_metric)[-k_hubs:][::-1]
    hub_indices_set = set(int(i) for i in hub_indices)
    
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Draw edges: intra-network (grey) vs inter-network (red/blue)
    for i, j in G.edges():
        edge_data = G.get_edge_data(i, j)
        fc_val = edge_data['fc_val']
        is_intra = edge_data['is_intra']
        weight = edge_data['weight']
        
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        
        if is_intra:
            # Intra-network: light grey, low alpha, thin
            ax.plot([x0, x1], [y0, y1], color='grey', alpha=0.08, linewidth=0.3, zorder=1)
        else:
            # Inter-network: color by sign, alpha by weight^3
            color = "red" if fc_val > 0 else "blue"
            alpha = (weight ** 3) * 0.8  # Scale by weight cubed
            width = (weight ** 2) * 3.0  # Width by weight squared
            ax.plot([x0, x1], [y0, y1], color=color, alpha=alpha, linewidth=width, zorder=2)
    
    # Draw nodes: size by strength, color by network
    for network in unique_networks:
        node_indices = [i for i, net in enumerate(network_names) if net == network]
        # Separate hubs from regular nodes
        regular_idx = [idx for idx in node_indices if idx not in hub_indices_set]
        hub_idx = [idx for idx in node_indices if idx in hub_indices_set]
        
        # Draw regular nodes
        if regular_idx:
            x_reg = [pos[i][0] for i in regular_idx]
            y_reg = [pos[i][1] for i in regular_idx]
            sizes_reg = [node_sizes[i] for i in regular_idx]
            ax.scatter(x_reg, y_reg, s=sizes_reg, c=[network_color_map[network]] * len(regular_idx),
                      label=network, alpha=0.8, edgecolors="black", linewidth=0.8, zorder=3)
        
        # Draw hub nodes with larger size and black outline
        if hub_idx:
            x_hub = [pos[i][0] for i in hub_idx]
            y_hub = [pos[i][1] for i in hub_idx]
            sizes_hub = [node_sizes[i] * 1.35 + 50.0 for i in hub_idx]
            ax.scatter(x_hub, y_hub, s=sizes_hub, c=[network_color_map[network]] * len(hub_idx),
                      alpha=0.95, edgecolors="black", linewidth=2.0, zorder=4)
    
    # Add labels for hub nodes
    for hub_node in hub_indices:
        x, y = pos[hub_node]
        # Use abbreviated network name
        net_name = network_names[hub_node]
        label = net_name[:4]  # First 4 chars
        ax.text(x, y, label, fontsize=6, ha='center', va='center', 
               fontweight='bold', color='white', zorder=5)
    
    ax.set_title(
        f"sub-{subject} Split Parcels Network (top {100 - int(pct)}% edges + top-{k_per_node} per node; hubs={hub_metric_name})",
                fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9, markerscale=0.7)
    ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_circular_network_plot(
    network_fc: np.ndarray,
    network_names: np.ndarray,
    title_label: str,
    output_path: Path,
    edge_threshold_percentile: int = 95,
    hub_selection_metric: str = "strength",
    infomap_color_map: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
) -> None:
    """Create a circular/star network plot with one center node.

    - Frontoparietal/FPN is placed in the center when present.
    - Edges are thresholded by top X% absolute FC.
    - Edge thickness scales with absolute FC.
    - Node size scales with network metric (strength or participation coefficient).
    """
    fc = np.asarray(network_fc, dtype=float)
    names = np.asarray(network_names).astype(str)

    if fc.ndim != 2 or fc.shape[0] != fc.shape[1]:
        raise ValueError(f"network_fc must be square, got shape={fc.shape}")
    if fc.shape[0] != len(names):
        raise ValueError(
            f"network_names length ({len(names)}) does not match FC size ({fc.shape[0]})"
        )

    n_nodes = len(names)
    if n_nodes == 0:
        raise ValueError("Cannot plot empty network matrix")

    np.fill_diagonal(fc, 0.0)
    abs_fc = np.abs(fc)
    abs_strength = np.sum(abs_fc, axis=1)

    # Prefer the frontoparietal network as center; otherwise use strongest node.
    normalized_names = [str(n).strip().lower().replace("_", " ") for n in names]
    center_idx = -1

    # 1) Exact preferred labels first.
    preferred_exact = {
        "frontoparietal",
        "fronto parietal",
        "fpn",
        "fpna",
        "fpnb",
        "fpn a",
        "fpn b",
    }
    for idx, label in enumerate(normalized_names):
        if label in preferred_exact:
            center_idx = idx
            break

    # 2) Then broader matches.
    if center_idx < 0:
        for idx, label in enumerate(normalized_names):
            if ("frontoparietal" in label) or ("fpn" in label):
                center_idx = idx
                break

    if center_idx < 0:
        center_idx = int(np.argmax(abs_strength))

    modules = names
    pc, pos_strength = compute_hub_metrics(fc, modules)
    node_metric = abs_strength if hub_selection_metric == "strength" else pc
    node_rank = normalize_to_percentile_rank(node_metric)
    node_sizes = 380.0 + 860.0 * (node_rank ** 1.8)

    # Build position map: center node + ring nodes.
    pos: Dict[int, Tuple[float, float]] = {center_idx: (0.0, 0.0)}
    ring_indices = [i for i in range(n_nodes) if i != center_idx]
    ring_r = 5.0
    for rank, node_idx in enumerate(ring_indices):
        angle = 2.0 * np.pi * float(rank) / float(max(1, len(ring_indices)))
        pos[node_idx] = (ring_r * np.cos(angle), ring_r * np.sin(angle))

    # Determine threshold on upper-triangle edges.
    tri = np.triu_indices(n_nodes, k=1)
    edge_abs_values = abs_fc[tri]
    edge_abs_values = edge_abs_values[np.isfinite(edge_abs_values)]
    pct = float(np.clip(edge_threshold_percentile, 0, 100))
    threshold = np.percentile(edge_abs_values, pct) if edge_abs_values.size > 0 else np.inf

    if infomap_color_map is None:
        infomap_color_map = {}

    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw edges first.
    kept_edges = 0
    max_abs = float(np.max(edge_abs_values)) if edge_abs_values.size > 0 else 1.0
    min_abs = float(np.min(edge_abs_values)) if edge_abs_values.size > 0 else 0.0
    denom = max(max_abs - min_abs, 1e-9)

    for i, j in zip(*tri):
        val = float(fc[i, j])
        w = abs(val)
        if not np.isfinite(w) or w < threshold:
            continue
        kept_edges += 1

        norm = (w - min_abs) / denom
        line_w = 0.5 + 5.5 * (norm ** 1.5)
        alpha = 0.15 + 0.75 * (norm ** 1.1)
        color = "#cf2f27" if val >= 0 else "#2855a6"

        x0, y0 = pos[i]
        x1, y1 = pos[j]
        ax.plot([x0, x1], [y0, y1], color=color, alpha=alpha, linewidth=line_w, zorder=1)

    # Draw nodes and labels.
    for i, name in enumerate(names):
        rgba = infomap_color_map.get(str(name), color_from_name(str(name), alpha=1.0))
        x, y = pos[i]
        edge_lw = 2.0 if i == center_idx else 1.0
        ax.scatter(
            [x],
            [y],
            s=[node_sizes[i] * (1.15 if i == center_idx else 1.0)],
            c=[rgba],
            edgecolors="black",
            linewidth=edge_lw,
            alpha=0.95,
            zorder=3,
        )
        ax.text(
            x,
            y,
            str(name),
            fontsize=8,
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
            zorder=4,
        )

    kept_pct = 100 - int(np.clip(edge_threshold_percentile, 0, 100))
    node_metric_label = "|FC| strength" if hub_selection_metric == "strength" else "participation"
    center_name = str(names[center_idx])
    ax.set_title(
        (
            f"{title_label} Circular Network Plot"
            f" (top {kept_pct}% edges, node metric={node_metric_label}, center={center_name}, edges kept={kept_edges})"
        ),
        fontsize=12,
        fontweight="bold",
    )
    ax.axis("off")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute resting-state FC and hub metrics for Glasser 360 or split network parcels.")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--fmriprep-base", default=DEFAULT_FMRIPREP_BASE)
    parser.add_argument("--assignment-dir", default=DEFAULT_ASSIGNMENT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--network-label-base", default=DEFAULT_NETWORK_LABEL_BASE)
    parser.add_argument("--parcellation-path", default=None)
    parser.add_argument(
        "--analysis-level",
        choices=[ANALYSIS_LEVEL_NETWORK, ANALYSIS_LEVEL_NETWORK_PARCEL],
        default=None,
        help="Analysis level: 'network' (Glasser 360) or 'network_parcel' (split parcels).",
    )
    parser.add_argument(
        "--fpn-mode",
        choices=[FPN_MODE_UNIFIED, FPN_MODE_SPLIT],
        default=FPN_MODE_UNIFIED,
        help="FPN handling mode for assignment-derived columns: unified (default) or split.",
    )
    parser.add_argument("--overlap-threshold", type=float, default=DEFAULT_OVERLAP_THRESHOLD, help="Overlap threshold for split parcels")
    parser.add_argument(
        "--save-network-fc",
        action="store_true",
        help="Also save parcel-FC collapsed to network-level FC (network x network).",
    )
    parser.add_argument(
        "--split-parcels",
        action="store_true",
        help="Deprecated alias for --analysis-level network_parcel.",
    )
    parser.add_argument("--edge-threshold-percentile", type=int, default=95, help="Percentile threshold for spring plot edges")
    parser.add_argument("--top-hubs-k", type=int, default=10, help="Number of top hubs to label in spring plot and optional dlabel export")
    parser.add_argument(
        "--hub-selection-metric",
        choices=["strength", "participation"],
        default="strength",
        help="Metric used to select top hubs for labels/export.",
    )
    parser.add_argument(
        "--save-top-hubs-dlabel",
        action="store_true",
        help="Save sub-XX_top_hubs_network_color.dlabel.nii with only top hubs labeled (all other grayordinates = 0).",
    )
    parser.add_argument(
        "--plot-only-from-saved-fc",
        action="store_true",
        help="For split-parcel analysis: regenerate spring plot (and optional top-hubs dlabel) from subject_fc_network_parcels.npz without recomputing FC.",
    )
    return parser.parse_args()


def resolve_analysis_level(args: argparse.Namespace) -> str:
    if args.analysis_level is not None:
        return str(args.analysis_level)
    if args.split_parcels:
        return ANALYSIS_LEVEL_NETWORK_PARCEL
    return ANALYSIS_LEVEL_NETWORK


def get_subject_infomap_color_map(subject: str, network_label_base: str, manifest: pd.DataFrame) -> Dict[str, Tuple[float, float, float, float]]:
    if "network_id" not in manifest.columns or "network_name" not in manifest.columns:
        return {}

    network_path = subject_network_label_path(network_label_base, subject)
    if not network_path.exists():
        return {}

    img = nib.load(str(network_path))
    rgba_by_id = extract_network_colors_from_dlabel(img)
    if not rgba_by_id:
        return {}

    mapping: Dict[str, Tuple[float, float, float, float]] = {}
    pairs = manifest[["network_id", "network_name"]].drop_duplicates()
    for _, row in pairs.iterrows():
        try:
            network_id = int(row["network_id"])
            network_name = str(row["network_name"])
        except (TypeError, ValueError):
            continue

        if network_id in rgba_by_id:
            mapping[network_name] = rgba_by_id[network_id]

    return mapping


def process_subject_split_parcels(subject: str, args: argparse.Namespace) -> str:
    print(f"sub-{subject}: computing FC for split network parcels...")
    fc, parcel_ids, manifest, runs = compute_subject_fc_network_parcels(
        subject, args.fmriprep_base, args.assignment_dir, args.overlap_threshold
    )

    subject_dir = ensure_dir(Path(args.output_dir) / f"sub-{subject}" / "static")

    np.savez_compressed(
        subject_dir / "subject_fc_network_parcels.npz",
        fc=fc,
        parcel_ids=parcel_ids,
        n_runs=len(runs),
        overlap_threshold=args.overlap_threshold,
    )

    if "network_name" not in manifest.columns:
        raise ValueError("manifest missing 'network_name' column")
    modules = manifest["network_name"].astype(str).to_numpy()
    pc, strength = compute_hub_metrics(fc, modules)
    abs_strength = compute_abs_strength(fc)

    if args.hub_selection_metric == "participation":
        hub_metric = pc
    else:
        hub_metric = abs_strength

    k_hubs = max(1, min(int(args.top_hubs_k), len(parcel_ids)))
    hub_indices_desc = np.argsort(hub_metric)[-k_hubs:][::-1]

    metric_df = pd.DataFrame(
        {
            "subject": subject,
            "parcel_id": parcel_ids.astype(int),
            "network_name": modules,
            "participation_coefficient": pc,
            "strength": strength,
            "n_rest_runs": len(runs),
        }
    )
    metric_df.to_csv(subject_dir / "static_hub_metrics_subject.csv", index=False)

    if args.save_network_fc:
        network_fc, network_names, pair_counts = collapse_parcel_fc_to_network_fc(fc, modules)
        np.savez_compressed(
            subject_dir / "subject_fc_network_collapsed.npz",
            fc=network_fc,
            network_names=network_names,
            pair_counts=pair_counts,
            source_analysis_level=ANALYSIS_LEVEL_NETWORK_PARCEL,
            source_parcel_count=len(parcel_ids),
            n_runs=len(runs),
            overlap_threshold=args.overlap_threshold,
        )

    infomap_color_map = get_subject_infomap_color_map(subject, args.network_label_base, manifest)

    if args.save_top_hubs_dlabel:
        write_top_hubs_split_dlabel(
            subject=subject,
            manifest=manifest,
            hub_indices_desc=hub_indices_desc,
            output_path=subject_dir / "top_hubs_network_color.dlabel.nii",
            assignment_dir=args.assignment_dir,
            network_label_base=args.network_label_base,
            parcellation_path=args.parcellation_path,
            infomap_color_map=infomap_color_map,
        )

    create_spring_network_plot(
        fc,
        manifest,
        subject,
        subject_dir
        / build_spring_plot_filename(
            analysis_level=ANALYSIS_LEVEL_NETWORK_PARCEL,
            edge_threshold_percentile=args.edge_threshold_percentile,
            top_hubs_k=args.top_hubs_k,
            hub_selection_metric=args.hub_selection_metric,
        ),
        infomap_color_map=infomap_color_map,
        edge_threshold_percentile=args.edge_threshold_percentile,
        hub_indices=hub_indices_desc,
        node_size_metric=abs_strength,
        hub_metric_name=args.hub_selection_metric,
    )

    return f"sub-{subject}: completed split parcel FC ({len(runs)} runs, {len(parcel_ids)} parcels)"


def process_subject_split_parcels_plot_only(subject: str, args: argparse.Namespace) -> str:
    subject_dir = Path(args.output_dir) / f"sub-{subject}" / "static"
    fc_path = subject_dir / "subject_fc_network_parcels.npz"
    if not fc_path.exists():
        raise FileNotFoundError(
            f"Missing saved FC for plot-only mode: {fc_path}. Run full split-parcel FC first."
        )

    with np.load(fc_path) as saved:
        if "fc" not in saved:
            raise ValueError(f"Saved FC file missing 'fc' array: {fc_path}")
        fc = np.asarray(saved["fc"], dtype=float)
        if fc.ndim != 2 or fc.shape[0] != fc.shape[1]:
            raise ValueError(f"Saved FC is not square in {fc_path}: shape={fc.shape}")

        if "parcel_ids" in saved:
            parcel_ids = np.asarray(saved["parcel_ids"], dtype=int)
        else:
            parcel_ids = np.arange(1, fc.shape[0] + 1, dtype=int)

        overlap_threshold = float(saved["overlap_threshold"]) if "overlap_threshold" in saved else float(args.overlap_threshold)

    manifest = load_split_parcel_manifest_retained(subject, args.assignment_dir, overlap_threshold)
    manifest = manifest.sort_values("split_mask_path").reset_index(drop=True)
    manifest["parcel_id"] = np.arange(1, len(manifest) + 1)

    if "network_name" not in manifest.columns:
        raise ValueError("manifest missing 'network_name' column")
    if len(manifest) != fc.shape[0]:
        raise ValueError(
            f"Saved FC parcel count ({fc.shape[0]}) does not match retained manifest rows ({len(manifest)}). "
            f"Check overlap threshold / assignment directory consistency."
        )

    modules = manifest["network_name"].astype(str).to_numpy()
    pc, _ = compute_hub_metrics(fc, modules)
    abs_strength = compute_abs_strength(fc)

    if args.hub_selection_metric == "participation":
        hub_metric = pc
    else:
        hub_metric = abs_strength

    k_hubs = max(1, min(int(args.top_hubs_k), len(parcel_ids)))
    hub_indices_desc = np.argsort(hub_metric)[-k_hubs:][::-1]

    infomap_color_map = get_subject_infomap_color_map(subject, args.network_label_base, manifest)

    if args.save_top_hubs_dlabel:
        write_top_hubs_split_dlabel(
            subject=subject,
            manifest=manifest,
            hub_indices_desc=hub_indices_desc,
            output_path=subject_dir / "top_hubs_network_color.dlabel.nii",
            assignment_dir=args.assignment_dir,
            network_label_base=args.network_label_base,
            parcellation_path=args.parcellation_path,
            infomap_color_map=infomap_color_map,
        )

    create_spring_network_plot(
        fc,
        manifest,
        subject,
        subject_dir
        / build_spring_plot_filename(
            analysis_level=ANALYSIS_LEVEL_NETWORK_PARCEL,
            edge_threshold_percentile=args.edge_threshold_percentile,
            top_hubs_k=args.top_hubs_k,
            hub_selection_metric=args.hub_selection_metric,
        ),
        infomap_color_map=infomap_color_map,
        edge_threshold_percentile=args.edge_threshold_percentile,
        hub_indices=hub_indices_desc,
        node_size_metric=abs_strength,
        hub_metric_name=args.hub_selection_metric,
    )

    return (
        f"sub-{subject}: regenerated split parcel spring plot from saved FC "
        f"({len(parcel_ids)} parcels, overlap_threshold={overlap_threshold:.2f})"
    )


def process_subject_network_level(
    subject: str,
    args: argparse.Namespace,
    output_dir: Path,
    parcel_ref: Optional[np.ndarray],
) -> tuple[np.ndarray, str]:
    assignment = load_assignment_for_subject(subject, args.assignment_dir)
    assignment = assignment.sort_values("parcel_id").reset_index(drop=True)

    fc, parcel_ids, runs = compute_subject_fc(subject, args.fmriprep_base, args.parcellation_path)
    if len(assignment) != len(parcel_ids):
        raise ValueError(
            f"Assignment rows ({len(assignment)}) do not match parcel count ({len(parcel_ids)})"
        )

    if parcel_ref is None:
        parcel_ref = parcel_ids.copy()
    elif not np.array_equal(parcel_ref, parcel_ids):
        raise ValueError("Parcel ID ordering mismatch across subjects")

    modules = assignment["assigned_network_name"].astype(str).to_numpy()
    pc, strength = compute_hub_metrics(fc, modules)
    network_fc, network_names, pair_counts = collapse_parcel_fc_to_network_fc(fc, modules)

    is_fpn_series = assignment.get("is_fpn", pd.Series(np.zeros(len(parcel_ids), dtype=int))).fillna(0).astype(int)
    if args.fpn_mode == FPN_MODE_SPLIT:
        fpna_series = assignment.get("fpna_selected", pd.Series(np.zeros(len(parcel_ids), dtype=int))).fillna(0).astype(int)
        fpnb_series = assignment.get("fpnb_selected", pd.Series(np.zeros(len(parcel_ids), dtype=int))).fillna(0).astype(int)
    else:
        fpna_series = pd.Series(np.zeros(len(parcel_ids), dtype=int))
        fpnb_series = pd.Series(np.zeros(len(parcel_ids), dtype=int))

    subject_dir = ensure_dir(output_dir / f"sub-{subject}" / "static")
    np.savez_compressed(subject_dir / "subject_fc_360x360.npz", fc=fc, parcel_ids=parcel_ids, n_runs=len(runs))

    metric_df = pd.DataFrame(
        {
            "subject": subject,
            "parcel_id": parcel_ids.astype(int),
            "parcel_name": assignment["parcel_name"].values,
            "assigned_network_name": assignment["assigned_network_name"].values,
            "is_fpn": is_fpn_series.values,
            "fpna_selected": fpna_series.values,
            "fpnb_selected": fpnb_series.values,
            "participation_coefficient": pc,
            "strength": strength,
            "n_rest_runs": len(runs),
        }
    )
    metric_df.to_csv(subject_dir / "static_hub_metrics_subject.csv", index=False)

    infomap_color_map = get_subject_infomap_color_map(
        subject,
        args.network_label_base,
        assignment.rename(
            columns={
                "assigned_network_id": "network_id",
                "assigned_network_name": "network_name",
            }
        ),
    )

    create_circular_network_plot(
        network_fc=network_fc,
        network_names=network_names,
        title_label=f"sub-{subject}",
        output_path=subject_dir
        / build_circular_plot_filename(
            analysis_level=ANALYSIS_LEVEL_NETWORK,
            edge_threshold_percentile=args.edge_threshold_percentile,
            hub_selection_metric=args.hub_selection_metric,
        ),
        edge_threshold_percentile=args.edge_threshold_percentile,
        hub_selection_metric=args.hub_selection_metric,
        infomap_color_map=infomap_color_map,
    )

    if args.save_network_fc:
        np.savez_compressed(
            subject_dir / "subject_fc_network_collapsed.npz",
            fc=network_fc,
            network_names=network_names,
            pair_counts=pair_counts,
            source_analysis_level=ANALYSIS_LEVEL_NETWORK,
            source_parcel_count=len(parcel_ids),
            n_runs=len(runs),
        )

    return parcel_ref, f"sub-{subject}: completed static FC and hub metrics ({len(runs)} runs)"


def main() -> None:
    args = parse_args()
    analysis_level = resolve_analysis_level(args)
    
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = discover_subjects_from_subdirs(args.fmriprep_base)

    if not subjects:
        raise ValueError("No subjects found")

    failures = []

    output_dir = ensure_dir(args.output_dir)
    parcel_ref: Optional[np.ndarray] = None

    for subject in subjects:
        try:
            if analysis_level == ANALYSIS_LEVEL_NETWORK_PARCEL:
                if args.plot_only_from_saved_fc:
                    status = process_subject_split_parcels_plot_only(subject, args)
                else:
                    status = process_subject_split_parcels(subject, args)
            else:
                if args.plot_only_from_saved_fc:
                    raise ValueError("--plot-only-from-saved-fc is only supported with --analysis-level network_parcel")
                parcel_ref, status = process_subject_network_level(subject, args, output_dir, parcel_ref)
            print(status)
        except Exception as exc:
            failures.append((subject, str(exc)))
            print(f"sub-{subject}: FAILED -> {exc}")

    if failures:
        print("\nCompleted with failures:")
        for sub, msg in failures:
            print(f"  sub-{sub}: {msg}")



if __name__ == "__main__":
    main()
