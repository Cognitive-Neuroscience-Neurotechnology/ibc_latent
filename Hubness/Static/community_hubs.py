"""Compute and visualize community-level functional connectivity with spring embedding.

Steps:
1) Load community timeseries from subject's fmriprep-derived individual networks
2) Compute Pearson correlation matrix between communities (community FC matrix)
3) Extract community names and network assignments from dlabel files
4) Create spring-embedded network plot with:
   - Communities as nodes, colored by network
   - Functional connectivity as edges (threshold-filtered)
   - Node size by strength, hub highlighting
5) Save FC matrix, metrics, and visualizations to output directory

Outputs:
- subject_community_fc.npz: Community-level FC correlation matrix + metadata
- community_hub_metrics_subject.csv: Strength per community and network assignment
- spring_plot_communities_<analysis>_edgesXXX_<metric>.png: Spring visualization
"""
# Run with: python Hubness/Static/community_hubs.py --subjects 04 ...

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

DEFAULT_NETWORK_LABEL_BASE = "/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks/derived_networks"
DEFAULT_OUTPUT_DIR = "/ptmp/hmueller2/2025_ibc_latent/outputs/hubness"


def subject_resting_state_dir(subject: str, network_label_base: str) -> Path:
    subject_dir = Path(network_label_base) / f"sub-{subject}" / "resting_state"
    if not subject_dir.exists():
        raise FileNotFoundError(f"No resting_state directory for sub-{subject}")
    return subject_dir


def resolve_subject_dtseries(subject: str, network_label_base: str) -> Path:
    """Pick the subject-level dtseries used to derive community timecourses."""
    subject_dir = subject_resting_state_dir(subject, network_label_base)
    path = subject_dir / f"sub-{subject}_all-tasks_concatenated_cleaned_fsLR_cortexOnly.dtseries.nii"
    if path.exists():
        return path
    raise FileNotFoundError(f"Missing cortex-only dtseries for sub-{subject}: {path}")


def discover_subjects_from_network_base(network_label_base: str) -> List[str]:
    """Discover subjects from individual network directory structure."""
    base_path = Path(network_label_base)
    subjects = []
    for sub_dir in sorted(base_path.glob("sub-*")):
        if sub_dir.is_dir():
            subject = sub_dir.name.replace("sub-", "")
            subjects.append(subject)
    return subjects


def load_infomap_community_definition(
    subject: str, network_label_base: str
) -> Tuple[np.ndarray, List[str], Dict[int, Tuple[float, float, float, float]]]:
    """Load row-wise InfoMap community masks and per-community network labels.

    Returns:
        community_masks: bool array of shape (n_communities, n_grayordinates)
        community_names: network label for each community (length n_communities)
        community_colors: RGBA color for each community index
    """
    subject_dir = subject_resting_state_dir(subject, network_label_base)
    label_file = (
        subject_dir
        / "Bipartite_PhysicalCommunities+AlgorithmicLabeling_InfoMapCommunities.dlabel.nii"
    )
    if not label_file.exists():
        raise FileNotFoundError(f"Missing label file: {label_file}")

    img = nib.load(str(label_file))
    data = np.asarray(img.get_fdata(), dtype=float)
    if data.ndim != 2:
        raise ValueError(f"Unexpected InfoMap dlabel shape for {label_file}: {data.shape}")

    label_axis = img.header.get_axis(0)
    label_table = label_axis.label[0]

    community_masks = data > 0
    n_communities = community_masks.shape[0]
    community_names: List[str] = []
    community_colors: Dict[int, Tuple[float, float, float, float]] = {}

    for i in range(n_communities):
        row = data[i]
        nonzero = row[np.isfinite(row) & (row > 0)]
        if nonzero.size == 0:
            community_names.append("Unknown")
            community_colors[i] = (0.5, 0.5, 0.5, 1.0)
            continue

        network_id = int(np.round(np.median(nonzero)))
        if network_id in label_table:
            name, rgba = label_table[network_id]
            name_str = str(name).strip()
            if name_str.lower() in ("noise", "???"):
                community_names.append("Unknown")
                community_colors[i] = (0.5, 0.5, 0.5, 1.0)
            else:
                community_names.append(name_str)
                community_colors[i] = tuple(float(x) for x in rgba)
        else:
            community_names.append(f"Network_{network_id}")
            community_colors[i] = (0.5, 0.5, 0.5, 1.0)

    return community_masks, community_names, community_colors


def load_community_timeseries(subject: str, network_label_base: str) -> Tuple[np.ndarray, List[str], List[str], Dict[int, Tuple[float, float, float, float]], Dict[str, Tuple[float, float, float, float]]]:
    """Build community timecourses from subject dtseries and InfoMap community masks.

    Returns:
        community_ts: (n_timepoints, n_communities) community timecourses
        community_names: network label name per community
        file_paths: loaded source files
        community_colors: color per community index
        network_canonical_colors: complete network->color map for all 20 networks
    """
    dtseries_file = resolve_subject_dtseries(subject, network_label_base)
    community_masks, community_names, community_colors = load_infomap_community_definition(subject, network_label_base)
    
    # Build complete network color map from label table
    network_canonical_colors = _build_network_color_map(subject, network_label_base)

    ts_img = nib.load(str(dtseries_file))
    ts_data = np.asarray(ts_img.get_fdata(dtype=np.float32), dtype=float)
    if ts_data.ndim != 2:
        raise ValueError(f"Expected 2D dtseries, got shape {ts_data.shape} for {dtseries_file}")

    n_timepoints, n_grayordinates = ts_data.shape
    if community_masks.shape[1] != n_grayordinates:
        raise ValueError(
            "Grayordinate mismatch between dtseries and community dlabel: "
            f"dtseries={n_grayordinates}, dlabel={community_masks.shape[1]}"
        )

    n_communities = community_masks.shape[0]
    community_ts = np.zeros((n_timepoints, n_communities), dtype=float)
    for i in range(n_communities):
        mask = community_masks[i]
        if not np.any(mask):
            community_ts[:, i] = 0.0
            continue
        community_ts[:, i] = np.nanmean(ts_data[:, mask], axis=1)

    return community_ts, community_names, [str(dtseries_file)], community_colors, network_canonical_colors


def _build_network_color_map(subject: str, network_label_base: str) -> Dict[str, Tuple[float, float, float, float]]:
    """Build canonical network->color map from InfoMap label table."""
    subject_dir = subject_resting_state_dir(subject, network_label_base)
    label_file = (
        subject_dir
        / "Bipartite_PhysicalCommunities+AlgorithmicLabeling_InfoMapCommunities.dlabel.nii"
    )
    img = nib.load(str(label_file))
    label_axis = img.header.get_axis(0)
    label_table = label_axis.label[0]
    
    network_colors: Dict[str, Tuple[float, float, float, float]] = {}
    for network_id in sorted(label_table.keys()):
        name_str, rgba = label_table[network_id]
        name_str = str(name_str).strip()
        if name_str.lower() not in ('noise', '???'):
            network_colors[name_str] = tuple(float(x) for x in rgba)
    return network_colors


def extract_network_from_community_name(community_name: str) -> str:
    """Normalize the label name used for metrics and plotting.

    Keep the full InfoMap label name so distinct labels such as
    'Default_Parietal' and 'Default_Anterolateral' remain separate.
    """
    return str(community_name).strip()


def compute_community_fc(community_ts: np.ndarray) -> np.ndarray:
    """Compute community-level FC correlation matrix.
    
    Args:
        community_ts: (n_timepoints, n_communities) array
        
    Returns:
        fc_matrix: (n_communities, n_communities) correlation matrix
    """
    # Normalize each community timeseries
    ts = np.asarray(community_ts, dtype=float)
    n_timepoints, n_communities = ts.shape
    ts_norm = np.zeros_like(ts)

    for i in range(n_communities):
        ts_i = ts[:, i]
        valid = np.isfinite(ts_i)
        if np.sum(valid) > 1:
            mean = np.mean(ts_i[valid])
            std = np.std(ts_i[valid])
            if std > 0:
                ts_norm[:, i] = (ts_i - mean) / std
            else:
                ts_norm[:, i] = ts_i - mean
        else:
            ts_norm[:, i] = ts_i

    # Compute correlation (communities x communities)
    corr = np.corrcoef(ts_norm, rowvar=False)
    corr = np.asarray(corr, dtype=float)
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 0.0)

    return corr


def compute_strength(fc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute positive and absolute strength from FC matrix.
    
    Returns:
        pos_strength: Sum of positive connections per node
        abs_strength: Sum of absolute value of connections per node
    """
    fc_pos = np.array(fc, dtype=float)
    fc_pos[fc_pos < 0] = 0.0
    np.fill_diagonal(fc_pos, 0.0)
    pos_strength = np.sum(fc_pos, axis=1)

    fc_abs = np.abs(np.asarray(fc, dtype=float))
    np.fill_diagonal(fc_abs, 0.0)
    abs_strength = np.sum(fc_abs, axis=1)

    return pos_strength, abs_strength


def normalize_to_percentile_rank(values: np.ndarray) -> np.ndarray:
    """Normalize values to percentile ranks [0, 1]."""
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


def create_community_spring_plot(
    fc_matrix: np.ndarray,
    community_names: List[str],
    community_colors: Dict[int, Tuple[float, float, float, float]],
    subject: str,
    output_path: Path,
    edge_threshold_percentile: int = 90,
    hub_metric_name: str = "strength",
    network_canonical_colors: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
) -> None:
    """Create spring-embedded network plot of communities.
    
    Features:
    - Communities as nodes, colored by network
    - Top edges globally + top k per node
    - Intra-network edges: light grey, low alpha
    - Inter-network edges: red (positive) / blue (negative), alpha scaled by weight^3
    - Node size: absolute strength
    - Hub highlighting: top 10 nodes by strength
    """
    G = nx.Graph()
    layout_graph = nx.Graph()

    # Compute node metrics
    pos_strength, abs_strength = compute_strength(fc_matrix)
    size_rank = normalize_to_percentile_rank(abs_strength)
    node_sizes = 50.0 + 350.0 * (size_rank ** 1.7)

    # Extract network assignment for each community
    network_names = [extract_network_from_community_name(name) for name in community_names]
    unique_networks = sorted(set(network_names))

    # Create network color map: prefer canonical colors, fall back to community colors
    network_color_map = {}
    if network_canonical_colors:
        network_color_map.update(network_canonical_colors)
    for i, net_name in enumerate(unique_networks):
        if net_name not in network_color_map:
            # Find a community in this network and use its color
            for comm_id, comm_name in enumerate(community_names):
                if extract_network_from_community_name(comm_name) == net_name:
                    if comm_id in community_colors:
                        network_color_map[net_name] = community_colors[comm_id]
                        break
        if net_name not in network_color_map:
            # Fallback: use HSV colormap
            hue = (sum(ord(ch) for ch in str(net_name)) % 360) / 360.0
            rgba = tuple(float(c) for c in plt.cm.hsv(hue)[:3]) + (1.0,)
            network_color_map[net_name] = rgba

    # Add nodes
    for comm_id, comm_name in enumerate(community_names):
        net = extract_network_from_community_name(comm_name)
        G.add_node(comm_id, community=str(comm_name), network=str(net), strength=float(abs_strength[comm_id]))
        layout_graph.add_node(comm_id)

    # Build a dense weighted graph for layout so spring forces can organize the nodes
    fc_upper = np.triu_indices_from(fc_matrix, k=1)
    for i, j in zip(*fc_upper):
        weight = float(np.abs(fc_matrix[i, j]))
        if weight > 0:
            layout_graph.add_edge(i, j, weight=weight)

    # Edge thresholding: top X% of positive edges globally
    positive_mask = fc_matrix[fc_upper] > 0
    positive_values = fc_matrix[fc_upper][positive_mask]
    pct = float(np.clip(edge_threshold_percentile, 0, 100))
    global_threshold = np.percentile(positive_values, pct) if len(positive_values) > 0 else 0.0

    # Add edges to graph (positive edges only)
    edges_to_add = []
    for idx, (i, j) in enumerate(zip(*fc_upper)):
        fc_val = fc_matrix[i, j]
        if fc_val > 0 and fc_val >= global_threshold:
            is_intra = network_names[i] == network_names[j]
            edges_to_add.append({
                'i': i, 'j': j, 'fc_val': fc_val,
                'is_intra': is_intra, 'weight': fc_val
            })

    for edge in edges_to_add:
        G.add_edge(edge['i'], edge['j'],
                   fc_val=edge['fc_val'],
                   is_intra=edge['is_intra'],
                   weight=edge['weight'])

    # Spring layout in two stages: first arrange network clusters, then arrange communities within each cluster.
    network_graph = nx.Graph()
    for net_name in unique_networks:
        network_graph.add_node(net_name)

    for idx_a, net_a in enumerate(unique_networks):
        nodes_a = [i for i, net in enumerate(network_names) if net == net_a]
        for net_b in unique_networks[idx_a + 1:]:
            nodes_b = [i for i, net in enumerate(network_names) if net == net_b]
            inter_weight = float(np.sum(np.abs(fc_matrix[np.ix_(nodes_a, nodes_b)])))
            if inter_weight > 0:
                network_graph.add_edge(net_a, net_b, weight=inter_weight)

    network_centers = nx.spring_layout(
        network_graph,
        weight="weight",
        seed=42,
        k=1.8 / np.sqrt(max(len(network_graph), 1)),
        iterations=300,
        scale=7.0,
    )

    pos = {}
    for net_name in unique_networks:
        node_indices = [i for i, net in enumerate(network_names) if net == net_name]
        if not node_indices:
            continue

        subgraph = layout_graph.subgraph(node_indices).copy()
        if len(subgraph) == 1:
            local_pos = {node_indices[0]: np.array([0.0, 0.0])}
        else:
            local_pos = nx.spring_layout(
                subgraph,
                weight="weight",
                seed=42,
                k=1.2 / np.sqrt(len(subgraph)),
                iterations=250,
                scale=1.5,
            )

        center = np.asarray(network_centers[net_name], dtype=float)
        for node_i in node_indices:
            offset = np.asarray(local_pos[node_i], dtype=float)
            pos[node_i] = tuple(center + offset)

    # Identify hub nodes
    k_hubs = min(10, len(community_names))
    hub_indices = np.argsort(abs_strength)[-k_hubs:][::-1]
    hub_indices_set = set(int(i) for i in hub_indices)

    fig, ax = plt.subplots(figsize=(16, 14))

    # Draw edges (positive only)
    for i, j in G.edges():
        edge_data = G.get_edge_data(i, j)
        fc_val = edge_data['fc_val']
        is_intra = edge_data['is_intra']
        weight = edge_data['weight']

        x0, y0 = pos[i]
        x1, y1 = pos[j]

        if is_intra:
            # Intra-network: light grey
            ax.plot([x0, x1], [y0, y1], color='grey', alpha=0.15, linewidth=0.8, zorder=1)
        else:
            # Inter-network: red (positive)
            alpha = (weight ** 2) * 0.7
            width = (weight ** 2) * 5.0
            ax.plot([x0, x1], [y0, y1], color='red', alpha=alpha, linewidth=width, zorder=2)

    # Draw nodes
    for net_name in unique_networks:
        node_indices = [i for i, net in enumerate(network_names) if net == net_name]
        regular_idx = [idx for idx in node_indices if idx not in hub_indices_set]
        hub_idx = [idx for idx in node_indices if idx in hub_indices_set]

        # Regular nodes
        if regular_idx:
            x_reg = [pos[i][0] for i in regular_idx]
            y_reg = [pos[i][1] for i in regular_idx]
            sizes_reg = [node_sizes[i] for i in regular_idx]
            ax.scatter(x_reg, y_reg, s=sizes_reg, c=[network_color_map[net_name]] * len(regular_idx),
                      label=net_name, alpha=0.8, edgecolors="black", linewidth=0.8, zorder=3)

        # Hub nodes
        if hub_idx:
            x_hub = [pos[i][0] for i in hub_idx]
            y_hub = [pos[i][1] for i in hub_idx]
            sizes_hub = [node_sizes[i] * 1.35 + 50.0 for i in hub_idx]
            ax.scatter(x_hub, y_hub, s=sizes_hub, c=[network_color_map[net_name]] * len(hub_idx),
                      alpha=0.95, edgecolors="black", linewidth=2.0, zorder=4)

    # Add labels for hub nodes
    for hub_node in hub_indices:
        x, y = pos[hub_node]
        comm_name = community_names[hub_node]
        # Use abbreviated name
        label = comm_name[:6]
        ax.text(x, y, label, fontsize=6, ha='center', va='center',
               fontweight='bold', color='white', zorder=5)

    kept_pct = 100 - int(np.clip(edge_threshold_percentile, 0, 100))
    ax.set_title(
        f"sub-{subject} Community Network (positive edges, top {kept_pct}%; hubs={hub_metric_name})",
        fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9, markerscale=0.7)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists and return path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute community-level FC and create spring plot visualization."
    )
    parser.add_argument("--subjects", nargs="+", default=None, help="List of subject IDs to process")
    parser.add_argument("--network-label-base", default=DEFAULT_NETWORK_LABEL_BASE,
                       help="Base path for individual network outputs")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for results")
    parser.add_argument("--edge-threshold-percentile", type=int, default=90,
                       help="Percentile threshold for spring plot edges (default: 90 = top 30% edges)")
    parser.add_argument("--hub-metric", choices=["strength", "participation"], default="strength",
                       help="Metric for hub selection (currently only strength supported)")
    return parser.parse_args()


def process_subject(subject: str, args: argparse.Namespace) -> str:
    """Process single subject."""
    print(f"sub-{subject}: loading community timeseries and labels...")

    # Load data
    community_ts, community_names, source_files, community_colors, network_canonical_colors = load_community_timeseries(subject, args.network_label_base)
    n_communities = community_ts.shape[1]

    print(f"sub-{subject}: computing community FC ({n_communities} communities, {community_ts.shape[0]} timepoints)...")
    fc = compute_community_fc(community_ts)

    # Compute metrics
    pos_strength, abs_strength = compute_strength(fc)

    # Create output directory
    subject_dir = ensure_dir(Path(args.output_dir) / f"sub-{subject}" / "community")

    # Save FC matrix
    np.savez_compressed(
        subject_dir / "subject_community_fc.npz",
        fc=fc,
        community_names=np.array(community_names, dtype=object),
        n_timepoints=community_ts.shape[0],
        n_communities=n_communities,
        source_files=np.array(source_files, dtype=object),
    )

    # Save metrics
    network_names = [extract_network_from_community_name(name) for name in community_names]
    metric_df = pd.DataFrame({
        "subject": subject,
        "community_id": np.arange(len(community_names)),
        "community_name": community_names,
        "network_name": network_names,
        "strength": abs_strength,
        "positive_strength": pos_strength,
    })
    metric_df.to_csv(subject_dir / "community_hub_metrics_subject.csv", index=False)

    # Create visualization
    plot_filename = f"spring_plot_communities_edges{100 - int(np.clip(args.edge_threshold_percentile, 0, 100)):03d}_strength.png"
    print(f"sub-{subject}: creating spring plot...")
    create_community_spring_plot(
        fc,
        community_names,
        community_colors,
        subject,
        subject_dir / plot_filename,
        edge_threshold_percentile=args.edge_threshold_percentile,
        hub_metric_name=args.hub_metric,
        network_canonical_colors=network_canonical_colors,
    )

    return f"sub-{subject}: completed community FC analysis ({n_communities} communities)"


def main() -> None:
    args = parse_args()

    if args.subjects:
        subjects = args.subjects
    else:
        subjects = discover_subjects_from_network_base(args.network_label_base)

    if not subjects:
        raise ValueError("No subjects found")

    print(f"Processing {len(subjects)} subjects...")
    failures = []

    for subject in subjects:
        try:
            status = process_subject(subject, args)
            print(status)
        except Exception as exc:
            failures.append((subject, str(exc)))
            print(f"sub-{subject}: FAILED -> {exc}")

    if failures:
        print("\nCompleted with failures:")
        for sub, msg in failures:
            print(f"  sub-{sub}: {msg}")
    else:
        print("\nAll subjects completed successfully!")


if __name__ == "__main__":
    main()
