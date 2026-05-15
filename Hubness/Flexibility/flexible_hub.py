#!/usr/bin/env python3
"""Stage 2 of the flexible hub pipeline.

This script loads saved subject-level PPI outputs, computes flexibility metrics
(Cole-style variability summaries), and renders either a circular network plot
(network mode) or a spring plot (network_parcel mode).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

HUBNESS_DIR = Path(__file__).resolve().parents[1]
if str(HUBNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HUBNESS_DIR))

import matplotlib.pyplot as plt
import networkx as nx
import nibabel as nib
import numpy as np
import pandas as pd

from hubness_utils import (
    compute_participation_coefficient,
    discover_subjects_from_subdirs,
    ensure_dir,
    extract_network_colors_from_dlabel,
    infer_label_names_from_dlabel,
    is_fpn_network_name,
    load_split_parcel_manifest_retained,
    split_threshold_tag,
    subject_network_label_path,
)

DEFAULT_OUTPUT_DIR = "/ptmp/hmueller2/2025_ibc_latent/outputs/hubness"
DEFAULT_ASSIGNMENT_DIR = "/ptmp/hmueller2/2025_ibc_latent/outputs/hubness"
DEFAULT_NETWORK_LABEL_BASE = "/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks/derived_networks"

ANALYSIS_LEVEL_NETWORK = "network"
ANALYSIS_LEVEL_NETWORK_PARCEL = "network_parcel"


def color_from_name(name: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    hue = (sum(ord(ch) for ch in str(name)) % 360) / 360.0
    rgba = plt.cm.hsv(hue)
    return float(rgba[0]), float(rgba[1]), float(rgba[2]), float(alpha)


def build_plot_filename(analysis_level: str, edge_threshold_percentile: float, hub_selection_metric: str) -> str:
    kept_pct = 100 - int(np.clip(edge_threshold_percentile, 0, 100))
    return f"flexible_hub_{analysis_level}_edges{kept_pct:03d}_{hub_selection_metric}.png"


def load_stage1_results(subject: str, args: argparse.Namespace) -> pd.DataFrame:
    ppi_dir = Path(args.output_dir) / f"sub-{subject}" / "flexible" / "ppi"
    stage1_path = ppi_dir / f"flexible_hub_ppi_{args.analysis_level}_ffx.csv"
    if not stage1_path.exists():
        raise FileNotFoundError(f"Missing Stage 1 PPI output for sub-{subject}: {stage1_path}")

    df = pd.read_csv(stage1_path)
    if df.empty:
        raise ValueError(f"Stage 1 PPI file is empty for sub-{subject}: {stage1_path}")
    return df


def load_variability_state(subject: str, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    subject_dir = Path(args.output_dir) / f"sub-{subject}" / "flexible"
    state_path = subject_dir / f"flexible_hub_variability_{args.analysis_level}.npz"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing saved variability state for sub-{subject}: {state_path}")

    with np.load(state_path, allow_pickle=True) as data:
        required = {"sigma", "sigma_std", "node_names", "node_modules", "task_names"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Variability state missing keys {missing}: {state_path}")
        sigma = np.asarray(data["sigma"], dtype=float)
        sigma_std = np.asarray(data["sigma_std"], dtype=float)
        node_names = np.asarray(data["node_names"], dtype=str)
        modules = np.asarray(data["node_modules"], dtype=str)
        task_names = np.asarray(data["task_names"], dtype=str)
    return sigma, sigma_std, node_names, modules, task_names


def load_node_modules(subject: str, args: argparse.Namespace) -> pd.DataFrame:
    if args.analysis_level == ANALYSIS_LEVEL_NETWORK:
        p = Path(args.assignment_dir) / f"sub-{subject}" / "parcel_network_assignment_subject.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing network assignment file for sub-{subject}: {p}")
        df = pd.read_csv(p)
        required = {"parcel_id", "assigned_network_id", "assigned_network_name"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Assignment file missing columns {missing}: {p}")
        return (
            df[["assigned_network_id", "assigned_network_name"]]
            .drop_duplicates()
            .rename(columns={"assigned_network_id": "parcel_id", "assigned_network_name": "network_name"})
        )

    manifest = load_split_parcel_manifest_retained(subject, args.assignment_dir, args.overlap_threshold)
    if manifest.empty:
        raise ValueError(f"No retained split parcels found for sub-{subject}")
    manifest = manifest.sort_values("split_mask_path").reset_index(drop=True)
    if "network_name" not in manifest.columns:
        if "split_label" not in manifest.columns:
            raise ValueError("Split parcel manifest missing both network_name and split_label columns")
        manifest = manifest.copy()
        manifest["network_name"] = manifest["split_label"].astype(str).str.split("__").str[1].fillna(manifest["split_label"].astype(str))
    return manifest[["parcel_id", "network_name"]].copy()


def attach_modules(df: pd.DataFrame, module_map: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(module_map.rename(columns={"parcel_id": "seed_id", "network_name": "seed_module"}), on="seed_id", how="left")
    out = out.merge(module_map.rename(columns={"parcel_id": "target_id", "network_name": "target_module"}), on="target_id", how="left")
    out["seed_module"] = out["seed_module"].fillna(out["seed_name"])
    out["target_module"] = out["target_module"].fillna(out["target_name"])
    return out


def collapse_to_network_level(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    rows: list[dict[str, object]] = []
    for keys, group in df.groupby(["subject", "session", "run_id", "task", "condition", "task_condition", "seed_module", "target_module"], as_index=False):
        subject, session, run_id, task, condition, task_condition, seed_module, target_module = keys
        betas = group["beta_ffx"].to_numpy(dtype=float)
        se = group["se_ffx"].to_numpy(dtype=float)
        variances = np.square(se)
        valid = np.isfinite(betas) & np.isfinite(variances) & (variances > 0)
        betas = betas[valid]
        variances = variances[valid]
        if len(betas) == 0:
            continue
        if len(betas) == 1:
            beta_ffx = float(betas[0])
            se_ffx = float(np.sqrt(variances[0]))
            n_runs = 1
        else:
            weights = 1.0 / variances
            beta_ffx = float(np.sum(betas * weights) / np.sum(weights))
            se_ffx = float(np.sqrt(1.0 / np.sum(weights)))
            n_runs = int(len(betas))
        rows.append(
            {
                "subject": subject,
                "session": session,
                "run_id": run_id,
                "task": task,
                "condition": condition,
                "task_condition": task_condition,
                "seed_name": str(seed_module),
                "target_name": str(target_module),
                "beta_ffx": beta_ffx,
                "se_ffx": se_ffx,
                "n_runs": n_runs,
            }
        )

    return pd.DataFrame(rows)


def build_task_sigma_matrices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if df.empty:
        raise ValueError("Cannot build variability matrices from empty dataframe")

    node_names = np.array(sorted(set(df["seed_name"].astype(str)) | set(df["target_name"].astype(str))))
    node_to_idx = {name: idx for idx, name in enumerate(node_names)}
    modules = {}
    for _, row in df[["seed_name", "target_name"]].drop_duplicates().iterrows():
        pass

    module_map: dict[str, str] = {}
    for _, row in df[["seed_name", "target_name"]].drop_duplicates().iterrows():
        seed = str(row["seed_name"])
        target = str(row["target_name"])
        module_map.setdefault(seed, seed)
        module_map.setdefault(target, target)

    task_names = np.array(sorted(df["task"].astype(str).unique()))
    task_sigma = np.zeros((len(task_names), len(node_names), len(node_names)), dtype=float)

    for task_idx, task in enumerate(task_names):
        task_df = df[df["task"].astype(str) == str(task)].copy()
        pair_sigma = {}
        for (seed_name, target_name), pair_group in task_df.groupby(["seed_name", "target_name"], as_index=False):
            sigma = float(np.std(pair_group["beta_ffx"].to_numpy(dtype=float), ddof=0))
            pair_sigma[(str(seed_name), str(target_name))] = sigma

        matrix = np.zeros((len(node_names), len(node_names)), dtype=float)
        for (seed_name, target_name), sigma in pair_sigma.items():
            if seed_name not in node_to_idx or target_name not in node_to_idx:
                continue
            i = node_to_idx[seed_name]
            j = node_to_idx[target_name]
            matrix[i, j] = sigma

        # symmetrize
        matrix = 0.5 * (matrix + matrix.T)
        np.fill_diagonal(matrix, 0.0)
        task_sigma[task_idx] = matrix

    sigma_mean = np.mean(task_sigma, axis=0)
    sigma_std = np.std(task_sigma, axis=0)
    return sigma_mean, sigma_std, node_names, task_names


def load_module_names_for_nodes(subject: str, args: argparse.Namespace, node_names: np.ndarray) -> np.ndarray:
    if args.analysis_level == ANALYSIS_LEVEL_NETWORK:
        assign = pd.read_csv(Path(args.assignment_dir) / f"sub-{subject}" / "parcel_network_assignment_subject.csv")
        assign = assign[["assigned_network_id", "assigned_network_name"]].drop_duplicates().sort_values("assigned_network_id")
        mapping = assign.set_index("assigned_network_id")["assigned_network_name"].to_dict()
        return np.array([str(mapping.get(int(name), name)) if str(name).isdigit() else str(name) for name in node_names], dtype=str)

    manifest = load_split_parcel_manifest_retained(subject, args.assignment_dir, args.overlap_threshold)
    manifest = manifest.sort_values("split_mask_path").reset_index(drop=True)
    if "network_name" not in manifest.columns:
        if "split_label" not in manifest.columns:
            raise ValueError("Split parcel manifest missing network_name/split_label")
        manifest = manifest.copy()
        manifest["network_name"] = manifest["split_label"].astype(str).str.split("__").str[1].fillna(manifest["split_label"].astype(str))
    name_map = manifest.set_index("split_label" if "split_label" in manifest.columns else "parcel_id")["network_name"].to_dict()
    # In parcel mode, stage-1 names are parcel names; module names default to node names if no exact mapping exists.
    return np.array([str(name_map.get(name, name)) for name in node_names], dtype=str)


def load_infomap_color_map(subject: str, args: argparse.Namespace, node_modules: np.ndarray) -> dict[str, tuple[float, float, float, float]]:
    network_path = subject_network_label_path(args.network_label_base, subject)
    if not network_path.exists():
        return {}

    img = nib.load(str(network_path))
    rgba_by_id = extract_network_colors_from_dlabel(img)
    if not rgba_by_id:
        return {}

    names_by_id = infer_label_names_from_dlabel(img)
    lookup: dict[str, tuple[float, float, float, float]] = {}
    for network_id, network_name in names_by_id.items():
        if network_id in rgba_by_id:
            lookup[str(network_name)] = rgba_by_id[network_id]
    return {name: lookup[name] for name in node_modules if name in lookup}


def summarize_variability(sigma: np.ndarray, node_names: np.ndarray, modules: np.ndarray, subject: str) -> pd.DataFrame:
    gvc = np.mean(sigma, axis=1)
    variability_strength = np.sum(sigma, axis=1)
    variability_participation = compute_participation_coefficient(sigma, modules)

    out = pd.DataFrame(
        {
            "subject": subject,
            "node_name": node_names.astype(str),
            "module_name": modules.astype(str),
            "gvc": gvc,
            "variability_strength": variability_strength,
            "variability_participation": variability_participation,
            "is_fpn": [int(is_fpn_network_name(str(name))) for name in modules],
        }
    )
    out["gvc_rank_desc"] = out["gvc"].rank(ascending=False, method="dense").astype(int)
    out["variability_strength_rank_desc"] = out["variability_strength"].rank(ascending=False, method="dense").astype(int)
    return out


def choose_center_index(sigma: np.ndarray, node_names: np.ndarray, modules: np.ndarray) -> int:
    for idx, module_name in enumerate(modules.astype(str)):
        if is_fpn_network_name(module_name):
            return idx
    return int(np.argmax(np.mean(sigma, axis=1)))


def plot_circular(
    sigma: np.ndarray,
    node_names: np.ndarray,
    modules: np.ndarray,
    subject: str,
    output_path: Path,
    edge_threshold_percentile: float,
    hub_selection_metric: str,
    color_map: dict[str, tuple[float, float, float, float]],
) -> None:
    node_metric = np.mean(sigma, axis=1) if hub_selection_metric == "gvc" else compute_participation_coefficient(sigma, modules)
    sizes_rank = np.argsort(np.argsort(node_metric)).astype(float)
    if len(node_metric) > 1:
        sizes_rank = sizes_rank / float(len(node_metric) - 1)
    node_sizes = 380.0 + 860.0 * (sizes_rank ** 1.8)

    center_idx = choose_center_index(sigma, node_names, modules)
    pos: dict[int, tuple[float, float]] = {center_idx: (0.0, 0.0)}
    ring_indices = [i for i in range(len(node_names)) if i != center_idx]
    for rank, idx in enumerate(ring_indices):
        angle = 2.0 * np.pi * float(rank) / float(max(1, len(ring_indices)))
        pos[idx] = (5.0 * np.cos(angle), 5.0 * np.sin(angle))

    tri = np.triu_indices(len(node_names), k=1)
    edge_vals = sigma[tri]
    edge_vals = edge_vals[np.isfinite(edge_vals)]
    threshold = np.percentile(edge_vals, float(np.clip(edge_threshold_percentile, 0, 100))) if edge_vals.size else np.inf
    max_abs = float(np.max(edge_vals)) if edge_vals.size else 1.0
    min_abs = float(np.min(edge_vals)) if edge_vals.size else 0.0
    denom = max(max_abs - min_abs, 1e-9)

    fig, ax = plt.subplots(figsize=(12, 12))
    kept = 0
    for i, j in zip(*tri):
        value = float(sigma[i, j])
        if not np.isfinite(value) or value < threshold:
            continue
        kept += 1
        norm = (value - min_abs) / denom
        line_w = 0.5 + 5.5 * (norm ** 1.5)
        alpha = 0.12 + 0.78 * (norm ** 1.1)
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        ax.plot([x0, x1], [y0, y1], color="#3d3d3d", alpha=alpha, linewidth=line_w, zorder=1)

    for i, node in enumerate(node_names):
        module_name = str(modules[i])
        rgba = color_map.get(module_name, color_from_name(module_name))
        x, y = pos[i]
        ax.scatter([x], [y], s=[node_sizes[i] * (1.15 if i == center_idx else 1.0)], c=[rgba], edgecolors="black", linewidth=2.0 if i == center_idx else 1.0, zorder=3)
        ax.text(x, y, str(node), fontsize=8, color="white", ha="center", va="center", fontweight="bold", zorder=4)

    ax.set_title(
        f"sub-{subject} Flexible Hub Variability Circular Plot (top {100 - int(np.clip(edge_threshold_percentile, 0, 100))}% edges; center={node_names[center_idx]})",
        fontsize=12,
        fontweight="bold",
    )
    ax.axis("off")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_spring(
    sigma: np.ndarray,
    node_names: np.ndarray,
    modules: np.ndarray,
    subject: str,
    output_path: Path,
    edge_threshold_percentile: float,
    hub_selection_metric: str,
    color_map: dict[str, tuple[float, float, float, float]],
    spring_k: float,
    spring_iterations: int,
    spring_scale: float,
    max_labels: int,
) -> None:
    node_metric = np.mean(sigma, axis=1) if hub_selection_metric == "gvc" else compute_participation_coefficient(sigma, modules)
    sizes_rank = np.argsort(np.argsort(node_metric)).astype(float)
    if len(node_metric) > 1:
        sizes_rank = sizes_rank / float(len(node_metric) - 1)
    node_sizes = 28.0 + 260.0 * (sizes_rank ** 1.7)

    G = nx.Graph()
    for idx, node in enumerate(node_names):
        G.add_node(idx, name=str(node), module=str(modules[idx]), size=float(node_sizes[idx]))

    tri = np.triu_indices(len(node_names), k=1)
    edge_vals = sigma[tri]
    edge_vals = edge_vals[np.isfinite(edge_vals)]
    threshold = np.percentile(edge_vals, float(np.clip(edge_threshold_percentile, 0, 100))) if edge_vals.size else np.inf
    edge_span = max(float(np.max(edge_vals)) - float(threshold), 1e-9) if edge_vals.size else 1.0
    kept = 0
    for i, j in zip(*tri):
        value = float(sigma[i, j])
        if not np.isfinite(value) or value < threshold:
            continue
        kept += 1
        G.add_edge(i, j, weight=value)

    pos = nx.spring_layout(
        G,
        seed=42,
        k=float(spring_k),
        iterations=int(spring_iterations),
        weight=None,
        scale=float(spring_scale),
        center=(0.0, 0.0),
    )
    fig, ax = plt.subplots(figsize=(16, 14))

    for i, j, data in G.edges(data=True):
        value = float(data.get("weight", 0.0))
        edge_norm = np.clip((value - float(threshold)) / edge_span, 0.0, 1.0)
        width = 0.7 + 4.8 * (edge_norm ** 1.2)
        alpha = 0.20 + 0.70 * (edge_norm ** 0.9)
        edge_color = plt.cm.magma(0.25 + 0.75 * edge_norm)
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        ax.plot([x0, x1], [y0, y1], color=edge_color, alpha=alpha, linewidth=width, zorder=1)

    hub_rank = np.argsort(node_metric)[-min(max_labels, len(node_metric)) :][::-1]
    hub_set = set(int(i) for i in hub_rank)
    for idx, node in enumerate(node_names):
        module_name = str(modules[idx])
        rgba = color_map.get(module_name, color_from_name(module_name))
        x, y = pos[idx]
        ax.scatter([x], [y], s=[node_sizes[idx] * (1.55 if idx in hub_set else 1.05)], c=[rgba], edgecolors="black", linewidth=2.2 if idx in hub_set else 1.0, zorder=3)
        if idx in hub_set:
            ax.text(x, y, str(node)[:16], fontsize=6, color="white", ha="center", va="center", fontweight="bold", zorder=4)

    ax.set_title(
        f"sub-{subject} Flexible Hub Variability Spring Plot (top {100 - int(np.clip(edge_threshold_percentile, 0, 100))}% edges; edges kept={kept})",
        fontsize=12,
        fontweight="bold",
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: compute flexible hub metrics and plots from saved PPI outputs.")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--analysis-level", choices=[ANALYSIS_LEVEL_NETWORK, ANALYSIS_LEVEL_NETWORK_PARCEL], default=ANALYSIS_LEVEL_NETWORK)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--assignment-dir", default=DEFAULT_ASSIGNMENT_DIR)
    parser.add_argument("--network-label-base", default=DEFAULT_NETWORK_LABEL_BASE)
    parser.add_argument("--overlap-threshold", type=float, default=0.30)
    parser.add_argument("--edge-threshold-percentile", type=float, default=80.0)
    parser.add_argument("--hub-selection-metric", choices=["gvc", "participation"], default="gvc")
    parser.add_argument("--plot-only", action="store_true", help="Only regenerate plots from the saved variability .npz files.")
    parser.add_argument("--spring-k", type=float, default=3.0, help="Spring-layout repulsion strength; larger values spread nodes further apart.")
    parser.add_argument("--spring-iterations", type=int, default=400, help="Number of iterations for the spring layout solver.")
    parser.add_argument("--spring-scale", type=float, default=5.0, help="Final scale of the spring layout coordinates.")
    parser.add_argument("--spring-max-labels", type=int, default=15, help="Number of highest-ranked nodes to label in the spring plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    if args.subjects:
        subjects = [str(s).replace("sub-", "") for s in args.subjects]
    else:
        subjects = discover_subjects_from_subdirs(args.assignment_dir)

    if not subjects:
        raise ValueError("No subjects found")

    failures: list[tuple[str, str]] = []
    for subject in subjects:
        try:
            subject_dir = ensure_dir(output_dir / f"sub-{subject}" / "flexible")
            if args.plot_only:
                sigma, sigma_std, node_names, modules, task_names = load_variability_state(subject, args)
            else:
                stage1 = load_stage1_results(subject, args)
                module_map = load_node_modules(subject, args)
                stage1 = attach_modules(stage1, module_map)

                if stage1.empty:
                    raise ValueError(f"No usable PPI rows for sub-{subject}")

                sigma, sigma_std, node_names, task_names = build_task_sigma_matrices(stage1)
                modules = load_module_names_for_nodes(subject, args, node_names)
                metrics = summarize_variability(sigma, node_names, modules, subject)

                np.savez_compressed(
                    subject_dir / f"flexible_hub_variability_{args.analysis_level}.npz",
                    sigma=sigma,
                    sigma_std=sigma_std,
                    node_names=node_names,
                    node_modules=modules,
                    task_names=task_names,
                    analysis_level=args.analysis_level,
                    subject=subject,
                )
                metrics.to_csv(subject_dir / f"flexible_hub_metrics_{args.analysis_level}.csv", index=False)

            color_map = load_infomap_color_map(subject, args, modules)
            if args.analysis_level == ANALYSIS_LEVEL_NETWORK:
                plot_circular(
                    sigma=sigma,
                    node_names=node_names,
                    modules=modules,
                    subject=subject,
                    output_path=subject_dir / build_plot_filename(args.analysis_level, args.edge_threshold_percentile, args.hub_selection_metric),
                    edge_threshold_percentile=args.edge_threshold_percentile,
                    hub_selection_metric=args.hub_selection_metric,
                    color_map=color_map,
                )
            else:
                plot_spring(
                    sigma=sigma,
                    node_names=node_names,
                    modules=modules,
                    subject=subject,
                    output_path=subject_dir / build_plot_filename(args.analysis_level, args.edge_threshold_percentile, args.hub_selection_metric),
                    edge_threshold_percentile=args.edge_threshold_percentile,
                    hub_selection_metric=args.hub_selection_metric,
                    color_map=color_map,
                    spring_k=args.spring_k,
                    spring_iterations=args.spring_iterations,
                    spring_scale=args.spring_scale,
                    max_labels=args.spring_max_labels,
                )

            if args.plot_only:
                print(f"sub-{subject}: rewrote flexible plot only ({args.analysis_level})")
            else:
                print(f"sub-{subject}: wrote flexible variability outputs ({args.analysis_level})")
        except Exception as exc:
            failures.append((subject, str(exc)))
            print(f"sub-{subject}: FAILED -> {exc}")

    if failures:
        print("\nCompleted with failures:")
        for sub, msg in failures:
            print(f"  sub-{sub}: {msg}")


if __name__ == "__main__":
    main()
