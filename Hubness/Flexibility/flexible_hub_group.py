#!/usr/bin/env python3
"""
Group aggregation for the flexible hub pipeline.
Loads subject-level variability matrices saved by flexible_hub.py, aligns them
by node name, averages them, and writes group-level summaries and plots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HUBNESS_DIR = Path(__file__).resolve().parents[1]
if str(HUBNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HUBNESS_DIR))

import numpy as np
import pandas as pd

from hubness_utils import discover_subjects_from_subdirs, ensure_dir
from flexible_hub import (
    ANALYSIS_LEVEL_NETWORK,
    ANALYSIS_LEVEL_NETWORK_PARCEL,
    build_plot_filename,
    load_infomap_color_map,
    plot_circular,
    plot_spring,
    summarize_variability,
)

DEFAULT_OUTPUT_DIR = "/ptmp/hmueller2/2025_ibc_latent/outputs/hubness"
DEFAULT_ASSIGNMENT_DIR = "/ptmp/hmueller2/2025_ibc_latent/outputs/hubness"
DEFAULT_NETWORK_LABEL_BASE = "/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks/derived_networks"


def load_subject_variability(subject: str, output_dir: str, analysis_level: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = Path(output_dir) / f"sub-{subject}" / "flexible" / f"flexible_hub_variability_{analysis_level}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing subject variability file: {p}")
    with np.load(p, allow_pickle=False) as data:
        sigma = np.asarray(data["sigma"], dtype=float)
        node_names = np.asarray(data["node_names"]).astype(str)
        node_modules = np.asarray(data["node_modules"]).astype(str)
        task_names = np.asarray(data["task_names"]).astype(str)
    return sigma, node_names, node_modules, task_names


def align_to_canonical(
    sigma: np.ndarray,
    node_names: np.ndarray,
    node_modules: np.ndarray,
    canonical_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    aligned = np.zeros((len(canonical_names), len(canonical_names)), dtype=float)
    aligned_modules = canonical_names.astype(str).copy()
    name_to_idx = {str(name): idx for idx, name in enumerate(node_names.astype(str))}

    for i, name_i in enumerate(canonical_names):
        if str(name_i) not in name_to_idx:
            continue
        ii = name_to_idx[str(name_i)]
        for j, name_j in enumerate(canonical_names):
            if str(name_j) not in name_to_idx:
                continue
            jj = name_to_idx[str(name_j)]
            aligned[i, j] = float(sigma[ii, jj])

    # Preserve first seen module names where possible.
    for i, name in enumerate(canonical_names):
        if str(name) in name_to_idx:
            aligned_modules[i] = str(node_modules[name_to_idx[str(name)]])
    return aligned, aligned_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group aggregation for flexible hub variability outputs.")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--analysis-level", choices=[ANALYSIS_LEVEL_NETWORK, ANALYSIS_LEVEL_NETWORK_PARCEL], default=ANALYSIS_LEVEL_NETWORK)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--assignment-dir", default=DEFAULT_ASSIGNMENT_DIR)
    parser.add_argument("--network-label-base", default=DEFAULT_NETWORK_LABEL_BASE)
    parser.add_argument("--edge-threshold-percentile", type=int, default=80)
    parser.add_argument("--hub-selection-metric", choices=["gvc", "participation"], default="gvc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    if args.subjects:
        subjects = [str(s).replace("sub-", "") for s in args.subjects]
    else:
        subjects = discover_subjects_from_subdirs(args.output_dir)

    if not subjects:
        raise ValueError("No subjects found")

    loaded: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for subject in subjects:
        sigma, node_names, node_modules, task_names = load_subject_variability(subject, args.output_dir, args.analysis_level)
        loaded.append((subject, sigma, node_names, node_modules, task_names))

    if not loaded:
        raise ValueError("No subject variability outputs found")

    canonical_names = np.array(sorted(set().union(*[set(x[2].astype(str)) for x in loaded])))
    aligned = []
    aligned_modules = None
    task_name_sets = []
    for _, sigma, node_names, node_modules, task_names in loaded:
        sigma_aligned, modules_aligned = align_to_canonical(sigma, node_names, node_modules, canonical_names)
        aligned.append(sigma_aligned)
        if aligned_modules is None:
            aligned_modules = modules_aligned
        task_name_sets.append(task_names.astype(str))

    group_sigma = np.mean(np.stack(aligned, axis=0), axis=0)
    group_sigma_std = np.std(np.stack(aligned, axis=0), axis=0)
    if aligned_modules is None:
        aligned_modules = canonical_names.astype(str)

    group_dir = ensure_dir(output_dir / "group" / "flexible")
    np.savez_compressed(
        group_dir / f"flexible_hub_group_variability_{args.analysis_level}.npz",
        sigma=group_sigma,
        sigma_std=group_sigma_std,
        node_names=canonical_names,
        node_modules=aligned_modules,
        n_subjects=len(loaded),
        subjects=np.asarray(subjects, dtype=str),
        analysis_level=args.analysis_level,
    )

    summary = summarize_variability(group_sigma, canonical_names, aligned_modules, subject="group")
    summary.to_csv(group_dir / f"flexible_hub_group_metrics_{args.analysis_level}.csv", index=False)

    if args.analysis_level == ANALYSIS_LEVEL_NETWORK:
        # Reuse the circular plotting path from the subject-level script.
        # Colors are loaded from the first available subject.
        color_map = load_infomap_color_map(
            loaded[0][0],
            argparse.Namespace(
                network_label_base=args.network_label_base,
                analysis_level=args.analysis_level,
                assignment_dir=args.assignment_dir,
                overlap_threshold=0.30,
            ),
            aligned_modules,
        )
        plot_circular(
            sigma=group_sigma,
            node_names=canonical_names,
            modules=aligned_modules,
            subject="group",
            output_path=group_dir / build_plot_filename(args.analysis_level, args.edge_threshold_percentile, args.hub_selection_metric),
            edge_threshold_percentile=args.edge_threshold_percentile,
            hub_selection_metric=args.hub_selection_metric,
            color_map=color_map,
        )
    else:
        plot_spring(
            sigma=group_sigma,
            node_names=canonical_names,
            modules=aligned_modules,
            subject="group",
            output_path=group_dir / build_plot_filename(args.analysis_level, args.edge_threshold_percentile, args.hub_selection_metric),
            edge_threshold_percentile=args.edge_threshold_percentile,
            hub_selection_metric=args.hub_selection_metric,
            color_map={name: (0.5, 0.5, 0.5, 1.0) for name in np.unique(aligned_modules.astype(str))},
        )

    print(f"group: wrote flexible hub variability outputs for {len(loaded)} subjects")


if __name__ == "__main__":
    main()
