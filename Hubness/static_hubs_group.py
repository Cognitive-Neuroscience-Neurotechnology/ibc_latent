#!/usr/bin/env python3
"""Aggregate subject-level network FC matrices and render a group circular plot.

Expected inputs per subject:
- sub-XX/static/subject_fc_network_collapsed.npz

Outputs:
- group/static/group_fc_network_collapsed.npz
- group/static/group_network_strength_summary.csv
- group/static/circular_plot_group_network_edgesXXX_<metric>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from hubness_utils import ensure_dir
from hubness_utils import extract_network_colors_from_dlabel
from static_hubs import build_circular_plot_filename, create_circular_network_plot
import nibabel as nib

DEFAULT_NETWORK_LABEL_BASE = "/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks/derived_networks"
DEFAULT_OUTPUT_DIR = "/ptmp/hmueller2/2025_ibc_latent/outputs/hubness"

def load_group_infomap_colors(output_dir: Path, subjects: List[str], network_label_base: str = DEFAULT_NETWORK_LABEL_BASE) -> dict[str, tuple[float, float, float, float]]:
    """Load infomap colors from the first available subject's network dlabel.
    
    Returns a mapping of network_name -> (R, G, B, A) with values normalized to [0, 1].
    """
    from hubness_utils import subject_network_label_path
    
    color_map: dict[str, tuple[float, float, float, float]] = {}
    
    for subject in subjects:
        network_path = subject_network_label_path(network_label_base, subject)
        if not network_path.exists():
            continue
        
        try:
            img = nib.load(str(network_path))
            rgba_by_id = extract_network_colors_from_dlabel(img)
            if not rgba_by_id:
                continue
            
            # Load the network names from one of the subject FC files
            static_path = output_dir / f"sub-{subject}" / "static" / "subject_fc_network_collapsed.npz"
            legacy_path = output_dir / f"sub-{subject}" / "subject_fc_network_collapsed.npz"
            fc_path = static_path if static_path.exists() else legacy_path
            if not fc_path.exists():
                continue
            
            with np.load(fc_path, allow_pickle=False) as data:
                if "network_names" not in data:
                    continue
                network_names = np.asarray(data["network_names"]).astype(str)
            
            # Map network names to colors by checking the network dlabel
            from hubness_utils import infer_label_names_from_dlabel
            net_names = infer_label_names_from_dlabel(img)
            
            for net_id, net_name in net_names.items():
                if net_id in rgba_by_id and net_name in network_names:
                    color_map[net_name] = rgba_by_id[net_id]
            
            if color_map:
                return color_map
        except Exception:
            continue
    
    return color_map



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create group-mean network FC and circular plot from subject outputs.")
    parser.add_argument("--subjects", nargs="+", default=None, help="Optional subject IDs (without sub- prefix).")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Hubness output root containing sub-XX/static folders.")
    parser.add_argument(
        "--edge-threshold-percentile",
        type=int,
        default=95,
        help="Percentile threshold for plotting edges (top X%% by |FC| retained).",
    )
    parser.add_argument(
        "--hub-selection-metric",
        choices=["strength", "participation"],
        default="strength",
        help="Node-size metric for circular plot.",
    )
    return parser.parse_args()


def discover_subjects_from_collapsed_fc(output_dir: Path) -> List[str]:
    subjects: List[str] = []
    for subject_dir in sorted(output_dir.glob("sub-*")):
        static_path = subject_dir / "static" / "subject_fc_network_collapsed.npz"
        legacy_path = subject_dir / "subject_fc_network_collapsed.npz"
        if subject_dir.is_dir() and (static_path.exists() or legacy_path.exists()):
            subjects.append(subject_dir.name.replace("sub-", ""))
    return subjects


def load_subject_collapsed_fc(output_dir: Path, subject: str) -> Tuple[np.ndarray, np.ndarray]:
    static_path = output_dir / f"sub-{subject}" / "static" / "subject_fc_network_collapsed.npz"
    legacy_path = output_dir / f"sub-{subject}" / "subject_fc_network_collapsed.npz"
    fc_path = static_path if static_path.exists() else legacy_path
    if not fc_path.exists():
        raise FileNotFoundError(f"Missing collapsed network FC for sub-{subject}: {fc_path}")

    with np.load(fc_path, allow_pickle=False) as data:
        if "fc" not in data or "network_names" not in data:
            raise ValueError(f"Missing required arrays in {fc_path}; expected 'fc' and 'network_names'")
        fc = np.asarray(data["fc"], dtype=float)
        network_names = np.asarray(data["network_names"]).astype(str)

    if fc.ndim != 2 or fc.shape[0] != fc.shape[1]:
        raise ValueError(f"Collapsed FC must be square for sub-{subject}, got shape={fc.shape}")
    if fc.shape[0] != len(network_names):
        raise ValueError(
            f"Network name length mismatch for sub-{subject}: matrix={fc.shape[0]}, names={len(network_names)}"
        )

    return fc, network_names


def align_subject_fc_to_canonical(
    fc: np.ndarray,
    subject_names: np.ndarray,
    canonical_names: np.ndarray,
) -> np.ndarray:
    """Reorder subject FC matrix to match canonical network ordering, padding with zeros if needed."""
    if len(subject_names) == len(canonical_names) and np.array_equal(subject_names, canonical_names):
        return fc
    
    # Create mapping from subject to canonical indices
    n_canonical = len(canonical_names)
    aligned_fc = np.zeros((n_canonical, n_canonical), dtype=float)
    
    for canon_idx, canon_name in enumerate(canonical_names):
        subject_idx_list = np.where(subject_names == canon_name)[0]
        if len(subject_idx_list) == 0:
            continue  # Network not present in this subject; leave as zeros
        subject_idx = int(subject_idx_list[0])
        
        for canon_idx_j, canon_name_j in enumerate(canonical_names):
            subject_idx_j_list = np.where(subject_names == canon_name_j)[0]
            if len(subject_idx_j_list) == 0:
                continue  # Network not present in this subject; leave as zeros
            subject_idx_j = int(subject_idx_j_list[0])
            aligned_fc[canon_idx, canon_idx_j] = fc[subject_idx, subject_idx_j]
    
    return aligned_fc


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.subjects:
        subjects = [str(s).replace("sub-", "") for s in args.subjects]
    else:
        subjects = discover_subjects_from_collapsed_fc(output_dir)

    if not subjects:
        raise ValueError(
            "No subjects found with subject_fc_network_collapsed.npz. "
            "Run static_hubs.py with --analysis-level network --save-network-fc first."
        )

    # First pass: collect all unique network names across all subjects to create canonical ordering
    all_network_names_set = set()
    subject_data: dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    
    for subject in subjects:
        fc, names = load_subject_collapsed_fc(output_dir, subject)
        subject_data[subject] = (fc, names)
        all_network_names_set.update(names)
    
    # Sort canonical names for consistent ordering
    canonical_names = np.array(sorted(list(all_network_names_set)))
    
    # Second pass: align all subject FC matrices to canonical ordering
    matrices: List[np.ndarray] = []
    for subject in subjects:
        fc, names = subject_data[subject]
        aligned_fc = align_subject_fc_to_canonical(fc, names, canonical_names)
        matrices.append(aligned_fc)

    group_fc = np.mean(np.stack(matrices, axis=0), axis=0)
    group_std = np.std(np.stack(matrices, axis=0), axis=0)
    np.fill_diagonal(group_fc, 0.0)

    group_dir = ensure_dir(output_dir / "group" / "static")
    np.savez_compressed(
        group_dir / "group_fc_network_collapsed.npz",
        fc=group_fc,
        std=group_std,
        network_names=canonical_names,
        n_subjects=len(subjects),
        subjects=np.asarray(subjects, dtype=str),
    )

    abs_strength = np.sum(np.abs(group_fc), axis=1)
    summary = pd.DataFrame(
        {
            "network_name": canonical_names,
            "abs_strength": abs_strength,
            "rank_desc": pd.Series(abs_strength).rank(ascending=False, method="dense").astype(int),
        }
    ).sort_values("abs_strength", ascending=False)
    summary.to_csv(group_dir / "group_network_strength_summary.csv", index=False)

    plot_name = build_circular_plot_filename(
        analysis_level="group_network",
        edge_threshold_percentile=args.edge_threshold_percentile,
        hub_selection_metric=args.hub_selection_metric,
    )
    create_circular_network_plot(
        network_fc=group_fc,
        network_names=canonical_names,
        title_label=f"group mean (n={len(subjects)})",
        output_path=group_dir / plot_name,
        edge_threshold_percentile=args.edge_threshold_percentile,
        hub_selection_metric=args.hub_selection_metric,
        infomap_color_map=load_group_infomap_colors(output_dir, subjects),
    )

    print(f"group: completed mean FC and circular plot for {len(subjects)} subjects")


if __name__ == "__main__":
    main()
