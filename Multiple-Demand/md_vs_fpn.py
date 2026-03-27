"""
Compare MD maps against subject-specific FPN_A and FPN_B subnetworks.

Approach A:
    Threshold the MD map and quantify overlap with FPN_A, FPN_B, and FPN_A|B.

Approach B:
    Summarize raw MD values inside FPN_A, FPN_B, and FPN_A|B.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import nibabel as nib
import numpy as np


DEFAULT_MD_DIR = "/ptmp/hmueller2/2025_ibc_latent/outputs/md_system/vertex_wise"
DEFAULT_SUBNETWORK_DIR = "/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_derivation/infomap"
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_MD_DIR, "md_vs_fpn")


def load_cifti_vector(path: str) -> tuple[np.ndarray, nib.Cifti2Image]:
    img = nib.load(path)
    data = img.get_fdata()
    if data.ndim == 2 and data.shape[0] == 1:
        return data[0], img
    if data.ndim == 1:
        return data, img
    raise ValueError(f"Expected a single-map CIFTI file, got shape {data.shape} for {path}")


def load_cifti_row(path: str, row_index: int) -> tuple[np.ndarray, nib.Cifti2Image]:
    img = nib.load(path)
    data = img.get_fdata()
    if data.ndim == 1:
        if row_index != 0:
            raise ValueError(f"Requested row {row_index} from 1D CIFTI data in {path}")
        return data, img
    if row_index < 0 or row_index >= data.shape[0]:
        raise ValueError(f"Row index {row_index} out of range for {path} with shape {data.shape}")
    return data[row_index], img


def canonicalize_label(name: str) -> str:
    return name.lower().replace("_", "").replace(" ", "").replace("-", "")


def infer_label_ids(img: nib.Cifti2Image, row_index: int, explicit_a: int | None, explicit_b: int | None) -> tuple[int, int]:
    if explicit_a is not None and explicit_b is not None:
        return explicit_a, explicit_b

    label_a = explicit_a
    label_b = explicit_b

    try:
        label_axis = img.header.get_axis(0)
        label_table = label_axis.label[row_index]
    except Exception:
        label_table = {}

    for key, value in label_table.items():
        name = value[0] if isinstance(value, tuple) and value else str(value)
        normalized = canonicalize_label(name)
        if label_a is None and normalized in {"fpna", "fpnsubnetworka", "network1"}:
            label_a = int(key)
        if label_b is None and normalized in {"fpnb", "fpnsubnetworkb", "network2"}:
            label_b = int(key)

    if label_a is None:
        label_a = 1
    if label_b is None:
        label_b = 2
    return label_a, label_b


def resolve_subjects(md_dir: str, subnetwork_dir: str, requested_subjects: list[str] | None) -> list[str]:
    if requested_subjects:
        return requested_subjects

    available = []
    for subject_dir in sorted(Path(subnetwork_dir).glob("sub-*")):
        subject = subject_dir.name.replace("sub-", "")
        md_map = Path(md_dir) / f"sub-{subject}" / f"sub-{subject}_MD_mean.dscalar.nii"
        subnet_map = subject_dir / f"{subject}_FPN_infomap_communities_kmeans_relabeled.dlabel.nii"
        if md_map.exists() and subnet_map.exists():
            available.append(subject)
    return available


def subject_md_map_path(md_dir: str, subject: str) -> str:
    return os.path.join(md_dir, f"sub-{subject}", f"sub-{subject}_MD_mean.dscalar.nii")


def subject_subnetwork_path(subnetwork_dir: str, subject: str) -> str:
    base = os.path.join(subnetwork_dir, f"sub-{subject}")
    preferred = os.path.join(base, f"{subject}_FPN_infomap_communities_kmeans_relabeled.dlabel.nii")
    fallback = os.path.join(base, f"{subject}_FPN_infomap_communities_kmeans_relabeled.dscalar.nii")
    if os.path.exists(preferred):
        return preferred
    return fallback


def group_md_map_path(md_dir: str) -> str:
    return os.path.join(md_dir, "group", "group_MD_mean.dscalar.nii")


def threshold_md_map(md_data: np.ndarray, threshold_z: float | None, threshold_percent: float | None) -> tuple[np.ndarray, str, float]:
    if threshold_z is not None and threshold_percent is not None:
        raise ValueError("Use either threshold_z or threshold_percent, not both.")

    if threshold_percent is not None:
        if threshold_percent <= 0 or threshold_percent > 100:
            raise ValueError(f"threshold_percent must be in (0, 100], got {threshold_percent}")
        cutoff = float(np.percentile(md_data, 100.0 - threshold_percent))
        return md_data >= cutoff, f"top{threshold_percent:g}pct", cutoff

    if threshold_z is not None:
        return md_data >= threshold_z, f"z{threshold_z:g}", float(threshold_z)

    cutoff = 0.0
    return md_data > cutoff, "positive", cutoff


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def overlap_stats(md_mask: np.ndarray, region_mask: np.ndarray) -> dict[str, float]:
    n_md = int(md_mask.sum())
    n_region = int(region_mask.sum())
    n_overlap = int((md_mask & region_mask).sum())
    union = int((md_mask | region_mask).sum())
    dice_denom = n_md + n_region
    return {
        "region_vertices": n_region,
        "overlap_vertices": n_overlap,
        "pct_md_in_region": safe_ratio(n_overlap, n_md),
        "pct_region_covered_by_md": safe_ratio(n_overlap, n_region),
        "dice": safe_ratio(2 * n_overlap, dice_denom),
        "jaccard": safe_ratio(n_overlap, union),
    }


def build_overlap_visualization_maps(
    md_data: np.ndarray,
    fpna_mask: np.ndarray,
    fpnb_mask: np.ndarray,
    threshold_z: float | None,
    threshold_percent: float | None,
) -> tuple[np.ndarray, list[str], str, float]:
    md_mask, threshold_mode, threshold_value = threshold_md_map(md_data, threshold_z, threshold_percent)
    fpnab_mask = fpna_mask | fpnb_mask
    overlap_fpna = md_mask & fpna_mask
    overlap_fpnb = md_mask & fpnb_mask
    overlap_fpnab = md_mask & fpnab_mask

    map_names = [
        "md_thresholded",
        "fpn_a_mask",
        "fpn_b_mask",
        "fpn_ab_mask",
        "overlap_md_fpn_a",
        "overlap_md_fpn_b",
        "overlap_md_fpn_ab",
    ]
    maps_2d = np.stack(
        [
            md_mask.astype(np.float32),
            fpna_mask.astype(np.float32),
            fpnb_mask.astype(np.float32),
            fpnab_mask.astype(np.float32),
            overlap_fpna.astype(np.float32),
            overlap_fpnb.astype(np.float32),
            overlap_fpnab.astype(np.float32),
        ],
        axis=0,
    )
    return maps_2d, map_names, threshold_mode, threshold_value


def build_masked_md_value_maps(
    md_data: np.ndarray,
    fpna_mask: np.ndarray,
    fpnb_mask: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    fpnab_mask = fpna_mask | fpnb_mask
    map_names = [
        "md_in_fpn_a",
        "md_in_fpn_b",
        "md_in_fpn_ab",
    ]
    maps_2d = np.stack(
        [
            np.where(fpna_mask, md_data, 0.0).astype(np.float32),
            np.where(fpnb_mask, md_data, 0.0).astype(np.float32),
            np.where(fpnab_mask, md_data, 0.0).astype(np.float32),
        ],
        axis=0,
    )
    return maps_2d, map_names


def save_dscalar_maps(path: str, maps_2d: np.ndarray, map_names: list[str], template_img: nib.Cifti2Image) -> None:
    if maps_2d.ndim != 2:
        raise ValueError(f"Expected 2D array for dscalar maps, got shape {maps_2d.shape}")
    if maps_2d.shape[0] != len(map_names):
        raise ValueError(
            f"Number of maps ({maps_2d.shape[0]}) does not match map names ({len(map_names)})"
        )

    brain_axis = template_img.header.get_axis(1)
    scalar_axis = nib.cifti2.ScalarAxis(map_names)
    header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis))
    out_img = nib.Cifti2Image(maps_2d.astype(np.float32), header=header)
    nib.save(out_img, path)


def sanitize_token(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value)


def value_stats(md_data: np.ndarray, region_mask: np.ndarray) -> dict[str, float]:
    region_values = md_data[region_mask]
    if region_values.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(region_values)),
        "median": float(np.median(region_values)),
        "std": float(np.std(region_values)),
        "max": float(np.max(region_values)),
    }


def analyze_single_map(
    analysis_level: str,
    map_subject: str,
    mask_subject: str,
    md_data: np.ndarray,
    fpna_mask: np.ndarray,
    fpnb_mask: np.ndarray,
    threshold_z: float | None,
    threshold_percent: float | None,
) -> dict[str, float | str]:
    md_mask, threshold_mode, threshold_value = threshold_md_map(md_data, threshold_z, threshold_percent)
    fpnab_mask = fpna_mask | fpnb_mask

    fpna_values = value_stats(md_data, fpna_mask)
    fpnb_values = value_stats(md_data, fpnb_mask)
    fpnab_values = value_stats(md_data, fpnab_mask)

    result: dict[str, float | str] = {
        "analysis_level": analysis_level,
        "map_subject": map_subject,
        "mask_subject": mask_subject,
        "threshold_mode": threshold_mode,
        "threshold_value": threshold_value,
        "md_vertices_thresholded": int(md_mask.sum()),
        "fpna_vertices": int(fpna_mask.sum()),
        "fpnb_vertices": int(fpnb_mask.sum()),
        "fpnab_vertices": int(fpnab_mask.sum()),
        "mean_md_fpna": fpna_values["mean"],
        "mean_md_fpnb": fpnb_values["mean"],
        "mean_md_fpnab": fpnab_values["mean"],
        "median_md_fpna": fpna_values["median"],
        "median_md_fpnb": fpnb_values["median"],
        "mean_diff_fpna_minus_fpnb": float(fpna_values["mean"] - fpnb_values["mean"]),
    }

    for prefix, mask in (("fpna", fpna_mask), ("fpnb", fpnb_mask), ("fpnab", fpnab_mask)):
        for key, value in overlap_stats(md_mask, mask).items():
            result[f"{prefix}_{key}"] = value

    return result


def write_csv(path: str, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return

    # Build a stable superset of keys so heterogeneous rows can be written safely.
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_rows(rows: list[dict[str, float | str]], label: str) -> dict[str, float | str]:
    if not rows:
        return {"summary_level": label, "n_rows": 0}

    summary: dict[str, float | str] = {
        "summary_level": label,
        "n_rows": len(rows),
    }
    numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float, np.floating))]
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if isinstance(row[key], (int, float, np.floating)) and not np.isnan(float(row[key]))]
        if values:
            summary[f"mean_{key}"] = float(np.mean(values))
    return summary


def build_consensus_masks(mask_pairs: list[tuple[np.ndarray, np.ndarray]], consensus_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fpna_stack = np.stack([pair[0].astype(float) for pair in mask_pairs], axis=0)
    fpnb_stack = np.stack([pair[1].astype(float) for pair in mask_pairs], axis=0)
    fpna_prob = np.mean(fpna_stack, axis=0)
    fpnb_prob = np.mean(fpnb_stack, axis=0)
    fpna_consensus = fpna_prob >= consensus_threshold
    fpnb_consensus = fpnb_prob >= consensus_threshold
    return fpna_prob, fpnb_prob, fpna_consensus, fpnb_consensus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MD maps with individual FPN_A/FPN_B subnetworks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python md_vs_fpn.py --subjects 04 06 07 09 11 13 14 15\n"
            "  python md_vs_fpn.py --all-subjects --threshold-percent 20\n"
            "  python md_vs_fpn.py --subjects 04 06 --threshold-z 2.3\n\n"
            "Notes:\n"
            "  - For current group MD mean maps, percentile thresholding is usually more practical\n"
            "    than z-thresholding because the group file is an averaged MD map, not a group z-stat map."
        ),
    )
    parser.add_argument("--subjects", nargs="+", help="Subject IDs, for example: 04 06 07")
    parser.add_argument("--all-subjects", action="store_true", help="Use all subjects that have both MD and subnetwork files")
    parser.add_argument("--md-dir", default=DEFAULT_MD_DIR, help=f"Vertex-wise MD output directory (default: {DEFAULT_MD_DIR})")
    parser.add_argument("--subnetwork-dir", default=DEFAULT_SUBNETWORK_DIR, help=f"Infomap subnetwork directory (default: {DEFAULT_SUBNETWORK_DIR})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--threshold-z", type=float, default=None, help="Threshold MD map at z >= this value")
    parser.add_argument("--threshold-percent", type=float, default=20.0, help="Keep top %% of MD vertices for overlap analysis (default: 20)")
    parser.add_argument("--k-index", type=int, default=0, help="Row index in the relabeled subnetwork file to use (default: 0)")
    parser.add_argument("--fpna-label", type=int, default=None, help="Explicit label value for FPN_A")
    parser.add_argument("--fpnb-label", type=int, default=None, help="Explicit label value for FPN_B")
    parser.add_argument("--consensus-threshold", type=float, default=0.5, help="Subject-consensus threshold for group FPN masks (default: 0.5)")
    parser.add_argument("--skip-group", action="store_true", help="Skip group MD analyses")
    parser.add_argument("--skip-overlap-dscalars", action="store_true", help="Skip workbench-ready overlap dscalar outputs")
    args = parser.parse_args()

    if args.threshold_z is not None and args.threshold_percent is not None:
        parser.error("Use either --threshold-z or --threshold-percent, not both.")
    if not args.all_subjects and not args.subjects:
        args.all_subjects = True
    if args.consensus_threshold <= 0 or args.consensus_threshold > 1:
        parser.error("--consensus-threshold must be in (0, 1].")
    return args


def main() -> None:
    args = parse_args()

    subjects = resolve_subjects(args.md_dir, args.subnetwork_dir, args.subjects)
    if not subjects:
        raise SystemExit("No subjects found with both MD maps and FPN relabeled files.")

    os.makedirs(args.output, exist_ok=True)

    individual_rows: list[dict[str, float | str]] = []
    group_subject_rows: list[dict[str, float | str]] = []
    mask_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    overlap_outputs: list[str] = []
    approach_b_outputs: list[str] = []

    overlap_dir = os.path.join(args.output, "overlap_dscalars")
    individual_overlap_dir = os.path.join(overlap_dir, "individual")
    group_overlap_dir = os.path.join(overlap_dir, "group")
    approach_b_dir = os.path.join(args.output, "approach_b_md_value_maps")
    individual_approach_b_dir = os.path.join(approach_b_dir, "individual")
    group_approach_b_dir = os.path.join(approach_b_dir, "group")
    if not args.skip_overlap_dscalars:
        os.makedirs(individual_overlap_dir, exist_ok=True)
        os.makedirs(group_overlap_dir, exist_ok=True)
    os.makedirs(individual_approach_b_dir, exist_ok=True)
    os.makedirs(group_approach_b_dir, exist_ok=True)

    group_path = group_md_map_path(args.md_dir)
    run_group = (not args.skip_group) and os.path.exists(group_path)
    if run_group:
        group_md_data, group_md_img = load_cifti_vector(group_path)
    else:
        group_md_data = None
        group_md_img = None

    print(f"Subjects: {' '.join(subjects)}")
    print(f"MD directory: {args.md_dir}")
    print(f"Subnetwork directory: {args.subnetwork_dir}")
    if args.threshold_percent is not None:
        print(f"Overlap threshold: top {args.threshold_percent:g}% of MD vertices")
    elif args.threshold_z is not None:
        print(f"Overlap threshold: z >= {args.threshold_z:g}")
    else:
        print("Overlap threshold: MD > 0")

    for subject in subjects:
        md_path = subject_md_map_path(args.md_dir, subject)
        subnet_path = subject_subnetwork_path(args.subnetwork_dir, subject)

        if not os.path.exists(md_path):
            print(f"Skipping sub-{subject}: missing MD map {md_path}")
            continue
        if not os.path.exists(subnet_path):
            print(f"Skipping sub-{subject}: missing subnetwork map {subnet_path}")
            continue

        md_data, md_img = load_cifti_vector(md_path)
        subnetwork_data, subnet_img = load_cifti_row(subnet_path, args.k_index)
        label_a, label_b = infer_label_ids(subnet_img, args.k_index, args.fpna_label, args.fpnb_label)

        fpna_mask = subnetwork_data == label_a
        fpnb_mask = subnetwork_data == label_b
        if not np.any(fpna_mask) or not np.any(fpnb_mask):
            print(
                f"Skipping sub-{subject}: empty mask after label selection "
                f"(FPN_A={label_a}, FPN_B={label_b})"
            )
            continue

        mask_pairs.append((fpna_mask, fpnb_mask))
        row = analyze_single_map(
            analysis_level="individual",
            map_subject=subject,
            mask_subject=subject,
            md_data=md_data,
            fpna_mask=fpna_mask,
            fpnb_mask=fpnb_mask,
            threshold_z=args.threshold_z,
            threshold_percent=args.threshold_percent,
        )
        individual_rows.append(row)

        print(
            f"sub-{subject}: mean(FPN_A)={row['mean_md_fpna']:.3f}, "
            f"mean(FPN_B)={row['mean_md_fpnb']:.3f}, "
            f"A-B={row['mean_diff_fpna_minus_fpnb']:.3f}, "
            f"thresholded_MD={row['md_vertices_thresholded']}"
        )

        if not args.skip_overlap_dscalars:
            maps_2d, map_names, threshold_mode, _ = build_overlap_visualization_maps(
                md_data=md_data,
                fpna_mask=fpna_mask,
                fpnb_mask=fpnb_mask,
                threshold_z=args.threshold_z,
                threshold_percent=args.threshold_percent,
            )
            threshold_tag = sanitize_token(threshold_mode)
            subject_overlap_path = os.path.join(
                individual_overlap_dir,
                f"sub-{subject}_md_fpn_overlap_{threshold_tag}.dscalar.nii",
            )
            save_dscalar_maps(subject_overlap_path, maps_2d, map_names, md_img)
            overlap_outputs.append(subject_overlap_path)

        approach_b_maps_2d, approach_b_map_names = build_masked_md_value_maps(
            md_data=md_data,
            fpna_mask=fpna_mask,
            fpnb_mask=fpnb_mask,
        )
        subject_approach_b_path = os.path.join(
            individual_approach_b_dir,
            f"sub-{subject}_md_in_fpn_masks.dscalar.nii",
        )
        save_dscalar_maps(subject_approach_b_path, approach_b_maps_2d, approach_b_map_names, md_img)
        approach_b_outputs.append(subject_approach_b_path)

        if run_group and group_md_data is not None:
            group_row = analyze_single_map(
                analysis_level="group_map_vs_individual_mask",
                map_subject="group",
                mask_subject=subject,
                md_data=group_md_data,
                fpna_mask=fpna_mask,
                fpnb_mask=fpnb_mask,
                threshold_z=args.threshold_z,
                threshold_percent=args.threshold_percent,
            )
            group_subject_rows.append(group_row)

            if not args.skip_overlap_dscalars and group_md_img is not None:
                group_maps_2d, group_map_names, threshold_mode, _ = build_overlap_visualization_maps(
                    md_data=group_md_data,
                    fpna_mask=fpna_mask,
                    fpnb_mask=fpnb_mask,
                    threshold_z=args.threshold_z,
                    threshold_percent=args.threshold_percent,
                )
                threshold_tag = sanitize_token(threshold_mode)
                group_subject_overlap_path = os.path.join(
                    group_overlap_dir,
                    f"group_md_vs_sub-{subject}_fpn_overlap_{threshold_tag}.dscalar.nii",
                )
                save_dscalar_maps(group_subject_overlap_path, group_maps_2d, group_map_names, group_md_img)
                overlap_outputs.append(group_subject_overlap_path)

            if group_md_img is not None:
                group_approach_b_maps_2d, group_approach_b_map_names = build_masked_md_value_maps(
                    md_data=group_md_data,
                    fpna_mask=fpna_mask,
                    fpnb_mask=fpnb_mask,
                )
                group_subject_approach_b_path = os.path.join(
                    group_approach_b_dir,
                    f"group_md_in_sub-{subject}_fpn_masks.dscalar.nii",
                )
                save_dscalar_maps(
                    group_subject_approach_b_path,
                    group_approach_b_maps_2d,
                    group_approach_b_map_names,
                    group_md_img,
                )
                approach_b_outputs.append(group_subject_approach_b_path)

    if not individual_rows:
        raise SystemExit("No valid subject comparisons were produced.")

    individual_csv = os.path.join(args.output, "individual_subjects.csv")
    write_csv(individual_csv, individual_rows)

    summaries = [summarize_rows(individual_rows, "individual_subjects")]

    if group_subject_rows:
        group_subject_csv = os.path.join(args.output, "group_vs_individual_masks.csv")
        write_csv(group_subject_csv, group_subject_rows)
        summaries.append(summarize_rows(group_subject_rows, "group_vs_individual_masks"))

    if run_group and group_md_data is not None and mask_pairs:
        fpna_prob, fpnb_prob, fpna_consensus, fpnb_consensus = build_consensus_masks(mask_pairs, args.consensus_threshold)
        consensus_row = analyze_single_map(
            analysis_level="group_map_vs_consensus_masks",
            map_subject="group",
            mask_subject="consensus",
            md_data=group_md_data,
            fpna_mask=fpna_consensus,
            fpnb_mask=fpnb_consensus,
            threshold_z=args.threshold_z,
            threshold_percent=args.threshold_percent,
        )
        consensus_row["fpna_consensus_threshold"] = args.consensus_threshold
        consensus_row["fpnb_consensus_threshold"] = args.consensus_threshold

        consensus_csv = os.path.join(args.output, "group_vs_consensus_masks.csv")
        write_csv(consensus_csv, [consensus_row])
        summaries.append(consensus_row)

        np.savez(
            os.path.join(args.output, "group_mask_probabilities.npz"),
            subjects=np.array(subjects, dtype=object),
            fpna_probability=fpna_prob,
            fpnb_probability=fpnb_prob,
            fpna_consensus=fpna_consensus.astype(np.uint8),
            fpnb_consensus=fpnb_consensus.astype(np.uint8),
            consensus_threshold=args.consensus_threshold,
        )

        print(
            "group consensus: "
            f"mean(FPN_A)={consensus_row['mean_md_fpna']:.3f}, "
            f"mean(FPN_B)={consensus_row['mean_md_fpnb']:.3f}, "
            f"A-B={consensus_row['mean_diff_fpna_minus_fpnb']:.3f}"
        )

        if not args.skip_overlap_dscalars and group_md_img is not None:
            consensus_maps_2d, consensus_map_names, threshold_mode, _ = build_overlap_visualization_maps(
                md_data=group_md_data,
                fpna_mask=fpna_consensus,
                fpnb_mask=fpnb_consensus,
                threshold_z=args.threshold_z,
                threshold_percent=args.threshold_percent,
            )
            threshold_tag = sanitize_token(threshold_mode)
            consensus_overlap_path = os.path.join(
                group_overlap_dir,
                f"group_md_vs_consensus_fpn_overlap_{threshold_tag}.dscalar.nii",
            )
            save_dscalar_maps(consensus_overlap_path, consensus_maps_2d, consensus_map_names, group_md_img)
            overlap_outputs.append(consensus_overlap_path)

        if group_md_img is not None:
            consensus_approach_b_maps_2d, consensus_approach_b_map_names = build_masked_md_value_maps(
                md_data=group_md_data,
                fpna_mask=fpna_consensus,
                fpnb_mask=fpnb_consensus,
            )
            consensus_approach_b_path = os.path.join(
                group_approach_b_dir,
                "group_md_in_consensus_fpn_masks.dscalar.nii",
            )
            save_dscalar_maps(
                consensus_approach_b_path,
                consensus_approach_b_maps_2d,
                consensus_approach_b_map_names,
                group_md_img,
            )
            approach_b_outputs.append(consensus_approach_b_path)

    summary_csv = os.path.join(args.output, "summary.csv")
    write_csv(summary_csv, summaries)

    print("\nSaved:")
    print(f"  {individual_csv}")
    if group_subject_rows:
        print(f"  {os.path.join(args.output, 'group_vs_individual_masks.csv')}")
    if run_group and group_md_data is not None and mask_pairs:
        print(f"  {os.path.join(args.output, 'group_vs_consensus_masks.csv')}")
        print(f"  {os.path.join(args.output, 'group_mask_probabilities.npz')}")
    if overlap_outputs:
        print(f"  {overlap_dir} ({len(overlap_outputs)} dscalar files)")
    if approach_b_outputs:
        print(f"  {approach_b_dir} ({len(approach_b_outputs)} dscalar files)")
    print(f"  {summary_csv}")


if __name__ == "__main__":
    main()