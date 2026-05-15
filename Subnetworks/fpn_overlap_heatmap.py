#!/usr/bin/env python3
"""Build across-subject FPN overlap heatmaps as Workbench-ready dscalar files.

Supports:
- whole FPN (all non-zero labels)
- FPN subnetworks (A/B by label IDs)
- both outputs in one run
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
HUBNESS_DIR = REPO_ROOT / "Hubness"
if str(HUBNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HUBNESS_DIR))

from hubness_utils import (  # noqa: E402
    build_binary_mask,
    count_to_fraction_map,
    load_cifti_row,
    save_dscalar_maps,
    subject_mask_count_map,
)

DEFAULT_SUBNETWORK_BASE = "/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_derivation/infomap"
DEFAULT_OUTPUT_DIR = "/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/fpn_overlap_heatmap"
DEFAULT_SUBJECTS_FILE = "/ptmp/hmueller2/2025_ibc_latent/misc/subjects_resting.txt"
DEFAULT_EXPECTED_SUBJECTS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create vertex-wise subject-count overlap maps for FPN masks.")
    parser.add_argument("--subjects", nargs="+", default=None, help="Subject IDs without sub- prefix")
    parser.add_argument("--subjects-file", default=DEFAULT_SUBJECTS_FILE, help="File with one subject per line")
    parser.add_argument("--subnetwork-base", default=DEFAULT_SUBNETWORK_BASE, help="Base directory with sub-XX folders")
    parser.add_argument("--input-name", default=None, help="Optional exact input filename in each subject folder")
    parser.add_argument("--k-index", type=int, default=0, help="Row index for 2D CIFTI inputs (default: 0)")
    parser.add_argument("--mode", choices=["whole", "subnetworks", "both"], default="both")
    parser.add_argument("--fpna-label", type=int, default=None, help="Explicit label ID for FPN-A")
    parser.add_argument("--fpnb-label", type=int, default=None, help="Explicit label ID for FPN-B")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-subjects", type=int, default=DEFAULT_EXPECTED_SUBJECTS)
    return parser.parse_args()


def parse_subjects(subjects: list[str] | None, subjects_file: str) -> list[str]:
    if subjects:
        return [str(s).replace("sub-", "") for s in subjects]

    path = Path(subjects_file)
    if not path.exists():
        raise FileNotFoundError(f"Subjects file not found: {path}")

    parsed: list[str] = []
    for line in path.read_text().splitlines():
        token = line.strip().split()
        if not token:
            continue
        parsed.append(token[0].replace("sub-", ""))

    if not parsed:
        raise ValueError(f"No subjects found in {path}")
    return parsed


def resolve_input_path(base: Path, subject: str, explicit_name: str | None) -> Path:
    subject_dir = base / f"sub-{subject}"
    if explicit_name:
        path = subject_dir / explicit_name
        if not path.exists():
            raise FileNotFoundError(f"Input file not found for sub-{subject}: {path}")
        return path

    dlabel = subject_dir / f"{subject}_FPN_infomap_communities_kmeans_relabeled.dlabel.nii"
    if dlabel.exists():
        return dlabel

    dscalar = subject_dir / f"{subject}_FPN_infomap_communities_kmeans_relabeled.dscalar.nii"
    if dscalar.exists():
        return dscalar

    raise FileNotFoundError(
        f"No relabeled FPN map found for sub-{subject}. Tried {dlabel.name} and {dscalar.name}"
    )


def infer_subnetwork_ids(data_1d: np.ndarray, explicit_a: int | None, explicit_b: int | None) -> tuple[int, int]:
    if explicit_a is not None and explicit_b is not None:
        if explicit_a == explicit_b:
            raise ValueError("--fpna-label and --fpnb-label must be different")
        return int(explicit_a), int(explicit_b)

    positive_labels = sorted(int(v) for v in np.unique(data_1d.astype(int)) if int(v) > 0)
    if len(positive_labels) < 2:
        raise ValueError(f"Expected at least 2 positive labels for subnetworks, found: {positive_labels}")

    return positive_labels[0], positive_labels[1]


def main() -> None:
    args = parse_args()

    subjects = parse_subjects(args.subjects, args.subjects_file)
    if args.expected_subjects and len(subjects) != int(args.expected_subjects):
        print(f"warning: expected {int(args.expected_subjects)} subjects but got {len(subjects)}")

    base = Path(args.subnetwork_base)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    whole_masks: list[np.ndarray] = []
    fpna_masks: list[np.ndarray] = []
    fpnb_masks: list[np.ndarray] = []
    union_masks: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []

    template_img = None
    expected_shape: tuple[int, ...] | None = None
    used_subjects: list[str] = []

    for subject in subjects:
        in_path = resolve_input_path(base, subject, args.input_name)
        data, img = load_cifti_row(in_path, row_index=args.k_index)

        if template_img is None:
            template_img = img
            expected_shape = data.shape
        elif expected_shape is not None and data.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for sub-{subject}: got {data.shape}, expected {expected_shape}"
            )

        used_subjects.append(subject)

        if args.mode in {"whole", "both"}:
            whole_mask = build_binary_mask(data, include_values=None, positive_as_true=True)
            whole_masks.append(whole_mask)

        if args.mode in {"subnetworks", "both"}:
            fpna_id, fpnb_id = infer_subnetwork_ids(data, args.fpna_label, args.fpnb_label)
            fpna_mask = build_binary_mask(data, include_values=[fpna_id])
            fpnb_mask = build_binary_mask(data, include_values=[fpnb_id])
            union_mask = fpna_mask | fpnb_mask

            fpna_masks.append(fpna_mask)
            fpnb_masks.append(fpnb_mask)
            union_masks.append(union_mask)

            metadata_rows.append(
                {
                    "subject": subject,
                    "input_path": str(in_path),
                    "fpna_label": fpna_id,
                    "fpnb_label": fpnb_id,
                    "fpna_vertices": int(fpna_mask.sum()),
                    "fpnb_vertices": int(fpnb_mask.sum()),
                    "fpn_union_vertices": int(union_mask.sum()),
                }
            )

    if template_img is None or not used_subjects:
        raise ValueError("No valid subject inputs were loaded")

    if args.mode in {"whole", "both"}:
        whole_count = subject_mask_count_map(whole_masks)
        whole_fraction = count_to_fraction_map(whole_count, len(whole_masks))
        whole_any = (whole_count > 0).astype(np.float32)

        whole_maps = np.stack(
            [
                whole_count.astype(np.float32),
                whole_fraction.astype(np.float32),
                whole_any,
            ],
            axis=0,
        )
        whole_names = [
            "whole_fpn_subject_count",
            "whole_fpn_subject_fraction",
            "whole_fpn_any_subject",
        ]
        whole_out = out_dir / f"fpn_whole_overlap_heatmap_n{len(whole_masks)}.dscalar.nii"
        save_dscalar_maps(whole_out, whole_maps, whole_names, template_img)
        print(f"saved: {whole_out}")

    if args.mode in {"subnetworks", "both"}:
        fpna_count = subject_mask_count_map(fpna_masks)
        fpnb_count = subject_mask_count_map(fpnb_masks)
        union_count = subject_mask_count_map(union_masks)

        fpna_fraction = count_to_fraction_map(fpna_count, len(fpna_masks))
        fpnb_fraction = count_to_fraction_map(fpnb_count, len(fpnb_masks))
        union_fraction = count_to_fraction_map(union_count, len(union_masks))

        subnet_maps = np.stack(
            [
                fpna_count.astype(np.float32),
                fpna_fraction.astype(np.float32),
                fpnb_count.astype(np.float32),
                fpnb_fraction.astype(np.float32),
                union_count.astype(np.float32),
                union_fraction.astype(np.float32),
            ],
            axis=0,
        )
        subnet_names = [
            "fpn_a_subject_count",
            "fpn_a_subject_fraction",
            "fpn_b_subject_count",
            "fpn_b_subject_fraction",
            "fpn_union_subject_count",
            "fpn_union_subject_fraction",
        ]
        subnet_out = out_dir / f"fpn_subnet_overlap_heatmap_n{len(fpna_masks)}.dscalar.nii"
        save_dscalar_maps(subnet_out, subnet_maps, subnet_names, template_img)
        print(f"saved: {subnet_out}")

        metadata_csv = out_dir / "fpn_subnetwork_label_metadata.csv"
        with metadata_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "subject",
                    "input_path",
                    "fpna_label",
                    "fpnb_label",
                    "fpna_vertices",
                    "fpnb_vertices",
                    "fpn_union_vertices",
                ],
            )
            writer.writeheader()
            for row in metadata_rows:
                writer.writerow(row)
        print(f"saved: {metadata_csv}")

    np.savez_compressed(
        out_dir / "fpn_overlap_heatmap_subjects.npz",
        subjects=np.asarray(used_subjects, dtype=object),
        mode=args.mode,
    )
    print(f"done: {len(used_subjects)} subjects")


if __name__ == "__main__":
    main()
