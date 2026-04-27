"""
Split Glasser parcels by overlapping network labels.

For each subject, this script:
1) Loads the subject network dlabel in fsLR CIFTI space.
2) Overlays it on the Glasser 360 cortical parcellation.
3) Retains parcel/network fragments whose parcel-fraction overlap exceeds a threshold.
4) Writes a split-parcel manifest plus one dense mask per retained fragment.
5) Keeps a hard parcel summary for downstream compatibility.

Outputs:
- sub-XX/parcel_network_assignment_subject.csv: summary of hard parcel assignments
- sub-XX/split_parcels_manifest_subject_t0ZZ.csv: detailed manifest of split fragments and their properties
- sub-XX/split_parcels_t0ZZ/parcel-YYY_*_partA.dscalar.nii: dense mask for each retained split fragment
- sub-XX/sub-XX_split_parcels_combined_t0ZZ_parcel_colored.dlabel.nii: combined dlabel with all retained split parcels for parcel visualization
- sub-XX/sub-XX_split_parcels_combined_t0ZZ_network_colored.dlabel.nii: combined dlabel with all retained split parcels for network visualization
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from hubness_utils import (
    discover_subjects_from_subdirs,
    ensure_dir,
    extract_network_colors_from_dlabel,
    infer_label_names_from_dlabel,
    is_fpn_network_name,
    load_glasser_parcellation,
    load_subject_fpna_fpnb_flags,
    map_to_cortex,
    sanitize_for_filename,
    split_manifest_path,
    split_parcels_dir,
    split_threshold_tag,
    subject_network_label_path,
)

DEFAULT_NETWORK_LABEL_BASE = "/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks/derived_networks"
DEFAULT_PARCELIZED_FPN_BASE = "/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/parcelized_fpn"
DEFAULT_OUTPUT_DIR = "/ptmp/hmueller2/2025_ibc_latent/outputs/hubness"
DEFAULT_OVERLAP_THRESHOLD = 0.30
FPN_MODE_UNIFIED = "unified"
FPN_MODE_SPLIT = "split"


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """Convert HSV color to RGB."""
    c = v * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = v - c
    if h < 1 / 6:
        r, g, b = c, x, 0
    elif h < 2 / 6:
        r, g, b = x, c, 0
    elif h < 3 / 6:
        r, g, b = 0, c, x
    elif h < 4 / 6:
        r, g, b = 0, x, c
    elif h < 5 / 6:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (r + m, g + m, b + m)


def save_split_mask(
    template_img: nib.Cifti2Image,
    cortical_indices: np.ndarray,
    parcel_mask: np.ndarray,
    out_path: Path,
    map_name: str,
) -> None:
    data = np.zeros(template_img.shape, dtype=np.float32)
    mask_values = parcel_mask.astype(np.float32)

    if data.ndim == 2:
        data[0, cortical_indices] = mask_values
    elif data.ndim == 1:
        data[cortical_indices] = mask_values
    else:
        raise ValueError(f"Unsupported CIFTI data shape for split mask: {data.shape}")

    img = nib.Cifti2Image(data, header=template_img.header.copy())
    nib.save(img, str(out_path))


def compute_subject_split_assignments(
    subject: str,
    output_dir: Path,
    parcellation_cortex: np.ndarray,
    unique_parcels: np.ndarray,
    parcel_name_map: dict[int, str],
    template_img: nib.Cifti2Image,
    cortical_indices: np.ndarray,
    net_cortex: np.ndarray,
    net_names: dict[int, str],
    overlap_threshold: float,
) -> Path:
    """Compute split parcel fragments above the overlap threshold."""
    split_rows: list[dict[str, object]] = []

    for parcel_id in unique_parcels:
        parcel_mask = parcellation_cortex == int(parcel_id)
        parcel_size = int(np.sum(parcel_mask))
        
        if parcel_size == 0:
            continue

        parcel_name = parcel_name_map.get(int(parcel_id), f"PARCEL_{int(parcel_id)}")
        labels_in_parcel = net_cortex[parcel_mask]
        labels_in_parcel = labels_in_parcel[labels_in_parcel > 0]

        if labels_in_parcel.size == 0:
            continue

        unique_nets, counts = np.unique(labels_in_parcel, return_counts=True)
        order = np.argsort(counts)[::-1]

        for candidate_rank, idx in enumerate(order, start=1):
            net_id = int(unique_nets[idx])
            count = int(counts[idx])
            overlap_fraction = float(count) / float(parcel_size)
            net_name = net_names.get(net_id, f"LABEL_{net_id}")
            retained = overlap_fraction >= float(overlap_threshold)
            split_label = f"{parcel_name}__{net_name}"

            split_rows.append(
                {
                    "subject": subject,
                    "parcel_id": int(parcel_id),
                    "parcel_name": parcel_name,
                    "parcel_vertex_count": parcel_size,
                    "network_id": net_id,
                    "network_name": net_name,
                    "overlap_vertices": count,
                    "overlap_fraction": round(overlap_fraction, 6),
                    "candidate_rank": candidate_rank,
                    "retained": int(retained),
                    "split_label": split_label,
                }
            )

    out = pd.DataFrame(split_rows)
    subject_dir = ensure_dir(output_dir / f"sub-{subject}")
    out_path = split_manifest_path(output_dir, subject, overlap_threshold)
    if not out.empty:
        out = out.sort_values(["parcel_id", "retained", "overlap_fraction", "network_id"], ascending=[True, False, False, True])

    retained = out[out["retained"] == 1].copy() if not out.empty else out
    if not retained.empty:
        retained["retained_rank"] = retained.groupby("parcel_id").cumcount() + 1
        for _, row in retained.iterrows():
            parcel_id = int(row["parcel_id"])
            parcel_name = str(row["parcel_name"])
            net_name = str(row["network_name"])
            retained_rank = int(row["retained_rank"])
            split_label = sanitize_for_filename(f"{parcel_name}__{net_name}__part{retained_rank}")
            mask = (parcellation_cortex == parcel_id) & (net_cortex == int(row["network_id"]))
            split_dir = ensure_dir(split_parcels_dir(output_dir, subject, overlap_threshold))
            mask_path = split_dir / f"sub-{subject}_parcel-{parcel_id:03d}_{split_label}.dscalar.nii"
            save_split_mask(
                template_img=template_img,
                cortical_indices=cortical_indices,
                parcel_mask=mask,
                out_path=mask_path,
                map_name=split_label,
            )
            out.loc[
                (out["parcel_id"] == parcel_id) & (out["network_id"] == int(row["network_id"])),
                "split_mask_path",
            ] = str(mask_path)

    if "split_mask_path" not in out.columns:
        out["split_mask_path"] = ""
    out["split_mask_path"] = out["split_mask_path"].fillna("")
    out.to_csv(out_path, index=False)
    return out_path


def compute_subject_assignment(
    subject: str,
    network_label_base: str,
    parcelized_fpn_base: str,
    output_dir: Path,
    parcellation_path: str | None,
    overlap_threshold: float,
    fpn_mode: str,
) -> tuple[Path, Path]:
    parcellation_cortex, unique_parcels, parcel_name_map, cortical_indices, _ = load_glasser_parcellation(parcellation_path)

    network_path = subject_network_label_path(network_label_base, subject)
    if not network_path.exists():
        raise FileNotFoundError(f"Missing network dlabel for sub-{subject}: {network_path}")

    net_img = nib.load(str(network_path))
    net_data = net_img.get_fdata().squeeze().astype(int)
    net_cortex = map_to_cortex(net_data, cortical_indices)
    net_names = infer_label_names_from_dlabel(net_img)

    rows = []
    for parcel_id in unique_parcels:
        mask = parcellation_cortex == int(parcel_id)
        labels = net_cortex[mask]
        labels = labels[labels > 0]

        if labels.size == 0:
            assigned_id = 0
            assigned_name = "UNASSIGNED"
            frac = 0.0
        else:
            vals, counts = np.unique(labels, return_counts=True)
            total_count = float(np.sum(counts))
            fractions = counts / total_count
            eligible = fractions >= float(overlap_threshold)

            if np.any(eligible):
                eligible_vals = vals[eligible]
                eligible_counts = counts[eligible]
                idx = int(np.argmax(eligible_counts))
                assigned_id = int(eligible_vals[idx])
                frac = float(eligible_counts[idx] / total_count)
            else:
                idx = int(np.argmax(counts))
                assigned_id = int(vals[idx])
                frac = float(counts[idx] / total_count)

            assigned_name = net_names.get(assigned_id, f"LABEL_{assigned_id}")

        rows.append(
            {
                "subject": subject,
                "parcel_id": int(parcel_id),
                "parcel_name": parcel_name_map.get(int(parcel_id), f"PARCEL_{int(parcel_id)}"),
                "assigned_network_id": assigned_id,
                "assigned_network_name": assigned_name,
                "assignment_fraction": frac,
                "is_fpn": int(is_fpn_network_name(assigned_name)),
            }
        )

    out = pd.DataFrame(rows)
    if fpn_mode == FPN_MODE_SPLIT:
        fpn_flags = load_subject_fpna_fpnb_flags(subject, parcelized_fpn_base)
        if not fpn_flags.empty:
            out = out.merge(fpn_flags, on="parcel_id", how="left")
        for col in ["fpna_selected", "fpnb_selected", "fpn_selected"]:
            if col not in out.columns:
                out[col] = 0
            out[col] = out[col].fillna(0).astype(int)
    else:
        # Unified mode: keep explicit FPN as a single flag and disable split flags.
        out["fpna_selected"] = 0
        out["fpnb_selected"] = 0
        out["fpn_selected"] = out["is_fpn"].fillna(0).astype(int)

    subject_dir = ensure_dir(output_dir / f"sub-{subject}")
    hard_assignment_path = subject_dir / "parcel_network_assignment_subject.csv"
    out.to_csv(hard_assignment_path, index=False)
    
    split_manifest_path = compute_subject_split_assignments(
        subject=subject,
        output_dir=output_dir,
        parcellation_cortex=parcellation_cortex,
        unique_parcels=unique_parcels,
        parcel_name_map=parcel_name_map,
        template_img=net_img,
        cortical_indices=cortical_indices,
        net_cortex=net_cortex,
        net_names=net_names,
        overlap_threshold=overlap_threshold,
    )
    
    return hard_assignment_path, split_manifest_path


def create_subject_split_dlabel(
    subject: str,
    output_dir: Path,
    parcellation_cortex: np.ndarray,
    unique_parcels: np.ndarray,
    parcel_name_map: dict[int, str],
    template_img: nib.Cifti2Image,
    cortical_indices: np.ndarray,
    net_cortex: np.ndarray,
    net_names: dict[int, str],
    color_mode: str = "parcel",
    overlap_threshold: float = 0.30,
) -> list[Path]:
    """Create combined dlabel file(s) with all split parcels for wb_view visualization.
    
    Args:
        color_mode: "parcel" for distinct colors per parcel, "network" for network colors,
                   "both" to create both versions.
    """
    subject_dir = Path(output_dir) / f"sub-{subject}"
    threshold_tag = split_threshold_tag(overlap_threshold)
    manifest_path = split_manifest_path(output_dir, subject, overlap_threshold)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    retained_manifest = manifest[manifest["retained"] == 1].copy()

    if retained_manifest.empty:
        return []

    # Get network colors if needed
    network_colors = {}
    if color_mode in ("network", "both"):
        network_colors = extract_network_colors_from_dlabel(template_img)

    output_paths = []
    
    for mode in ([color_mode] if color_mode != "both" else ["parcel", "network"]):
        # Create output data: full grayordinate space
        output_data = np.zeros(template_img.shape, dtype=np.int32)

        # Create label table for the output dlabel
        label_table = {}
        label_id = 1

        retained_manifest_sorted = retained_manifest.sort_values(["parcel_id", "overlap_fraction"], ascending=[True, False])

        for idx, row in retained_manifest_sorted.iterrows():
            parcel_id = int(row["parcel_id"])
            network_id = int(row["network_id"])
            network_name = str(row["network_name"])
            parcel_name = str(row["parcel_name"])

            # Create the split parcel mask
            parcel_mask = parcellation_cortex == parcel_id
            network_mask = net_cortex == network_id
            split_mask = parcel_mask & network_mask

            if not np.any(split_mask):
                continue

            # Assign label and create label entry with RGBA color
            label_name = f"{parcel_name}_{network_name}"
            
            if mode == "parcel":
                # Use consistent coloring: generate color from hash of label_id
                hue = (label_id * 137.5) % 360.0  # Golden angle for distinct colors
                rgb = _hsv_to_rgb(hue / 360.0, 0.8, 0.9)
                rgba = rgb + (1.0,)  # Add alpha channel (as float 0-1)
            else:  # mode == "network"
                # Use network color if available, otherwise fallback to parcel coloring
                if network_id in network_colors:
                    rgba = network_colors[network_id]
                else:
                    # Fallback: use a muted network-based color
                    hue = (network_id * 137.5) % 360.0
                    rgb = _hsv_to_rgb(hue / 360.0, 0.5, 0.8)
                    rgba = rgb + (1.0,)
            
            label_table[label_id] = (label_name, rgba)

            # Assign to output
            output_data[0, cortical_indices[split_mask]] = label_id
            label_id += 1

        # Create output dlabel with label table
        # Use the template header structure and update just the label axis
        second_axis = template_img.header.get_axis(1)
        
        # Create new label axis with our label table
        # Note: name needs to be a numpy array with shape (1,)
        label_axis = nib.cifti2.LabelAxis(name=np.array(["split_parcels"]), label=label_table)
        header = nib.cifti2.Cifti2Header.from_axes((label_axis, second_axis))
        
        out_img = nib.Cifti2Image(output_data.astype(np.int32), header=header)

        # Write output
        suffix = f"_{mode}_colored" if color_mode == "both" else ""
        out_path = subject_dir / f"sub-{subject}_split_parcels_combined{threshold_tag}{suffix}.dlabel.nii"
        nib.save(out_img, str(out_path))
        output_paths.append(out_path)

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1: Split Glasser parcels by overlapping network labels and write parcel masks.")
    parser.add_argument("--subjects", nargs="+", default=None, help="Subject IDs, e.g. 01 02 04")
    parser.add_argument("--network-label-base", default=DEFAULT_NETWORK_LABEL_BASE)
    parser.add_argument("--parcelized-fpn-base", default=DEFAULT_PARCELIZED_FPN_BASE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--parcellation-path", default=None)
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=DEFAULT_OVERLAP_THRESHOLD,
        help="Retain split fragments only when they cover at least this fraction of the original parcel.",
    )
    parser.add_argument(
        "--fpn-mode",
        choices=[FPN_MODE_UNIFIED, FPN_MODE_SPLIT],
        default=FPN_MODE_UNIFIED,
        help="FPN handling mode: unified FPN (default) or split FPNA/FPNB from subject-specific derivations.",
    )
    parser.add_argument(
        "--dlabel-coloring",
        choices=["parcel", "network", "both"],
        default="both",
        help="Coloring mode for dlabel: 'parcel' (each parcel distinct), 'network' (by network), or 'both' (create both).",
    )
    parser.add_argument("--skip-dlabel", action="store_true", help="Skip creating combined dlabel file for wb_view visualization.")
    parser.add_argument("--dlabel-only", action="store_true", help="Only create dlabels from existing split_parcels; skip all other computations.")
    return parser.parse_args()


def create_subject_dlabels(
    subject: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[Path]:
    net_img = nib.load(str(subject_network_label_path(args.network_label_base, subject)))
    net_data = net_img.get_fdata().squeeze().astype(int)
    net_names = infer_label_names_from_dlabel(net_img)

    parcellation_cortex, unique_parcels, parcel_name_map, cortical_indices, _ = load_glasser_parcellation(args.parcellation_path)
    net_cortex = map_to_cortex(net_data, cortical_indices)

    return create_subject_split_dlabel(
        subject=subject,
        output_dir=output_dir,
        parcellation_cortex=parcellation_cortex,
        unique_parcels=unique_parcels,
        parcel_name_map=parcel_name_map,
        template_img=net_img,
        cortical_indices=cortical_indices,
        net_cortex=net_cortex,
        net_names=net_names,
        color_mode=args.dlabel_coloring,
        overlap_threshold=args.overlap_threshold,
    )


def process_subject_dlabel_only(subject: str, args: argparse.Namespace, output_dir: Path) -> list[str]:
    manifest_path = split_manifest_path(output_dir, subject, args.overlap_threshold)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    dlabel_paths = create_subject_dlabels(subject, args, output_dir)
    return [f"sub-{subject}: wrote combined dlabel {dlabel_path}" for dlabel_path in dlabel_paths]


def process_subject_full(subject: str, args: argparse.Namespace, output_dir: Path) -> list[str]:
    hard_path, soft_path = compute_subject_assignment(
        subject=subject,
        network_label_base=args.network_label_base,
        parcelized_fpn_base=args.parcelized_fpn_base,
        output_dir=output_dir,
        parcellation_path=args.parcellation_path,
        overlap_threshold=args.overlap_threshold,
        fpn_mode=args.fpn_mode,
    )

    messages = [
        f"sub-{subject}: wrote parcel summary {hard_path}",
        f"sub-{subject}: wrote split manifest {soft_path}",
    ]

    if not args.skip_dlabel:
        dlabel_paths = create_subject_dlabels(subject, args, output_dir)
        messages.extend([f"sub-{subject}: wrote combined dlabel {dlabel_path}" for dlabel_path in dlabel_paths])

    return messages


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    if not 0.0 <= args.overlap_threshold <= 1.0:
        raise ValueError("--overlap-threshold must be within [0, 1].")

    if args.subjects:
        subjects = args.subjects
    else:
        subjects = discover_subjects_from_subdirs(args.network_label_base)

    if not subjects:
        raise ValueError("No subjects found")

    failures: list[tuple[str, str]] = []

    for subject in subjects:
        try:
            if args.dlabel_only:
                messages = process_subject_dlabel_only(subject, args, output_dir)
            else:
                messages = process_subject_full(subject, args, output_dir)
            for msg in messages:
                print(msg)
        except Exception as exc:
            failures.append((subject, str(exc)))
            print(f"sub-{subject}: FAILED -> {exc}")

    if failures:
        print("\nCompleted with failures:")
        for sub, msg in failures:
            print(f"  sub-{sub}: {msg}")


if __name__ == "__main__":
    main()
