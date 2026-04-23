"""
Split Glasser parcels by overlapping network labels.

For each subject, this script:
1) Loads the subject network dlabel in fsLR CIFTI space.
2) Overlays it on the Glasser 360 cortical parcellation.
3) Retains parcel/network fragments whose parcel-fraction overlap exceeds a threshold.
4) Writes a split-parcel manifest plus one dense mask per retained fragment.
5) Keeps a hard parcel summary for downstream compatibility.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from hubness_utils import (
    ensure_dir,
    infer_label_names_from_dlabel,
    is_fpn_network_name,
    load_glasser_parcellation,
    map_to_cortex,
)

DEFAULT_NETWORK_LABEL_BASE = "/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks/derived_networks"
DEFAULT_PARCELIZED_FPN_BASE = "/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/parcelized_fpn"
DEFAULT_OUTPUT_DIR = "/ptmp/hmueller2/2025_ibc_latent/outputs/hubness"
DEFAULT_OVERLAP_THRESHOLD = 0.30
NETWORK_DLABEL_NAME = "Bipartite_PhysicalCommunities+AlgorithmicLabeling.dlabel.nii"


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


def subject_network_label_path(network_label_base: str, subject: str) -> Path:
    return Path(network_label_base) / f"sub-{subject}" / "resting_state" / NETWORK_DLABEL_NAME


def load_subject_fpna_fpnb_flags(subject: str, parcelized_fpn_base: str) -> pd.DataFrame:
    overlap_path = Path(parcelized_fpn_base) / f"sub-{subject}" / f"sub-{subject}_fpn_parcel_overlap.csv"
    if not overlap_path.exists():
        return pd.DataFrame(columns=["parcel_id", "fpna_selected", "fpnb_selected", "fpn_selected"])

    df = pd.read_csv(overlap_path)
    keep = [c for c in ["parcel_id", "fpna_selected", "fpnb_selected", "fpn_selected"] if c in df.columns]
    return df[keep].copy() if keep else pd.DataFrame(columns=["parcel_id", "fpna_selected", "fpnb_selected", "fpn_selected"])


def sanitize_for_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


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
    threshold_str = f"_t{int(overlap_threshold * 100):03d}"
    out_path = subject_dir / f"parcel_split_manifest_subject{threshold_str}.csv"
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
            split_dir = ensure_dir(subject_dir / f"split_parcels{threshold_str}")
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
    fpn_flags = load_subject_fpna_fpnb_flags(subject, parcelized_fpn_base)
    if not fpn_flags.empty:
        out = out.merge(fpn_flags, on="parcel_id", how="left")
    for col in ["fpna_selected", "fpnb_selected", "fpn_selected"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = out[col].fillna(0).astype(int)

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


def extract_network_colors(template_img: nib.Cifti2Image) -> dict[int, tuple[float, float, float, float]]:
    """Extract RGBA colors from network dlabel label table.
    
    Returns a mapping of network_id -> (R, G, B, A) with values in [0, 1].
    The label table is a dict where each network_id maps to (network_name, (R, G, B, A)).
    """
    network_colors = {}
    first_axis = template_img.header.get_axis(0)
    
    # The label table is a numpy array of shape (1,) containing a dict
    if hasattr(first_axis, 'label') and first_axis.label is not None:
        try:
            # Access the dict from the numpy array
            label_dict = first_axis.label
            if isinstance(label_dict, np.ndarray) and label_dict.dtype == object:
                label_dict = label_dict[0]  # Extract the dict from shape (1,) array
            
            # Now iterate through the network labels
            if isinstance(label_dict, dict):
                for network_id, label_data in label_dict.items():
                    if isinstance(label_data, tuple) and len(label_data) >= 2:
                        # Format is (network_name, (R, G, B, A))
                        network_name, rgba = label_data[0], label_data[1]
                        if isinstance(rgba, (tuple, list)) and len(rgba) == 4:
                            network_colors[network_id] = tuple(float(c) for c in rgba)
        except (AttributeError, TypeError, IndexError, ValueError):
            # If extraction fails, return empty dict and use fallback coloring
            pass
    
    return network_colors


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
    threshold_str = f"_t{int(overlap_threshold * 100):03d}"
    manifest_path = subject_dir / f"parcel_split_manifest_subject{threshold_str}.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    retained_manifest = manifest[manifest["retained"] == 1].copy()

    if retained_manifest.empty:
        return []

    # Get network colors if needed
    network_colors = {}
    if color_mode in ("network", "both"):
        network_colors = extract_network_colors(template_img)

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
        threshold_str = f"_t{int(overlap_threshold * 100):03d}"
        out_path = subject_dir / f"sub-{subject}_split_parcels_combined{threshold_str}{suffix}.dlabel.nii"
        nib.save(out_img, str(out_path))
        output_paths.append(out_path)

    return output_paths


def build_group_consensus(subject_tables: list[Path], output_dir: Path) -> Path:
    tables = [pd.read_csv(p) for p in subject_tables]
    df = pd.concat(tables, ignore_index=True)

    consensus_rows = []
    for (parcel_id, parcel_name), grp in df.groupby(["parcel_id", "parcel_name"], sort=True):
        counts = grp["assigned_network_name"].value_counts()
        top_name = counts.index[0]
        top_n = int(counts.iloc[0])
        n_sub = int(len(grp))

        consensus_rows.append(
            {
                "parcel_id": int(parcel_id),
                "parcel_name": parcel_name,
                "assigned_network_name": top_name,
                "n_subjects": n_sub,
                "n_subjects_assigned_top": top_n,
                "pct_subjects_assigned_top": float(top_n / n_sub),
                "assignment_fraction_mean": float(grp["assignment_fraction"].mean()),
                "is_fpn": int(is_fpn_network_name(top_name)),
                "fpna_selected_subjects": int(grp["fpna_selected"].sum()),
                "fpnb_selected_subjects": int(grp["fpnb_selected"].sum()),
                "fpn_selected_subjects": int(grp["fpn_selected"].sum()),
            }
        )

    out = pd.DataFrame(consensus_rows).sort_values("parcel_id")
    out_path = output_dir / "parcel_network_assignment_group.csv"
    out.to_csv(out_path, index=False)
    return out_path


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
        "--dlabel-coloring",
        choices=["parcel", "network", "both"],
        default="both",
        help="Coloring mode for dlabel: 'parcel' (each parcel distinct), 'network' (by network), or 'both' (create both).",
    )
    parser.add_argument("--skip-dlabel", action="store_true", help="Skip creating combined dlabel file for wb_view visualization.")
    parser.add_argument("--dlabel-only", action="store_true", help="Only create dlabels from existing split_parcels; skip all other computations.")
    parser.add_argument("--skip-group", action="store_true", help="Skip writing group consensus output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    if not 0.0 <= args.overlap_threshold <= 1.0:
        raise ValueError("--overlap-threshold must be within [0, 1].")

    if args.subjects:
        subjects = args.subjects
    else:
        subjects = sorted([p.name.replace("sub-", "") for p in Path(args.network_label_base).glob("sub-*") if p.is_dir()])

    if not subjects:
        raise ValueError("No subjects found")

    subject_tables: list[Path] = []
    failures: list[tuple[str, str]] = []

    # Mode 1: dlabel-only (fast): just create dlabels from existing split_parcels
    if args.dlabel_only:
        for subject in subjects:
            try:
                subject_dir = Path(output_dir) / f"sub-{subject}"
                threshold_str = f"_t{int(args.overlap_threshold * 100):03d}"
                manifest_path = subject_dir / f"parcel_split_manifest_subject{threshold_str}.csv"
                
                if not manifest_path.exists():
                    raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
                
                # Load the network dlabel to get its structure
                net_img = nib.load(str(subject_network_label_path(args.network_label_base, subject)))
                net_data = net_img.get_fdata().squeeze().astype(int)
                net_names = infer_label_names_from_dlabel(net_img)
                
                # Load manifest to get network info
                manifest = pd.read_csv(manifest_path)
                retained_manifest = manifest[manifest["retained"] == 1].copy()
                
                # Load cortical data for mask reconstruction
                parcellation_cortex, unique_parcels, parcel_name_map, cortical_indices, _ = load_glasser_parcellation(args.parcellation_path)
                cortical_indices_arr = np.array(cortical_indices, dtype=bool)
                net_cortex = map_to_cortex(net_data, cortical_indices)
                
                dlabel_paths = create_subject_split_dlabel(
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
                for dlabel_path in dlabel_paths:
                    print(f"sub-{subject}: wrote combined dlabel {dlabel_path}")
                    
            except Exception as exc:
                failures.append((subject, str(exc)))
                print(f"sub-{subject}: FAILED -> {exc}")
    else:
        # Mode 2: Full pipeline with assignments and dlabels
        for subject in subjects:
            try:
                hard_path, soft_path = compute_subject_assignment(
                    subject=subject,
                    network_label_base=args.network_label_base,
                    parcelized_fpn_base=args.parcelized_fpn_base,
                    output_dir=output_dir,
                    parcellation_path=args.parcellation_path,
                    overlap_threshold=args.overlap_threshold,
                )
                subject_tables.append(hard_path)
                print(f"sub-{subject}: wrote parcel summary {hard_path}")
                print(f"sub-{subject}: wrote split manifest {soft_path}")

                # Create combined dlabel for visualization (unless skipped)
                if not args.skip_dlabel:
                    parcellation_cortex, unique_parcels, parcel_name_map, cortical_indices, _ = load_glasser_parcellation(args.parcellation_path)
                    net_img = nib.load(str(subject_network_label_path(args.network_label_base, subject)))
                    net_data = net_img.get_fdata().squeeze().astype(int)
                    net_cortex = map_to_cortex(net_data, cortical_indices)
                    net_names = infer_label_names_from_dlabel(net_img)
                    dlabel_paths = create_subject_split_dlabel(
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
                    for dlabel_path in dlabel_paths:
                        print(f"sub-{subject}: wrote combined dlabel {dlabel_path}")
            except Exception as exc:
                failures.append((subject, str(exc)))
                print(f"sub-{subject}: FAILED -> {exc}")

        if subject_tables and not args.skip_group:
            group_path = build_group_consensus(subject_tables, output_dir)
            print(f"Wrote group consensus: {group_path}")
        elif subject_tables and args.skip_group:
            print("Skipped group consensus (--skip-group).")

    if failures:
        print("\nCompleted with failures:")
        for sub, msg in failures:
            print(f"  sub-{sub}: {msg}")


if __name__ == "__main__":
    main()
