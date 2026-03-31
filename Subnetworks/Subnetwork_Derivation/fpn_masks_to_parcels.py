"""
Split individual FPN masks into Glasser parcels.

For each subject, this script:
1) Loads relabeled FPN communities (FPN_A and FPN_B) in fsLR CIFTI space.
2) Computes parcel-wise overlap fractions with HCP-MMP1 (Glasser 360).
3) Saves a CSV table with overlap stats per parcel.
4) Saves dense scalar CIFTI maps for wb_view (overlap fractions + thresholded parcel masks).
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


DEFAULT_SUBNETWORK_DIR = (
    "/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_derivation/infomap"
)
DEFAULT_OUTPUT_DIR = "/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/parcelized_fpn"
DEFAULT_PARCELLATION_PATHS = [
    "/home/hmueller2/atlases/fsLR/"
    "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii",
    "/usr/share/workbench/resources/"
    "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii",
]


def get_cortical_indices_from_cifti(img: nib.Cifti2Image) -> np.ndarray:
    """Return sorted cortical grayordinate indices from a CIFTI image."""
    brain_axis = img.header.get_axis(1)
    grayordinate_indices = np.arange(img.shape[1], dtype=int)
    cortical_indices: list[int] = []

    for bm in brain_axis.iter_structures():
        if len(bm) < 2:
            raise ValueError(f"Unexpected iter_structures entry: {bm}")
        structure_name, data_indices = bm[0], bm[1]

        if "CORTEX_LEFT" not in structure_name and "CORTEX_RIGHT" not in structure_name:
            continue

        if isinstance(data_indices, slice):
            # Some nibabel/CIFTI variants return open slices (e.g. start=None).
            # Indexing through an explicit grayordinate vector handles these safely.
            cortical_indices.extend(grayordinate_indices[data_indices].tolist())
        else:
            cortical_indices.extend(np.asarray(data_indices).tolist())

    if not cortical_indices:
        raise ValueError("No cortical structures found in CIFTI brain model axis.")

    return np.array(sorted(cortical_indices), dtype=int)


def resolve_parcellation_path(parcellation_path: str | None) -> str:
    """Resolve parcellation path from explicit arg or standard locations."""
    if parcellation_path:
        if not os.path.exists(parcellation_path):
            raise FileNotFoundError(f"Parcellation file not found: {parcellation_path}")
        return parcellation_path

    for candidate in DEFAULT_PARCELLATION_PATHS:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Glasser parcellation not found in default locations. Use --parcellation-path."
    )


def load_hcp_parcellation(
    parcellation_path: str | None,
) -> tuple[np.ndarray, np.ndarray, dict[int, str], nib.Cifti2Image, np.ndarray]:
    """Load cortical Glasser labels and metadata from a dlabel CIFTI."""
    path = resolve_parcellation_path(parcellation_path)
    img = nib.load(path)
    full_data = img.get_fdata()[0]
    cortical_indices = get_cortical_indices_from_cifti(img)
    parcellation = full_data[cortical_indices].astype(int)

    unique_parcels = np.unique(parcellation)
    unique_parcels = unique_parcels[unique_parcels > 0]

    label_axis = img.header.get_axis(0)
    label_table = label_axis.label[0]
    parcel_name_map: dict[int, str] = {}
    for parcel_id in unique_parcels:
        if int(parcel_id) in label_table:
            parcel_name_map[int(parcel_id)] = label_table[int(parcel_id)][0]
        else:
            parcel_name_map[int(parcel_id)] = f"Parcel_{int(parcel_id)}"

    return parcellation, unique_parcels, parcel_name_map, img, cortical_indices


def subject_subnetwork_path(subnetwork_dir: str, subject: str) -> str:
    """Return subject relabeled subnetwork CIFTI path (dlabel preferred)."""
    base = os.path.join(subnetwork_dir, f"sub-{subject}")
    preferred = os.path.join(base, f"{subject}_FPN_infomap_communities_kmeans_relabeled.dlabel.nii")
    fallback = os.path.join(base, f"{subject}_FPN_infomap_communities_kmeans_relabeled.dscalar.nii")
    if os.path.exists(preferred):
        return preferred
    return fallback


def load_cifti_row(path: str, row_index: int) -> tuple[np.ndarray, nib.Cifti2Image, int]:
    """Load one row from a CIFTI file (or single map), with robust single-row fallback."""
    img = nib.load(path)
    data = img.get_fdata()
    if data.ndim == 1:
        if row_index != 0:
            raise ValueError(f"Requested row {row_index} from 1D data: {path}")
        return data, img, 0
    if row_index < 0 or row_index >= data.shape[0]:
        if data.shape[0] == 1:
            print(
                f"Warning: requested --k-index {row_index} but only one map exists in {path}; using row 0."
            )
            return data[0], img, 0
        raise ValueError(f"Row index {row_index} out of range for {path} shape={data.shape}")
    return data[row_index], img, row_index


def canonicalize_label(name: str) -> str:
    return name.lower().replace("_", "").replace(" ", "").replace("-", "")


def infer_fpna_fpnb_labels(img: nib.Cifti2Image, row_index: int) -> tuple[int, int]:
    """Infer integer labels used for FPN_A and FPN_B from CIFTI label table."""
    label_a = None
    label_b = None
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


def extract_subject_fpn_masks(
    subnetwork_dir: str,
    subject: str,
    k_index: int,
    cortical_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load subject FPN labels and return cortical masks for FPN, FPN_A, FPN_B."""
    path = subject_subnetwork_path(subnetwork_dir, subject)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing relabeled FPN file for sub-{subject}: {path}")

    data, img, used_row_index = load_cifti_row(path, k_index)
    label_a, label_b = infer_fpna_fpnb_labels(img, used_row_index)

    data_i = data.astype(int)
    fpna = data_i == label_a
    fpnb = data_i == label_b
    fpn = fpna | fpnb

    if not np.any(fpna) or not np.any(fpnb):
        raise ValueError(
            f"Empty FPN_A/FPN_B mask for sub-{subject} (labels: A={label_a}, B={label_b})"
        )

    return fpn[cortical_indices], fpna[cortical_indices], fpnb[cortical_indices]


def parcel_overlap(mask: np.ndarray, parcellation: np.ndarray, unique_parcels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-parcel overlap counts, sizes, and fractions for a boolean cortical mask."""
    overlap_counts = np.zeros(len(unique_parcels), dtype=int)
    parcel_sizes = np.zeros(len(unique_parcels), dtype=int)

    for i, parcel_id in enumerate(unique_parcels):
        in_parcel = parcellation == int(parcel_id)
        parcel_sizes[i] = int(np.sum(in_parcel))
        overlap_counts[i] = int(np.sum(mask & in_parcel))

    with np.errstate(divide="ignore", invalid="ignore"):
        fractions = np.where(parcel_sizes > 0, overlap_counts / parcel_sizes, 0.0)
    return overlap_counts, parcel_sizes, fractions


def parcel_values_to_dense(
    values: np.ndarray,
    parcellation: np.ndarray,
    unique_parcels: np.ndarray,
    cortical_indices: np.ndarray,
    n_grayordinates: int,
) -> np.ndarray:
    """Expand parcel values back to dense grayordinate vector for wb_view."""
    cortical_dense = np.zeros(len(parcellation), dtype=float)
    for i, parcel_id in enumerate(unique_parcels):
        cortical_dense[parcellation == int(parcel_id)] = float(values[i])

    dense = np.zeros(n_grayordinates, dtype=float)
    dense[cortical_indices] = cortical_dense
    return dense


def save_dense_map(
    out_path: str,
    map_name: str,
    values: np.ndarray,
    parcellation: np.ndarray,
    unique_parcels: np.ndarray,
    template_img: nib.Cifti2Image,
    cortical_indices: np.ndarray,
) -> None:
    """Save a single-map dense scalar CIFTI for Workbench visualization."""
    brain_axis = template_img.header.get_axis(1)
    n_grayordinates = template_img.shape[1]
    dense = parcel_values_to_dense(
        values, parcellation, unique_parcels, cortical_indices, n_grayordinates
    )
    scalar_axis = nib.cifti2.ScalarAxis([map_name])
    header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis))
    out_img = nib.Cifti2Image(dense.reshape(1, -1), header=header)
    nib.save(out_img, out_path)

def thresholded_parcel_ids(
    selection: np.ndarray,
    unique_parcels: np.ndarray,
) -> np.ndarray:
    """Convert boolean selection array to parcel IDs (selected parcel ID, unselected 0)."""
    result = np.zeros(len(selection), dtype=float)
    for i, parcel_id in enumerate(unique_parcels):
        if selection[i]:
            result[i] = float(parcel_id)
    return result

def process_subject(
    subject: str,
    args: argparse.Namespace,
    parcellation: np.ndarray,
    unique_parcels: np.ndarray,
    parcel_name_map: dict[int, str],
    template_img: nib.Cifti2Image,
    cortical_indices: np.ndarray,
) -> None:
    """Compute parcelized subject FPN overlaps and write outputs."""
    fpn, fpna, fpnb = extract_subject_fpn_masks(
        subnetwork_dir=args.subnetwork_dir,
        subject=subject,
        k_index=args.k_index,
        cortical_indices=cortical_indices,
    )

    fpn_n, parcel_sizes, fpn_frac = parcel_overlap(fpn, parcellation, unique_parcels)
    fpna_n, _, fpna_frac = parcel_overlap(fpna, parcellation, unique_parcels)
    fpnb_n, _, fpnb_frac = parcel_overlap(fpnb, parcellation, unique_parcels)

    thr = args.overlap_threshold
    fpn_sel = fpn_frac >= thr
    fpna_sel = fpna_frac >= thr
    fpnb_sel = fpnb_frac >= thr

    subject_out = Path(args.output) / f"sub-{subject}"
    subject_out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "parcel_id": unique_parcels.astype(int),
            "parcel_name": [parcel_name_map[int(x)] for x in unique_parcels],
            "parcel_vertex_count": parcel_sizes,
            "fpn_overlap_vertices": fpn_n,
            "fpn_overlap_fraction": fpn_frac,
            "fpna_overlap_vertices": fpna_n,
            "fpna_overlap_fraction": fpna_frac,
            "fpnb_overlap_vertices": fpnb_n,
            "fpnb_overlap_fraction": fpnb_frac,
            "fpn_selected": fpn_sel.astype(int),
            "fpna_selected": fpna_sel.astype(int),
            "fpnb_selected": fpnb_sel.astype(int),
        }
    ).sort_values("fpn_overlap_fraction", ascending=False)

    csv_path = subject_out / f"sub-{subject}_fpn_parcel_overlap.csv"
    df.to_csv(csv_path, index=False)

    save_dense_map(
        out_path=str(subject_out / f"sub-{subject}_fpn_overlap_fraction.dscalar.nii"),
        map_name="FPN_overlap_fraction",
        values=fpn_frac,
        parcellation=parcellation,
        unique_parcels=unique_parcels,
        template_img=template_img,
        cortical_indices=cortical_indices,
    )
    save_dense_map(
        out_path=str(subject_out / f"sub-{subject}_fpna_overlap_fraction.dscalar.nii"),
        map_name="FPN_A_overlap_fraction",
        values=fpna_frac,
        parcellation=parcellation,
        unique_parcels=unique_parcels,
        template_img=template_img,
        cortical_indices=cortical_indices,
    )
    save_dense_map(
        out_path=str(subject_out / f"sub-{subject}_fpnb_overlap_fraction.dscalar.nii"),
        map_name="FPN_B_overlap_fraction",
        values=fpnb_frac,
        parcellation=parcellation,
        unique_parcels=unique_parcels,
        template_img=template_img,
        cortical_indices=cortical_indices,
    )
    save_dense_map(
        out_path=str(subject_out / f"sub-{subject}_fpn_parcels_thresholded.dscalar.nii"),
        map_name=f"FPN_parcel_ids_thr_{thr:.2f}",
        values=thresholded_parcel_ids(fpn_sel, unique_parcels),
        parcellation=parcellation,
        unique_parcels=unique_parcels,
        template_img=template_img,
        cortical_indices=cortical_indices,
    )
    save_dense_map(
        out_path=str(subject_out / f"sub-{subject}_fpna_parcels_thresholded.dscalar.nii"),
        map_name=f"FPN_A_parcel_ids_thr_{thr:.2f}",
        values=thresholded_parcel_ids(fpna_sel, unique_parcels),
        parcellation=parcellation,
        unique_parcels=unique_parcels,
        template_img=template_img,
        cortical_indices=cortical_indices,
    )
    save_dense_map(
        out_path=str(subject_out / f"sub-{subject}_fpnb_parcels_thresholded.dscalar.nii"),
        map_name=f"FPN_B_parcel_ids_thr_{thr:.2f}",
        values=thresholded_parcel_ids(fpnb_sel, unique_parcels),
        parcellation=parcellation,
        unique_parcels=unique_parcels,
        template_img=template_img,
        cortical_indices=cortical_indices,
    )

    print(
        f"sub-{subject}: saved overlap table and maps | "
        f"selected parcels (thr={thr:.2f}) -> FPN={int(fpn_sel.sum())}, "
        f"FPN_A={int(fpna_sel.sum())}, FPN_B={int(fpnb_sel.sum())}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Divide subject-level FPN masks into Glasser parcels (overlap fraction)."
    )
    parser.add_argument("--subject", type=str, help="Single subject ID, e.g. 01")
    parser.add_argument("--subjects", nargs="+", help="List of subject IDs")
    parser.add_argument("--all-subjects", action="store_true", help="Process all available subjects")
    parser.add_argument(
        "--subnetwork-dir",
        type=str,
        default=DEFAULT_SUBNETWORK_DIR,
        help="Base directory with sub-XX relabeled FPN files",
    )
    parser.add_argument(
        "--parcellation-path",
        type=str,
        default=None,
        help="Path to Glasser 32k fsLR dlabel file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--k-index",
        type=int,
        default=0,
        help="Row index in relabeled subnetwork file (default=0 for single-map dlabel files)",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.50,
        help="Parcel inclusion threshold based on overlap fraction [0,1]",
    )
    return parser.parse_args()


def discover_subjects(subnetwork_dir: str) -> list[str]:
    """Find subject IDs from sub-XX folders."""
    folders = sorted(glob.glob(os.path.join(subnetwork_dir, "sub-*")))
    return [os.path.basename(x).replace("sub-", "") for x in folders]


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.overlap_threshold <= 1.0:
        raise ValueError("--overlap-threshold must be within [0, 1].")

    parcellation, unique_parcels, parcel_name_map, template_img, cortical_indices = load_hcp_parcellation(
        args.parcellation_path
    )

    if args.subject:
        subjects = [args.subject]
    elif args.subjects:
        subjects = args.subjects
    else:
        subjects = discover_subjects(args.subnetwork_dir)

    if not subjects:
        raise ValueError(f"No subjects found in {args.subnetwork_dir}")

    os.makedirs(args.output, exist_ok=True)
    print(f"Processing {len(subjects)} subjects -> output: {args.output}")

    failures: list[tuple[str, str]] = []
    for subject in subjects:
        try:
            process_subject(
                subject=subject,
                args=args,
                parcellation=parcellation,
                unique_parcels=unique_parcels,
                parcel_name_map=parcel_name_map,
                template_img=template_img,
                cortical_indices=cortical_indices,
            )
        except Exception as exc:
            failures.append((subject, str(exc)))
            print(f"sub-{subject}: FAILED -> {exc}")

    if failures:
        print("\nCompleted with failures:")
        for subject, msg in failures:
            print(f"  sub-{subject}: {msg}")
    else:
        print("\nAll subjects processed successfully.")


if __name__ == "__main__":
    main()
