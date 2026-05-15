from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd

DEFAULT_PARCELLATION_PATHS = [
    "/home/hmueller2/atlases/fsLR/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii",
    "/usr/share/workbench/resources/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii",
]
NETWORK_DLABEL_NAME = "Bipartite_PhysicalCommunities+AlgorithmicLabeling.dlabel.nii"


def canonicalize_label(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "").replace("/", "").replace(" ", "")


def resolve_parcellation_path(parcellation_path: str | None = None) -> str:
    if parcellation_path:
        if not os.path.exists(parcellation_path):
            raise FileNotFoundError(f"Parcellation not found: {parcellation_path}")
        return parcellation_path

    for candidate in DEFAULT_PARCELLATION_PATHS:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError("Could not resolve Glasser parcellation path")


def get_cortical_indices_from_cifti(img: nib.Cifti2Image) -> np.ndarray:
    brain_axis = img.header.get_axis(1)
    grayordinate_indices = np.arange(img.shape[1], dtype=int)
    cortical_indices: list[int] = []

    for bm in brain_axis.iter_structures():
        if len(bm) < 2:
            continue
        structure_name, data_indices = bm[0], bm[1]
        if "CORTEX_LEFT" not in structure_name and "CORTEX_RIGHT" not in structure_name:
            continue

        if isinstance(data_indices, slice):
            cortical_indices.extend(grayordinate_indices[data_indices].tolist())
        else:
            cortical_indices.extend(np.asarray(data_indices).tolist())

    if not cortical_indices:
        raise ValueError("No cortical indices found in CIFTI brain models")

    return np.array(sorted(cortical_indices), dtype=int)


def load_glasser_parcellation(parcellation_path: str | None = None) -> tuple[np.ndarray, np.ndarray, dict[int, str], np.ndarray, nib.Cifti2Image]:
    path = resolve_parcellation_path(parcellation_path)
    img = nib.load(path)
    cortical_indices = get_cortical_indices_from_cifti(img)
    full_data = img.get_fdata()[0].astype(int)
    parcellation_cortex = full_data[cortical_indices]

    unique_parcels = np.unique(parcellation_cortex)
    unique_parcels = unique_parcels[unique_parcels > 0]

    label_axis = img.header.get_axis(0)
    label_table = label_axis.label[0]
    parcel_name_map: dict[int, str] = {}
    for parcel_id in unique_parcels:
        parcel_name_map[int(parcel_id)] = label_table.get(int(parcel_id), (f"PARCEL_{int(parcel_id)}", None))[0]

    return parcellation_cortex, unique_parcels.astype(int), parcel_name_map, cortical_indices, img


def infer_label_names_from_dlabel(label_img: nib.Cifti2Image) -> dict[int, str]:
    label_axis = label_img.header.get_axis(0)
    label_table = label_axis.label[0]
    names = {}
    for key, value in label_table.items():
        if isinstance(value, tuple) and len(value) > 0:
            names[int(key)] = str(value[0])
        else:
            names[int(key)] = str(value)
    return names


def extract_network_colors_from_dlabel(label_img: nib.Cifti2Image) -> dict[int, tuple[float, float, float, float]]:
    """Extract RGBA colors from a dlabel label table.

    Returns a mapping of network_id -> (R, G, B, A) with values normalized to [0, 1].
    """
    network_colors: dict[int, tuple[float, float, float, float]] = {}
    first_axis = label_img.header.get_axis(0)

    if not hasattr(first_axis, "label") or first_axis.label is None:
        return network_colors

    try:
        label_dict = first_axis.label
        if isinstance(label_dict, np.ndarray) and label_dict.dtype == object:
            label_dict = label_dict[0]

        if not isinstance(label_dict, dict):
            return network_colors

        for network_id, label_data in label_dict.items():
            if not isinstance(label_data, tuple) or len(label_data) < 2:
                continue
            rgba = label_data[1]
            if not isinstance(rgba, (tuple, list)) or len(rgba) != 4:
                continue

            rgba_float = tuple(float(c) for c in rgba)
            if max(rgba_float) > 1.0:
                rgba_float = tuple(c / 255.0 for c in rgba_float)
            network_colors[int(network_id)] = rgba_float
    except (AttributeError, TypeError, IndexError, ValueError):
        return {}

    return network_colors


def map_to_cortex(data_1d: np.ndarray, cortical_indices: np.ndarray) -> np.ndarray:
    if data_1d.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape={data_1d.shape}")

    if len(data_1d) == len(cortical_indices):
        return data_1d

    max_idx = int(cortical_indices.max())
    if len(data_1d) > max_idx:
        return data_1d[cortical_indices]

    raise ValueError(
        f"Could not map array of length {len(data_1d)} to cortex length {len(cortical_indices)}"
    )


def load_cifti_row(path: str | Path, row_index: int = 0) -> tuple[np.ndarray, nib.Cifti2Image]:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata())

    if data.ndim == 1:
        if row_index != 0:
            raise ValueError(f"Requested row {row_index} from 1D CIFTI data in {path}")
        row = data
    elif data.ndim == 2:
        if row_index < 0 or row_index >= data.shape[0]:
            raise ValueError(f"Row index {row_index} out of range for shape {data.shape} in {path}")
        row = data[row_index]
    else:
        raise ValueError(f"Unsupported CIFTI data shape {data.shape} for {path}")

    return np.asarray(row), img


def build_binary_mask(
    data_1d: np.ndarray,
    include_values: Iterable[int] | None = None,
    positive_as_true: bool = True,
) -> np.ndarray:
    if include_values is not None:
        include_array = np.array(list(include_values), dtype=int)
        return np.isin(data_1d.astype(int), include_array)

    if positive_as_true:
        return np.asarray(data_1d > 0)

    return np.asarray(data_1d != 0)


def calculate_cifti_overlap_mask(
    data_a: np.ndarray,
    data_b: np.ndarray,
    include_values_a: Iterable[int] | None = None,
    include_values_b: Iterable[int] | None = None,
) -> np.ndarray:
    mask_a = build_binary_mask(data_a, include_values=include_values_a)
    mask_b = build_binary_mask(data_b, include_values=include_values_b)
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Mask shape mismatch: {mask_a.shape} vs {mask_b.shape}")
    return mask_a & mask_b


def subject_mask_count_map(masks: list[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("No masks provided")

    first_shape = masks[0].shape
    for idx, mask in enumerate(masks, start=1):
        if mask.shape != first_shape:
            raise ValueError(
                f"Mask {idx} has shape {mask.shape}, expected {first_shape}"
            )

    stack = np.stack([np.asarray(mask, dtype=np.uint8) for mask in masks], axis=0)
    return np.sum(stack, axis=0).astype(np.int16)


def count_to_fraction_map(count_map: np.ndarray, n_subjects: int) -> np.ndarray:
    if n_subjects <= 0:
        raise ValueError("n_subjects must be > 0")
    return np.asarray(count_map, dtype=np.float32) / float(n_subjects)


def save_dscalar_maps(path: str | Path, maps_2d: np.ndarray, map_names: list[str], template_img: nib.Cifti2Image) -> None:
    maps_2d = np.asarray(maps_2d)
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
    nib.save(out_img, str(path))


def is_fpn_network_name(name: str) -> bool:
    n = canonicalize_label(name)
    return "frontoparietal" in n or n.startswith("fpn") or "frontpar" in n


def parcel_vertex_masks(parcellation_cortex: np.ndarray, parcel_ids: Iterable[int]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for parcel_id in parcel_ids:
        out[int(parcel_id)] = parcellation_cortex == int(parcel_id)
    return out


def compute_participation_coefficient(weight_matrix: np.ndarray, modules: np.ndarray) -> np.ndarray:
    if weight_matrix.ndim != 2 or weight_matrix.shape[0] != weight_matrix.shape[1]:
        raise ValueError("weight_matrix must be square")
    if weight_matrix.shape[0] != len(modules):
        raise ValueError("modules length must match matrix size")

    w = np.array(weight_matrix, dtype=float, copy=True)
    w[~np.isfinite(w)] = 0.0
    np.fill_diagonal(w, 0.0)

    # Keep positive edges for PC parity with common FC hubness practice.
    w[w < 0] = 0.0

    k_total = np.sum(w, axis=1)
    pc = np.zeros(len(modules), dtype=float)

    valid_module_ids = [m for m in np.unique(modules) if pd.notna(m)]
    for i in range(w.shape[0]):
        if k_total[i] <= 0:
            pc[i] = 0.0
            continue

        module_sum = 0.0
        for module_id in valid_module_ids:
            mask = modules == module_id
            k_im = np.sum(w[i, mask])
            module_sum += (k_im / k_total[i]) ** 2
        pc[i] = 1.0 - module_sum

    return pc


def zscore_columns(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    z = (x - mu) / sd
    z[~np.isfinite(z)] = 0.0
    return z


def discover_subjects_from_subdirs(base: str) -> list[str]:
    return sorted([p.name.replace("sub-", "") for p in Path(base).glob("sub-*") if p.is_dir()])


def sanitize_for_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def subject_network_label_path(network_label_base: str, subject: str, network_dlabel_name: str = NETWORK_DLABEL_NAME) -> Path:
    return Path(network_label_base) / f"sub-{subject}" / "resting_state" / network_dlabel_name


def load_subject_fpna_fpnb_flags(subject: str, parcelized_fpn_base: str) -> pd.DataFrame:
    overlap_path = Path(parcelized_fpn_base) / f"sub-{subject}" / f"sub-{subject}_fpn_parcel_overlap.csv"
    if not overlap_path.exists():
        return pd.DataFrame(columns=["parcel_id", "fpna_selected", "fpnb_selected", "fpn_selected"])

    df = pd.read_csv(overlap_path)
    keep = [c for c in ["parcel_id", "fpna_selected", "fpnb_selected", "fpn_selected"] if c in df.columns]
    return df[keep].copy() if keep else pd.DataFrame(columns=["parcel_id", "fpna_selected", "fpnb_selected", "fpn_selected"])


def find_rest_runs(subject: str, fmriprep_base: str) -> list[Path]:
    base = Path(fmriprep_base) / f"sub-{subject}"
    patterns = [
        "ses-*/postfmriprep/GLM/*task-RestingState*cleaned_noscrub.dtseries.nii",
        "ses-*/postfmriprep/GLM/*task-RestingState*cleaned.dtseries.nii",
    ]
    files: list[Path] = []
    for pat in patterns:
        files.extend([Path(p) for p in glob.glob(str(base / pat))])
    return sorted(set(files))


def split_threshold_suffix(overlap_threshold: float) -> str:
    return f"t{int(overlap_threshold * 100):03d}"


def split_threshold_tag(overlap_threshold: float) -> str:
    return f"_{split_threshold_suffix(overlap_threshold)}"


def split_manifest_filename(overlap_threshold: float) -> str:
    return f"parcel_split_manifest_subject{split_threshold_tag(overlap_threshold)}.csv"


def split_manifest_path(base_dir: str | Path, subject: str, overlap_threshold: float) -> Path:
    return Path(base_dir) / f"sub-{subject}" / split_manifest_filename(overlap_threshold)


def split_parcels_dir(base_dir: str | Path, subject: str, overlap_threshold: float) -> Path:
    return Path(base_dir) / f"sub-{subject}" / f"split_parcels{split_threshold_tag(overlap_threshold)}"


def load_split_parcel_manifest_retained(subject: str, assignment_dir: str, overlap_threshold: float = 0.30) -> pd.DataFrame:
    manifest_path = split_manifest_path(assignment_dir, subject, overlap_threshold)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Split parcel manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    retained = df[df["retained"] == 1].copy() if "retained" in df.columns else df
    return retained.reset_index(drop=True)


def load_split_parcel_masks(subject: str, assignment_dir: str, overlap_threshold: float = 0.30, cortical_indices: np.ndarray | None = None) -> dict[int, np.ndarray]:
    split_dir = split_parcels_dir(assignment_dir, subject, overlap_threshold)

    if not split_dir.exists():
        raise FileNotFoundError(f"Split parcels directory not found: {split_dir}")

    parcel_masks: dict[int, np.ndarray] = {}
    mask_files = sorted(split_dir.glob("*.dscalar.nii"))

    for mask_file in mask_files:
        img = nib.load(str(mask_file))
        data = img.get_fdata()

        if data.ndim == 1:
            mask = data > 0
        elif data.ndim == 2:
            mask = (data > 0).any(axis=0)
        else:
            continue

        if cortical_indices is not None and len(mask) > len(cortical_indices):
            mask = mask[cortical_indices]

        label_id = len(parcel_masks) + 1
        parcel_masks[label_id] = mask.astype(bool)

    if not parcel_masks:
        raise ValueError(f"No valid parcel masks found in {split_dir}")

    return parcel_masks


def aggregate_parcel_to_network_with_soft_weights(
    parcel_metric_df: pd.DataFrame,
    soft_assignments_df: pd.DataFrame,
    hard_assignments_df: pd.DataFrame,
    parcel_col: str = "parcel_id",
    metric_col: str = "metric_value",
    network_col: str = "network_id",
    weight_col: str = "overlap_weight",
) -> pd.DataFrame:
    """
    Aggregate parcel-level metrics to networks using soft assignment weights.
    
    Args:
        parcel_metric_df: DataFrame with parcel metrics, must have parcel_col and metric_col
        soft_assignments_df: DataFrame with soft assignments (parcel_col, network_col, weight_col)
        hard_assignments_df: DataFrame with hard assignments for fallback (parcel_col, "assigned_network_id")
        parcel_col: Name of parcel ID column
        metric_col: Name of metric column to aggregate
        network_col: Name of network ID column
        weight_col: Name of weight column
    
    Returns:
        DataFrame with aggregated metrics per network
    """
    # Merge soft assignments with parcel metrics
    merged = parcel_metric_df.merge(
        soft_assignments_df[[parcel_col, network_col, weight_col]],
        on=parcel_col,
        how="left"
    )
    
    # For parcels without soft assignments, use hard assignment
    unassigned_mask = merged[network_col].isna()
    if unassigned_mask.any():
        hard_subset = hard_assignments_df[[parcel_col, "assigned_network_id"]].drop_duplicates()
        unassigned_parcels = merged.loc[unassigned_mask, parcel_col].unique()
        hard_for_unassigned = hard_subset[hard_subset[parcel_col].isin(unassigned_parcels)]
        
        for _, row in hard_for_unassigned.iterrows():
            pid = row[parcel_col]
            nid = row["assigned_network_id"]
            idx = merged[(merged[parcel_col] == pid) & merged[network_col].isna()].index
            merged.loc[idx, network_col] = nid
            merged.loc[idx, weight_col] = 1.0
    
    # Fill any remaining NaNs
    merged = merged.dropna(subset=[network_col, metric_col, weight_col])
    
    # Weighted aggregation per network
    aggregated = (
        merged.groupby(network_col, as_index=False)
        .apply(
            lambda group: pd.Series({
                f"{metric_col}_network": float(
                    np.sum(group[metric_col].values * group[weight_col].values) / group[weight_col].sum()
                    if group[weight_col].sum() > 0 else np.nan
                ),
                "n_parcels_contributing": int(len(group)),
                "weight_sum": float(group[weight_col].sum()),
            }),
            include_groups=False
        )
        .reset_index(drop=True)
    )
    
    return aggregated


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
