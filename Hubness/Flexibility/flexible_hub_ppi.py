#!/usr/bin/env python3
"""
Stage 1 of the flexible hub pipeline:
Compute task PPI effects once per subject and save compact subject-level CSVs.

Stage 2 will consume those saved files to compute flexibility metrics and plots.

Supported analysis levels:
- network_parcel: split parcels (~450, threshold-dependent)
- network: Glasser 360 parcels
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable

HUBNESS_DIR = Path(__file__).resolve().parents[1]
if str(HUBNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HUBNESS_DIR))

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import compute_regressor, run_glm
from nilearn.signal import clean
from scipy import stats

from hubness_utils import (
    discover_subjects_from_subdirs,
    ensure_dir,
    load_glasser_parcellation,
    load_split_parcel_manifest_retained,
    load_split_parcel_masks,
)

DEFAULT_FMRIPREP_BASE = "/ptmp/hmueller2/2025_ibc_latent/outputs/preprocessing/fmriprep_out"
DEFAULT_ASSIGNMENT_DIR = "/ptmp/hmueller2/2025_ibc_latent/outputs/hubness"
DEFAULT_OUTPUT_DIR = "/ptmp/hmueller2/2025_ibc_latent/outputs/hubness"
DEFAULT_PARCELLATION_PATH = None

ANALYSIS_LEVEL_NETWORK = "network"
ANALYSIS_LEVEL_NETWORK_PARCEL = "network_parcel"


def find_events_file(subject: str, session: str, task: str, direction: str | None = None, run: str | None = None) -> str | None:
    base = os.path.join("/ptmp/hmueller2/2025_ibc_latent/data/ibc_raw", f"sub-{subject}", session, "func")
    candidates: list[str] = []

    if run:
        if direction:
            candidates.append(os.path.join(base, f"sub-{subject}_{session}_task-{task}_dir-{direction}_run-{run}_events.tsv"))
        candidates.append(os.path.join(base, f"sub-{subject}_{session}_task-{task}_run-{run}_events.tsv"))
        candidates.extend(sorted(glob.glob(os.path.join(base, f"sub-{subject}_{session}_task-{task}_*run-{run}*_events.tsv"))))
    else:
        if direction:
            candidates.append(os.path.join(base, f"sub-{subject}_{session}_task-{task}_dir-{direction}_events.tsv"))
        candidates.append(os.path.join(base, f"sub-{subject}_{session}_task-{task}_events.tsv"))
        candidates.extend(sorted(glob.glob(os.path.join(base, f"sub-{subject}_{session}_task-{task}_*_events.tsv"))))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def load_motion(motion_path: Path, n_scans: int) -> np.ndarray | None:
    if not motion_path.exists():
        return None

    motion = np.loadtxt(motion_path)
    if motion.ndim == 1:
        motion = motion.reshape(-1, 6) if motion.size % 6 == 0 else motion.reshape(-1, 1)

    diff = motion.shape[0] - n_scans
    if abs(diff) > 5:
        return None
    if diff > 0:
        motion = motion[:n_scans, :]
    elif diff < 0:
        motion = np.pad(motion, ((0, abs(diff)), (0, 0)), mode="constant")

    return motion


def extract_timeseries(func_data: np.ndarray, parcel_mask: np.ndarray) -> np.ndarray:
    if np.sum(parcel_mask) == 0:
        return np.full(func_data.shape[0], np.nan)
    return np.nanmean(func_data[:, parcel_mask], axis=1)


def load_parcel_set(
    subject: str,
    analysis_level: str,
    assignment_dir: str,
    overlap_threshold: float,
    parcellation_path: str | None,
) -> tuple[np.ndarray, list[int], dict[int, str], dict[int, np.ndarray]]:
    parcellation_cortex, unique_parcels, parcel_name_map, cortical_indices, _ = load_glasser_parcellation(parcellation_path)

    if analysis_level == ANALYSIS_LEVEL_NETWORK_PARCEL:
        manifest = load_split_parcel_manifest_retained(subject, assignment_dir, overlap_threshold)
        if manifest.empty:
            raise ValueError(f"No retained split parcels found for sub-{subject}")
        manifest = manifest.sort_values("split_mask_path").reset_index(drop=True)
        parcel_masks = load_split_parcel_masks(subject, assignment_dir, overlap_threshold, cortical_indices)
        parcel_ids = list(range(1, len(manifest) + 1))
        parcel_names = {
            int(idx + 1): str(row.get("split_label", f"split_parcel_{idx + 1}"))
            for idx, (_, row) in enumerate(manifest.iterrows())
        }
        return cortical_indices, parcel_ids, parcel_names, parcel_masks

    assignment_path = Path(assignment_dir) / f"sub-{subject}" / "parcel_network_assignment_subject.csv"
    if not assignment_path.exists():
        raise FileNotFoundError(f"Missing network assignment file for sub-{subject}: {assignment_path}")

    assignment = pd.read_csv(assignment_path)
    required = {"parcel_id", "assigned_network_id", "assigned_network_name"}
    missing = required - set(assignment.columns)
    if missing:
        raise ValueError(f"Assignment file missing columns {missing}: {assignment_path}")

    parcel_masks = {int(pid): (parcellation_cortex == int(pid)) for pid in unique_parcels}
    network_rows = assignment[["assigned_network_id", "assigned_network_name"]].drop_duplicates().sort_values("assigned_network_id")
    node_ids: list[int] = []
    node_name_map: dict[int, str] = {}
    node_masks: dict[int, np.ndarray] = {}

    for _, row in network_rows.iterrows():
        network_id = int(row["assigned_network_id"])
        network_name = str(row["assigned_network_name"])
        network_parcels = assignment.loc[assignment["assigned_network_id"] == network_id, "parcel_id"].astype(int).tolist()
        mask = np.zeros_like(parcellation_cortex, dtype=bool)
        for parcel_id in network_parcels:
            mask |= parcel_masks[int(parcel_id)]
        node_ids.append(network_id)
        node_name_map[network_id] = network_name
        node_masks[network_id] = mask

    return cortical_indices, node_ids, node_name_map, node_masks


def compute_subject_ffx(group: pd.DataFrame) -> pd.Series:
    betas = group["beta"].to_numpy(dtype=float)
    variances = group["variance"].to_numpy(dtype=float)
    valid = np.isfinite(betas) & np.isfinite(variances) & (variances > 0)
    betas = betas[valid]
    variances = variances[valid]

    if len(betas) == 0:
        return pd.Series({"beta_ffx": np.nan, "se_ffx": np.nan, "n_runs": 0})
    if len(betas) == 1:
        return pd.Series({"beta_ffx": float(betas[0]), "se_ffx": float(np.sqrt(variances[0])), "n_runs": 1})

    weights = 1.0 / variances
    beta_ffx = float(np.sum(betas * weights) / np.sum(weights))
    var_ffx = float(1.0 / np.sum(weights))
    return pd.Series({"beta_ffx": beta_ffx, "se_ffx": float(np.sqrt(var_ffx)), "n_runs": len(betas)})


def compute_rows_for_func_file(
    subject: str,
    session: str,
    func_path: str,
    task: str,
    direction: str,
    run: str | None,
    cortical_indices: np.ndarray,
    parcel_ids: list[int],
    parcel_name_map: dict[int, str],
    parcel_masks: dict[int, np.ndarray],
) -> list[dict[str, object]]:
    func_path_obj = Path(func_path)
    func_img = nib.load(str(func_path_obj))
    func_data_full = func_img.get_fdata()
    n_scans = func_data_full.shape[0]
    func_data = func_data_full[:, cortical_indices]

    ax0 = func_img.header.get_axis(0)
    ax1 = func_img.header.get_axis(1)
    ts_axis = ax0 if isinstance(ax0, nib.cifti2.SeriesAxis) else ax1
    tr = float(getattr(ts_axis, "step", 2.0))
    frame_times = np.arange(n_scans) * tr

    motion_fname = f"sub-{subject}_{session}_task-{task}_dir-{direction}" + (f"_run-{run}" if run else "") + "_motion.txt"
    motion_path = func_path_obj.parents[1] / "regressors" / motion_fname
    motion = load_motion(motion_path, n_scans)
    if motion is None:
        return []

    onset_path = find_events_file(subject, session, task, direction=direction, run=run)
    if onset_path is None:
        return []

    events = pd.read_csv(onset_path, sep="\t")
    events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
    events["duration"] = pd.to_numeric(events["duration"], errors="coerce")
    events = events.dropna(subset=["onset", "duration"]).reset_index(drop=True)
    conditions = [cond for cond, count in events["trial_type"].value_counts().items() if count >= 2]
    if not conditions:
        return []

    parcel_ts_dict: dict[int, np.ndarray] = {}
    for parcel_id in parcel_ids:
        parcel_mask = parcel_masks[int(parcel_id)]
        if len(parcel_mask) != func_data.shape[1]:
            continue
        ts_raw = extract_timeseries(func_data, parcel_mask)
        if not np.isfinite(ts_raw).all():
            continue
        ts = clean(ts_raw.reshape(-1, 1), detrend=True, standardize="zscore_sample", confounds=None).ravel()
        if np.isfinite(ts).all() and np.std(ts) > 1e-8:
            parcel_ts_dict[int(parcel_id)] = ts

    if len(parcel_ts_dict) < 2:
        return []

    rows: list[dict[str, object]] = []
    for seed_id in sorted(parcel_ts_dict.keys()):
        seed_ts = parcel_ts_dict[seed_id]
        for target_id in sorted(parcel_ts_dict.keys()):
            if target_id == seed_id:
                continue

            target_ts = parcel_ts_dict[target_id]
            design_dict: dict[str, np.ndarray] = {"physio": seed_ts}
            for i in range(motion.shape[1]):
                design_dict[f"motion_{i}"] = motion[:, i]

            for cond in conditions:
                cond_mask = events["trial_type"] == cond
                onsets = events.loc[cond_mask, "onset"].to_numpy(dtype=np.float64)
                durations = events.loc[cond_mask, "duration"].to_numpy(dtype=np.float64)
                if len(onsets) == 0:
                    continue

                psych, _ = compute_regressor((onsets, durations, np.ones(len(onsets))), hrf_model="spm", frame_times=frame_times)
                psych = psych[:, 0] - psych[:, 0].mean()
                if np.allclose(psych.std(), 0, atol=1e-8) or not np.isfinite(psych).all():
                    continue

                ppi = seed_ts * psych
                if np.var(ppi) < 1e-6:
                    continue

                design_dict[f"psych_{cond}"] = psych
                design_dict[f"ppi_{cond}"] = ppi

            if not any(key.startswith("ppi_") for key in design_dict):
                continue

            design_matrix = pd.DataFrame(design_dict)
            stds = design_matrix.std(axis=0, ddof=0)
            keep_cols = stds[stds > 1e-8].index.tolist()
            design_matrix = design_matrix[keep_cols]
            design_matrix["constant"] = 1.0

            X = design_matrix.values.astype(np.float64)
            y = target_ts.astype(np.float64).reshape(-1)
            rank = np.linalg.matrix_rank(X)
            df_res = X.shape[0] - rank
            if df_res <= 0:
                continue

            labels, results = run_glm(y.reshape(-1, 1), X, noise_model="ar1", bins=100)
            glm_result = results[labels[0]]
            theta = np.asarray(glm_result.theta, dtype=np.float64).reshape(-1)
            residuals = y - (X @ theta)
            sigma2 = float((residuals @ residuals) / df_res)
            XtX_pinv = np.linalg.pinv(X.T @ X)
            col_names = design_matrix.columns.tolist()

            for cond in conditions:
                ppi_col = f"ppi_{cond}"
                if ppi_col not in col_names:
                    continue
                idx = col_names.index(ppi_col)
                beta = float(theta[idx])
                variance = float(sigma2 * XtX_pinv[idx, idx])
                se = np.sqrt(variance) if variance > 0 else np.inf
                tstat = beta / se if np.isfinite(se) and se > 0 else 0.0
                pval = 2 * (1 - stats.t.cdf(abs(tstat), df_res)) if np.isfinite(tstat) else 1.0

                rows.append(
                    {
                        "subject": subject,
                        "session": session,
                        "run_id": f"task-{task}_dir-{direction}_run-{run}" if run else f"task-{task}_dir-{direction}",
                        "task": task,
                        "condition": cond,
                        "task_condition": f"{task}::{cond}",
                        "seed_id": int(seed_id),
                        "target_id": int(target_id),
                        "seed_name": parcel_name_map.get(int(seed_id), f"PARCEL_{int(seed_id)}"),
                        "target_name": parcel_name_map.get(int(target_id), f"PARCEL_{int(target_id)}"),
                        "beta": beta,
                        "variance": variance,
                        "tstat": float(tstat),
                        "pval": float(pval),
                    }
                )

    print(
        f"sub-{subject}: completed {session} {task} {direction} run-{run or 'none'} "
        f"nodes={len(parcel_ts_dict)} pairs={len(rows)}",
        flush=True,
    )
    return rows


def compute_subject(subject: str, args: argparse.Namespace, output_dir: Path) -> None:
    subject_root = Path(args.fmriprep_base) / f"sub-{subject}"
    sessions = sorted(glob.glob(str(subject_root / "ses-*")))
    if not sessions:
        raise FileNotFoundError(f"No sessions found for sub-{subject}")

    cortical_indices, parcel_ids, parcel_name_map, parcel_masks = load_parcel_set(
        subject=subject,
        analysis_level=args.analysis_level,
        assignment_dir=args.assignment_dir,
        overlap_threshold=args.overlap_threshold,
        parcellation_path=args.parcellation_path,
    )

    func_jobs: list[tuple[str, str, str, str, str, str | None]] = []

    for ses_path in sessions:
        session = os.path.basename(ses_path)
        glm_dir = Path(ses_path) / "postfmriprep" / "GLM"
        if not glm_dir.exists():
            continue

        func_files = sorted(glm_dir.glob("sub-*_task-*_dir-*_*cleaned_noscrub.dtseries.nii"))
        for func_path in func_files:
            parts = func_path.name.split("_")
            task_matches = [p.split("-")[1] for p in parts if p.startswith("task-")]
            dir_matches = [p.split("-")[1] for p in parts if p.startswith("dir-")]
            run_matches = [p.split("-")[1] for p in parts if p.startswith("run-")]
            if not task_matches or not dir_matches:
                continue
            task = task_matches[0]
            direction = dir_matches[0]
            run = run_matches[0] if run_matches else None
            func_jobs.append((subject, session, str(func_path), task, direction, run))

    if not func_jobs:
        raise RuntimeError(f"No task runs found for sub-{subject}")

    n_jobs = max(1, min(int(args.n_jobs), len(func_jobs)))
    all_rows: list[dict[str, object]] = []
    if n_jobs == 1:
        for job in func_jobs:
            all_rows.extend(
                compute_rows_for_func_file(
                    subject=job[0],
                    session=job[1],
                    func_path=job[2],
                    task=job[3],
                    direction=job[4],
                    run=job[5],
                    cortical_indices=cortical_indices,
                    parcel_ids=parcel_ids,
                    parcel_name_map=parcel_name_map,
                    parcel_masks=parcel_masks,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    compute_rows_for_func_file,
                    job[0],
                    job[1],
                    job[2],
                    job[3],
                    job[4],
                    job[5],
                    cortical_indices,
                    parcel_ids,
                    parcel_name_map,
                    parcel_masks,
                )
                for job in func_jobs
            ]
            for future in futures:
                all_rows.extend(future.result())

    if not all_rows:
        raise RuntimeError(f"No PPI rows computed for sub-{subject}")

    df = pd.DataFrame(all_rows)
    subject_dir = ensure_dir(output_dir / f"sub-{subject}" / "flexible" / "ppi")
    results_path = subject_dir / f"flexible_hub_ppi_{args.analysis_level}_results.csv"
    df.to_csv(results_path, index=False)

    ffx = (
        df.groupby(
            ["subject", "task", "condition", "task_condition", "seed_id", "target_id", "seed_name", "target_name"],
            as_index=False,
        )
        .apply(compute_subject_ffx, include_groups=False)
        .reset_index(drop=True)
    )
    ffx["analysis_level"] = args.analysis_level
    ffx.to_csv(subject_dir / f"flexible_hub_ppi_{args.analysis_level}_ffx.csv", index=False)
    print(f"sub-{subject}: wrote {results_path.name} and flexible_hub_ppi_{args.analysis_level}_ffx.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: compute and save flexible hub PPI outputs.")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--analysis-level", choices=[ANALYSIS_LEVEL_NETWORK, ANALYSIS_LEVEL_NETWORK_PARCEL], default=ANALYSIS_LEVEL_NETWORK)
    parser.add_argument("--fmriprep-base", default=DEFAULT_FMRIPREP_BASE)
    parser.add_argument("--assignment-dir", default=DEFAULT_ASSIGNMENT_DIR)
    parser.add_argument("--network-label-base", default=DEFAULT_OUTPUT_DIR, help="Accepted for launcher compatibility; not used in Stage 1.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--parcellation-path", default=DEFAULT_PARCELLATION_PATH)
    parser.add_argument("--overlap-threshold", type=float, default=0.30)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=int(os.environ.get("N_JOBS", os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
        help="Number of worker processes to use within each subject.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    if args.subjects:
        subjects = [str(s).replace("sub-", "") for s in args.subjects]
    else:
        subjects = discover_subjects_from_subdirs(args.fmriprep_base)

    if not subjects:
        raise ValueError("No subjects found")

    failures: list[tuple[str, str]] = []
    for subject in subjects:
        try:
            compute_subject(subject, args, output_dir)
        except Exception as exc:
            failures.append((subject, str(exc)))
            print(f"sub-{subject}: FAILED -> {exc}")

    if failures:
        print("\nCompleted with failures:")
        for sub, msg in failures:
            print(f"  sub-{sub}: {msg}")


if __name__ == "__main__":
    main()
