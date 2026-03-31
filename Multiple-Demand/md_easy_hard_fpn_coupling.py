"""
Compute Easy vs Hard FPN activation and DMN/DAN coupling for MD tasks.

For the four MD tasks used in md_mapping.py:
    - HcpWm: 2back-0back
    - Stroop: incongruent-congruent
    - Catell: hard-easy
    - Attention: double_incongruent-double_congruent

Outputs:
1) Per-subject CSV with task-level Easy/Hard activation and coupling metrics.
2) Group summary CSV averaged across subjects.

Activation metrics:
    mean activation in whole FPN, FPN_A, and FPN_B.

Coupling metrics:
    Pearson correlation between seed and target mean timeseries during each condition.
    Seeds: whole FPN, FPN_A, FPN_B
    Targets: DMN, DAN
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


DEFAULT_CONTRAST_BASE = "/ptmp/hmueller2/2025_ibc_latent/outputs/glm/contrast_maps_fsLR"
DEFAULT_DTSERIES_BASE = "/ptmp/hmueller2/2025_ibc_latent/outputs/preprocessing/fmriprep_out"
DEFAULT_EVENTS_BASE = "/ptmp/hmueller2/2025_ibc_latent/data/ibc_raw"
DEFAULT_SUBNETWORK_DIR = "/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_derivation/infomap"
DEFAULT_NETWORK_LABEL_BASE = "/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks/derived_networks"
DEFAULT_ALL_CONTRASTS = "/home/hmueller2/ibc_code/ibc_latent/Data Info/all_contrasts.tsv"
DEFAULT_OUTPUT = "/ptmp/hmueller2/2025_ibc_latent/outputs/md_system/vertex_wise/md_easy_hard_fpn_coupling"

MD_TASK_CONTRASTS = {
    "HcpWm": "2back-0back",
    "Stroop": "incongruent-congruent",
    "Catell": "hard-easy",
    "Attention": "double_incongruent-double_congruent",
}

DMN_LABELS = {1, 2, 3, 4}
DAN_LABELS = {10, 11}


def sanitize_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum() or ch == "_")


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = float(np.mean(x))
    s = float(np.std(x))
    if s <= 0:
        return np.zeros_like(x)
    return (x - m) / s


def safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return float("nan")
    a_z = zscore_1d(a)
    b_z = zscore_1d(b)
    if np.std(a_z) <= 0 or np.std(b_z) <= 0:
        return float("nan")
    return float(np.corrcoef(a_z, b_z)[0, 1])


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


def subject_subnetwork_path(subnetwork_dir: str, subject: str) -> str:
    base = os.path.join(subnetwork_dir, f"sub-{subject}")
    preferred = os.path.join(base, f"{subject}_FPN_infomap_communities_kmeans_relabeled.dlabel.nii")
    fallback = os.path.join(base, f"{subject}_FPN_infomap_communities_kmeans_relabeled.dscalar.nii")
    if os.path.exists(preferred):
        return preferred
    return fallback


def canonicalize_label(name: str) -> str:
    return name.lower().replace("_", "").replace(" ", "").replace("-", "")


def infer_fpna_fpnb_labels(img: nib.Cifti2Image, row_index: int) -> tuple[int, int]:
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


def load_fpn_masks(subnetwork_dir: str, subject: str, k_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = subject_subnetwork_path(subnetwork_dir, subject)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing subnetwork file for sub-{subject}: {path}")
    data, img = load_cifti_row(path, k_index)
    label_a, label_b = infer_fpna_fpnb_labels(img, k_index)
    fpna = data == label_a
    fpnb = data == label_b
    fpn = fpna | fpnb
    if not np.any(fpna) or not np.any(fpnb):
        raise ValueError(f"Empty FPNA/FPNB mask for sub-{subject} (labels: A={label_a}, B={label_b})")
    return fpn, fpna, fpnb


def load_dmn_dan_masks(network_label_base: str, subject: str) -> tuple[np.ndarray, np.ndarray]:
    path = os.path.join(
        network_label_base,
        f"sub-{subject}",
        "resting_state",
        "Bipartite_PhysicalCommunities+AlgorithmicLabeling.dlabel.nii",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing DMN/DAN label file for sub-{subject}: {path}")
    labels, _ = load_cifti_vector(path)
    labels_i = labels.astype(int)
    dmn = np.isin(labels_i, list(DMN_LABELS))
    dan = np.isin(labels_i, list(DAN_LABELS))
    if not np.any(dmn) or not np.any(dan):
        raise ValueError(f"Empty DMN or DAN mask for sub-{subject}")
    return dmn, dan


@dataclass(frozen=True)
class TaskDifficulty:
    task: str
    contrast: str
    hard_token: str
    easy_token: str


def build_task_difficulty_map(all_contrasts_tsv: str) -> dict[str, TaskDifficulty]:
    table = pd.read_csv(all_contrasts_tsv, sep="\t")
    out: dict[str, TaskDifficulty] = {}

    for task, contrast in MD_TASK_CONTRASTS.items():
        rows = table[(table["task"] == task) & (table["contrast"] == contrast)]
        if rows.empty:
            hard_raw, easy_raw = contrast.split("-", maxsplit=1)
        else:
            # Use contrast string split as canonical file token source.
            hard_raw, easy_raw = contrast.split("-", maxsplit=1)
        out[task] = TaskDifficulty(
            task=task,
            contrast=contrast,
            hard_token=sanitize_token(hard_raw),
            easy_token=sanitize_token(easy_raw),
        )
    return out


def find_condition_maps(task_z_dir: str, condition_token: str) -> list[str]:
    if not os.path.isdir(task_z_dir):
        return []

    files = sorted(str(p) for p in Path(task_z_dir).glob("*.dscalar.nii"))
    token = sanitize_token(condition_token)

    # Exact name first: <token>.dscalar.nii
    exact = [f for f in files if sanitize_token(Path(f).stem) == token]
    if exact:
        return exact

    # Prefix match for cases like 2back_face / 0back_tools
    pref = [f for f in files if sanitize_token(Path(f).stem).startswith(f"{token}_")]
    if pref:
        return pref

    # Last fallback: loose token containment.
    loose = [f for f in files if token in sanitize_token(Path(f).stem)]
    return loose


def aggregate_condition_map(paths: list[str]) -> np.ndarray | None:
    if not paths:
        return None
    maps = []
    for p in paths:
        vec, _ = load_cifti_vector(p)
        maps.append(vec.astype(np.float64))
    return np.mean(np.stack(maps, axis=0), axis=0)


def detect_subjects(contrast_base: str, subnetwork_dir: str, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    subjects = []
    for p in sorted(Path(contrast_base).glob("sub-*")):
        sid = p.name.replace("sub-", "")
        subnet = subject_subnetwork_path(subnetwork_dir, sid)
        if os.path.exists(subnet):
            subjects.append(sid)
    return subjects


def find_run_files(dtseries_base: str, subject: str, task: str) -> list[str]:
    pattern = os.path.join(
        dtseries_base,
        f"sub-{subject}",
        "ses-*",
        "postfmriprep",
        "GLM",
        f"sub-{subject}_ses-*_task-{task}_dir-*_*cleaned_noscrub.dtseries.nii",
    )
    return sorted(glob.glob(pattern))


def parse_run_parts(run_path: str) -> tuple[str, str, str, str | None]:
    name = os.path.basename(run_path)
    parts = name.split("_")
    session = next(p for p in parts if p.startswith("ses-")).split("-", 1)[1]
    task = next(p for p in parts if p.startswith("task-")).split("-", 1)[1]
    direction = next(p for p in parts if p.startswith("dir-")).split("-", 1)[1]
    run_parts = [p for p in parts if p.startswith("run-")]
    run = run_parts[0].split("-", 1)[1] if run_parts else None
    return session, task, direction, run


def find_events_file(events_base: str, subject: str, session: str, task: str, direction: str, run: str | None) -> str | None:
    base = os.path.join(events_base, f"sub-{subject}", f"ses-{session}", "func")
    candidates = []
    if run:
        candidates.append(os.path.join(base, f"sub-{subject}_ses-{session}_task-{task}_dir-{direction}_run-{run}_events.tsv"))
        candidates.append(os.path.join(base, f"sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv"))
        candidates.extend(sorted(glob.glob(os.path.join(base, f"sub-{subject}_ses-{session}_task-{task}_*run-{run}*_events.tsv"))))
    else:
        candidates.append(os.path.join(base, f"sub-{subject}_ses-{session}_task-{task}_dir-{direction}_events.tsv"))
        candidates.append(os.path.join(base, f"sub-{subject}_ses-{session}_task-{task}_events.tsv"))
        candidates.extend(sorted(glob.glob(os.path.join(base, f"sub-{subject}_ses-{session}_task-{task}_*_events.tsv"))))

    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def get_tr_seconds(img: nib.Cifti2Image) -> float:
    ax0 = img.header.get_axis(0)
    ax1 = img.header.get_axis(1)
    if isinstance(ax0, nib.cifti2.SeriesAxis):
        return float(ax0.step)
    if isinstance(ax1, nib.cifti2.SeriesAxis):
        return float(ax1.step)
    return 2.0


def condition_event_selector(task: str, token: str):
    if task == "HcpWm":
        # Match classes like 0back_face, 2back_tools, etc.
        return lambda trial_type: sanitize_token(str(trial_type)).startswith(f"{token}_")
    return lambda trial_type: sanitize_token(str(trial_type)) == token


def event_mask_from_tsv(events_path: str, task: str, token: str, n_scans: int, tr: float) -> np.ndarray:
    events = pd.read_csv(events_path, sep="\t")
    if "trial_type" not in events.columns:
        return np.zeros(n_scans, dtype=bool)

    events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
    events["duration"] = pd.to_numeric(events["duration"], errors="coerce")
    events = events.dropna(subset=["onset", "duration"]).reset_index(drop=True)

    selector = condition_event_selector(task, token)
    chosen = events[events["trial_type"].map(selector)]

    mask = np.zeros(n_scans, dtype=bool)
    for _, row in chosen.iterrows():
        start = int(np.floor(float(row["onset"]) / tr))
        end = int(np.ceil((float(row["onset"]) + float(row["duration"])) / tr))
        start = max(0, start)
        end = min(n_scans, end)
        if end > start:
            mask[start:end] = True
    return mask


def compute_condition_coupling(
    run_path: str,
    events_path: str,
    task: str,
    token: str,
    seed_masks: dict[str, np.ndarray],
    target_masks: dict[str, np.ndarray],
) -> dict[str, float] | None:
    img = nib.load(run_path)
    data = img.get_fdata().astype(np.float64)
    if data.ndim != 2:
        return None

    n_scans, n_vert = data.shape
    tr = get_tr_seconds(img)

    # All masks must align with dtseries vertex axis.
    for m in list(seed_masks.values()) + list(target_masks.values()):
        if m.shape[0] != n_vert:
            return None

    tmask = event_mask_from_tsv(events_path, task, token, n_scans, tr)
    if int(tmask.sum()) < 4:
        return None

    seed_ts = {
        name: zscore_1d(data[:, mask].mean(axis=1))
        for name, mask in seed_masks.items()
    }
    target_ts = {
        name: zscore_1d(data[:, mask].mean(axis=1))
        for name, mask in target_masks.items()
    }

    out: dict[str, float] = {}
    for s_name, s in seed_ts.items():
        for t_name, t in target_ts.items():
            out[f"coupling_{s_name}_to_{t_name}"] = safe_corr(s[tmask], t[tmask])
    out["n_condition_trs"] = float(int(tmask.sum()))
    return out


def write_csv(path: str, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def summarize_group(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    if not rows:
        return []
    numeric_cols: set[str] = set()
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float, np.floating)) and np.isfinite(float(v)):
                numeric_cols.add(k)

    grouped: dict[tuple[str, str], list[dict[str, float | str]]] = {}
    for r in rows:
        key = (str(r.get("level", "")), str(r.get("condition", "")))
        grouped.setdefault(key, []).append(r)

    out = []
    for (level, condition), group in grouped.items():
        item: dict[str, float | str] = {
            "level": level,
            "condition": condition,
            "n_subjects": len(set(str(g.get("subject", "")) for g in group)),
            "n_rows": len(group),
        }
        for c in sorted(numeric_cols):
            vals = [
                float(g[c])
                for g in group
                if c in g and isinstance(g[c], (int, float, np.floating)) and np.isfinite(float(g[c]))
            ]
            if vals:
                item[f"mean_{c}"] = float(np.mean(vals))
                item[f"std_{c}"] = float(np.std(vals, ddof=0))
        out.append(item)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Easy/Hard FPN activation and DMN/DAN coupling for MD tasks.")
    p.add_argument("--subjects", nargs="+", help="Subject IDs (e.g. 04 06 07)")
    p.add_argument("--all-subjects", action="store_true", help="Use all detectable subjects")
    p.add_argument("--contrast-base", default=DEFAULT_CONTRAST_BASE)
    p.add_argument("--dtseries-base", default=DEFAULT_DTSERIES_BASE)
    p.add_argument("--events-base", default=DEFAULT_EVENTS_BASE)
    p.add_argument("--subnetwork-dir", default=DEFAULT_SUBNETWORK_DIR)
    p.add_argument("--network-label-base", default=DEFAULT_NETWORK_LABEL_BASE)
    p.add_argument("--all-contrasts-tsv", default=DEFAULT_ALL_CONTRASTS)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--k-index", type=int, default=0, help="Row index in relabeled FPN file")
    p.add_argument("--strict", action="store_true", help="Fail on missing task-condition data instead of skipping")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    task_map = build_task_difficulty_map(args.all_contrasts_tsv)
    subjects = detect_subjects(args.contrast_base, args.subnetwork_dir, args.subjects)
    if not args.all_subjects and args.subjects is None:
        args.all_subjects = True

    if not subjects:
        raise SystemExit("No subjects found.")

    print(f"Subjects: {' '.join(subjects)}")
    print(f"Output: {args.output}")

    task_rows: list[dict[str, float | str]] = []
    subject_summary_rows: list[dict[str, float | str]] = []

    for subject in subjects:
        print(f"\n=== sub-{subject} ===")
        try:
            fpn_mask, fpna_mask, fpnb_mask = load_fpn_masks(args.subnetwork_dir, subject, args.k_index)
            dmn_mask, dan_mask = load_dmn_dan_masks(args.network_label_base, subject)
        except Exception as e:
            msg = f"Skipping sub-{subject}: {e}"
            if args.strict:
                raise RuntimeError(msg) from e
            print(msg)
            continue

        seed_masks = {
            "fpn": fpn_mask,
            "fpna": fpna_mask,
            "fpnb": fpnb_mask,
        }
        target_masks = {
            "dmn": dmn_mask,
            "dan": dan_mask,
        }

        subject_task_rows: list[dict[str, float | str]] = []

        for task, td in task_map.items():
            z_dir = os.path.join(
                args.contrast_base,
                f"sub-{subject}",
                f"res_task-{task}_space-fsLR_dir-ffx",
                "z_score_maps",
            )
            hard_files = find_condition_maps(z_dir, td.hard_token)
            easy_files = find_condition_maps(z_dir, td.easy_token)

            hard_map = aggregate_condition_map(hard_files)
            easy_map = aggregate_condition_map(easy_files)

            if hard_map is None or easy_map is None:
                msg = (
                    f"Skipping sub-{subject} task-{task}: missing hard/easy maps "
                    f"(hard_files={len(hard_files)}, easy_files={len(easy_files)})"
                )
                if args.strict:
                    raise RuntimeError(msg)
                print(msg)
                continue

            run_files = find_run_files(args.dtseries_base, subject, task)
            hard_couplings: list[dict[str, float]] = []
            easy_couplings: list[dict[str, float]] = []

            for run_path in run_files:
                session, parsed_task, direction, run = parse_run_parts(run_path)
                events_path = find_events_file(args.events_base, subject, session, parsed_task, direction, run)
                if events_path is None:
                    continue

                hc = compute_condition_coupling(
                    run_path,
                    events_path,
                    parsed_task,
                    td.hard_token,
                    seed_masks,
                    target_masks,
                )
                ec = compute_condition_coupling(
                    run_path,
                    events_path,
                    parsed_task,
                    td.easy_token,
                    seed_masks,
                    target_masks,
                )
                if hc is not None:
                    hard_couplings.append(hc)
                if ec is not None:
                    easy_couplings.append(ec)

            def coupling_mean(items: list[dict[str, float]], key: str) -> float:
                vals = [it[key] for it in items if key in it and np.isfinite(it[key])]
                return float(np.mean(vals)) if vals else float("nan")

            hard_row: dict[str, float | str] = {
                "level": "task_condition",
                "subject": subject,
                "task": task,
                "contrast": td.contrast,
                "condition": "hard",
                "condition_token": td.hard_token,
                "n_condition_maps": len(hard_files),
                "n_runs": len(run_files),
                "n_runs_with_coupling": len(hard_couplings),
                "mean_fpn_activation": safe_mean(hard_map[fpn_mask]),
                "mean_fpna_activation": safe_mean(hard_map[fpna_mask]),
                "mean_fpnb_activation": safe_mean(hard_map[fpnb_mask]),
                "coupling_fpn_to_dmn": coupling_mean(hard_couplings, "coupling_fpn_to_dmn"),
                "coupling_fpn_to_dan": coupling_mean(hard_couplings, "coupling_fpn_to_dan"),
                "coupling_fpna_to_dmn": coupling_mean(hard_couplings, "coupling_fpna_to_dmn"),
                "coupling_fpna_to_dan": coupling_mean(hard_couplings, "coupling_fpna_to_dan"),
                "coupling_fpnb_to_dmn": coupling_mean(hard_couplings, "coupling_fpnb_to_dmn"),
                "coupling_fpnb_to_dan": coupling_mean(hard_couplings, "coupling_fpnb_to_dan"),
                "mean_condition_trs": coupling_mean(hard_couplings, "n_condition_trs"),
            }

            easy_row: dict[str, float | str] = {
                "level": "task_condition",
                "subject": subject,
                "task": task,
                "contrast": td.contrast,
                "condition": "easy",
                "condition_token": td.easy_token,
                "n_condition_maps": len(easy_files),
                "n_runs": len(run_files),
                "n_runs_with_coupling": len(easy_couplings),
                "mean_fpn_activation": safe_mean(easy_map[fpn_mask]),
                "mean_fpna_activation": safe_mean(easy_map[fpna_mask]),
                "mean_fpnb_activation": safe_mean(easy_map[fpnb_mask]),
                "coupling_fpn_to_dmn": coupling_mean(easy_couplings, "coupling_fpn_to_dmn"),
                "coupling_fpn_to_dan": coupling_mean(easy_couplings, "coupling_fpn_to_dan"),
                "coupling_fpna_to_dmn": coupling_mean(easy_couplings, "coupling_fpna_to_dmn"),
                "coupling_fpna_to_dan": coupling_mean(easy_couplings, "coupling_fpna_to_dan"),
                "coupling_fpnb_to_dmn": coupling_mean(easy_couplings, "coupling_fpnb_to_dmn"),
                "coupling_fpnb_to_dan": coupling_mean(easy_couplings, "coupling_fpnb_to_dan"),
                "mean_condition_trs": coupling_mean(easy_couplings, "n_condition_trs"),
            }

            diff_row: dict[str, float | str] = {
                "level": "task_condition",
                "subject": subject,
                "task": task,
                "contrast": td.contrast,
                "condition": "hard_minus_easy",
                "condition_token": f"{td.hard_token}_minus_{td.easy_token}",
                "n_condition_maps": min(len(hard_files), len(easy_files)),
                "n_runs": len(run_files),
                "n_runs_with_coupling": min(len(hard_couplings), len(easy_couplings)),
            }
            for key in [
                "mean_fpn_activation",
                "mean_fpna_activation",
                "mean_fpnb_activation",
                "coupling_fpn_to_dmn",
                "coupling_fpn_to_dan",
                "coupling_fpna_to_dmn",
                "coupling_fpna_to_dan",
                "coupling_fpnb_to_dmn",
                "coupling_fpnb_to_dan",
                "mean_condition_trs",
            ]:
                hv = hard_row.get(key)
                ev = easy_row.get(key)
                if isinstance(hv, (int, float, np.floating)) and isinstance(ev, (int, float, np.floating)):
                    if np.isfinite(float(hv)) and np.isfinite(float(ev)):
                        diff_row[key] = float(hv) - float(ev)
                    else:
                        diff_row[key] = float("nan")

            subject_task_rows.extend([hard_row, easy_row, diff_row])
            print(
                f"sub-{subject} {task}: "
                f"A(h-e)={diff_row.get('mean_fpna_activation', float('nan')):.4f}/"
                f"{diff_row.get('mean_fpnb_activation', float('nan')):.4f} "
                f"runs={len(run_files)} coupling_runs={min(len(hard_couplings), len(easy_couplings))}"
            )

        if not subject_task_rows:
            continue

        task_rows.extend(subject_task_rows)

        # Subject summaries: average across tasks for each condition.
        for condition in ["hard", "easy", "hard_minus_easy"]:
            rows_cond = [r for r in subject_task_rows if r.get("condition") == condition]
            if not rows_cond:
                continue
            summary: dict[str, float | str] = {
                "level": "subject_summary",
                "subject": subject,
                "task": "ALL_MD_TASKS",
                "contrast": "MULTI",
                "condition": condition,
                "condition_token": condition,
                "n_tasks": len(rows_cond),
            }
            metric_keys = [
                "mean_fpn_activation",
                "mean_fpna_activation",
                "mean_fpnb_activation",
                "coupling_fpn_to_dmn",
                "coupling_fpn_to_dan",
                "coupling_fpna_to_dmn",
                "coupling_fpna_to_dan",
                "coupling_fpnb_to_dmn",
                "coupling_fpnb_to_dan",
                "mean_condition_trs",
            ]
            for k in metric_keys:
                vals = [
                    float(r[k])
                    for r in rows_cond
                    if k in r and isinstance(r[k], (int, float, np.floating)) and np.isfinite(float(r[k]))
                ]
                summary[k] = float(np.mean(vals)) if vals else float("nan")
            subject_summary_rows.append(summary)

    if not task_rows and not subject_summary_rows:
        raise SystemExit("No results were computed. Check inputs and subject availability.")

    per_subject_csv = os.path.join(args.output, "per_subject_results.csv")
    write_csv(per_subject_csv, task_rows + subject_summary_rows)

    group_rows = summarize_group(subject_summary_rows)
    group_csv = os.path.join(args.output, "group_summary.csv")
    write_csv(group_csv, group_rows)

    print("\nSaved:")
    print(f"  {per_subject_csv}")
    print(f"  {group_csv}")


if __name__ == "__main__":
    main()
