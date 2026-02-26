"""
Step 2: Build PPI contrast-level betas from condition-level PPI betas.

Input:
  - per-subject condition-level PPI results:
      /ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/ppi_results_dmn_dan/sub-*/ppi_dmn_dan_results.csv
      (output of ppi_analysis_DMN_DAN_filtered.py)

Output:
  - per-subject contrast-level PPI results:
      /ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/ppi_results_dmn_dan/sub-*/ppi_dmn_dan_contrasts.csv
"""
# filepath: /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/ppi_build_contrasts_from_conditions.py

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add project root to path: /home/hmueller2/ibc_code/ibc_latent
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT_DIR)

from public_analysis.ibc_public.utils_contrasts import make_contrasts

PPI_BASE = "/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/ppi_results_dmn_dan"
# NEW: where all_contrast.tsv actually lives (same folder as this script)
DATA_INFO_DIR = os.path.join(os.path.dirname(__file__), "Data Info")

# Additional fallback locations (including your /ptmp/Downloads copy)
ALL_CONTRAST_CANDIDATES = [
    os.path.join(DATA_INFO_DIR, "all_contrast.tsv"),
    "/ptmp/hmueller2/2025_ibc_latent/misc/all_contrasts.tsv",  # note the "s" in the filename you mentioned
]


def _rename_seed(seed: str) -> str:
    """Map internal FPN1/FPN2 labels to FPNA/FPNB."""
    return seed.replace("FPN1", "FPNA").replace("FPN2", "FPNB")


def build_contrast_level_ppi_for_subject(subject):
    sub_id = f"sub-{subject}"
    print(f"\n=== Building PPI contrasts for {sub_id} ===")

    subj_dir = os.path.join(PPI_BASE, sub_id)
    cond_file = os.path.join(subj_dir, "ppi_dmn_dan_results.csv")

    if not os.path.exists(cond_file):
        print(f"  Skipping {sub_id}: {cond_file} not found")
        return None

    cond_df = pd.read_csv(cond_file)
    if cond_df.empty:
        print(f"  Skipping {sub_id}: empty condition-level file")
        return None

    # Drop rows with missing condition values
    cond_df = cond_df.dropna(subset=["condition"]).reset_index(drop=True)
    if cond_df.empty:
        print(f"  Skipping {sub_id}: no valid conditions after dropping NaN")
        return None

    # Sanity: required columns
    required_cols = {"task", "condition", "seed", "target", "run_id", "beta"}
    missing = required_cols - set(cond_df.columns)
    if missing:
        print(f"  Skipping {sub_id}: missing required columns {missing}")
        return None

    contrast_rows = []


    # Loop over task × seed × target × run
    for (task, seed, target, run_id), group in cond_df.groupby(["task", "seed", "target", "run_id"]):
        # Conditions present for this combination
        cond_names = sorted(group["condition"].unique())

        # Map condition name → beta
        beta_by_cond = group.set_index("condition")["beta"].to_dict()

        try:
            contrasts_dict = make_contrasts(task, cond_names)
        except Exception as e:
            print(f"  {sub_id} {run_id} {seed}->{target}: make_contrasts failed for task {task}: {e}")
            continue

        if not contrasts_dict:
            print(f"  {sub_id} {run_id} {seed}->{target}: no contrasts returned for task {task}")
            continue

        # Build vector of betas aligned with cond_names
        beta_vec = np.array([beta_by_cond.get(c, np.nan) for c in cond_names], dtype=float)

        # If all NaN or no finite values, skip
        if not np.isfinite(beta_vec).any():
            print(f"  {sub_id} {run_id} {seed}->{target}: all NaN betas, skipping")
            continue

        for contrast_name, weights in contrasts_dict.items():
            w = np.asarray(weights, dtype=float).ravel()

            if w.shape[0] != len(cond_names):
                # Contrast not defined on this exact set of conditions
                continue

            # If some betas missing, ignore those positions in both beta_vec and w
            valid = np.isfinite(beta_vec) & np.isfinite(w)
            if not valid.any():
                continue

            beta_c = float(np.dot(w[valid], beta_vec[valid]))

            contrast_rows.append(
                {
                    "subject": subject,
                    "task": task,
                    "contrast": contrast_name,
                    # rename seeds for output
                    "seed": _rename_seed(seed),
                    "target": target,
                    "run_id": run_id,
                    "beta_contrast": beta_c,
                }
            )

    if not contrast_rows:
        print(f"  No contrast-level rows created for {sub_id}")
        return None

    contrast_df = pd.DataFrame(contrast_rows)
    out_file = os.path.join(subj_dir, "ppi_dmn_dan_contrasts.csv")
    contrast_df.to_csv(out_file, index=False)
    print(f"  ✓ Saved contrast-level PPI for {sub_id} to {out_file}")

    return contrast_df


def _calculate_cohens_d(values: np.ndarray) -> float:
    """Cohen's d relative to zero."""
    if len(values) < 2:
        return np.nan
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    return mean / std if std > 0 else 0.0


def _build_group_contrast_stats(group_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build group-level summary with:
    task, condition, task_condition, seed, target, seed_target,
    mean_effect, median_effect, std_effect, t_statistic, p_value,
    cohens_d, n_subjects, consistency
    """
    # Expect columns: subject, task, contrast, seed, target, run_id, beta_contrast
    # We interpret "contrast" as "condition" here to match requested naming.
    df = group_df.copy()
    df = df.rename(columns={"contrast": "condition", "beta_contrast": "beta"})
    # aggregate to subject-level within task×condition×seed×target
    subj_level = (
        df.groupby(["subject", "task", "condition", "seed", "target"])["beta"]
        .mean()
        .reset_index()
    )

    stats_rows = []
    for (task, condition, seed, target), g in subj_level.groupby(
        ["task", "condition", "seed", "target"]
    ):
        effects = g["beta"].dropna().values
        if len(effects) == 0:
            continue

        n_subjects = len(effects)
        mean_effect = float(np.mean(effects))
        median_effect = float(np.median(effects))
        std_effect = float(np.std(effects, ddof=1)) if n_subjects > 1 else np.nan

        if n_subjects > 1:
            t_stat, p_val = stats.ttest_1samp(effects, 0.0)
            cohens_d = _calculate_cohens_d(effects)
        else:
            t_stat, p_val, cohens_d = np.nan, np.nan, np.nan

        pos_count = int(np.sum(effects > 0))
        consistency = max(pos_count, n_subjects - pos_count) / n_subjects

        seed_target = f"{seed}→{target}"
        task_condition = f"{task}_{condition}"

        stats_rows.append(
            {
                "task": task,
                "condition": condition,
                "task_condition": task_condition,
                "seed": seed,
                "target": target,
                "seed_target": seed_target,
                "mean_effect": mean_effect,
                "median_effect": median_effect,
                "std_effect": std_effect,
                "t_statistic": t_stat,
                "p_value": p_val,
                "cohens_d": cohens_d,
                "n_subjects": n_subjects,
                "consistency": consistency,
            }
        )

    return pd.DataFrame(stats_rows)


def _attach_contrast_metadata(
    combo_stats: pd.DataFrame, root_dir: str
) -> pd.DataFrame:
    """
    Attach 'pretty name' and 'tags' from /Data Info/all_contrast.tsv
    to combo_stats based on task_condition.
    """
    # Try multiple possible locations
    tsv_path = None
    for candidate in ALL_CONTRAST_CANDIDATES:
        if os.path.exists(candidate):
            tsv_path = candidate
            break

    if tsv_path is None:
        print(
            "  ⚠ all_contrast.tsv not found at any of: "
            + ", ".join(ALL_CONTRAST_CANDIDATES)
            + " – skipping metadata merge"
        )
        combo_stats["pretty_name"] = np.nan
        combo_stats["tags"] = np.nan
        return combo_stats

    print(f"  Using all_contrast.tsv from: {tsv_path}")

    try:
        contrast_meta = pd.read_csv(tsv_path, sep="\t")
    except Exception as e:
        print(f"  ⚠ Failed to read {tsv_path}: {e} – skipping metadata merge")
        combo_stats["pretty_name"] = np.nan
        combo_stats["tags"] = np.nan
        return combo_stats

    # Expect columns: "task", "contrast", "pretty name", "tags"
    required_cols = {"task", "contrast"}
    missing = required_cols - set(contrast_meta.columns)
    if missing:
        print(
            f"  ⚠ all_contrast.tsv missing required columns {missing} – skipping metadata merge"
        )
        combo_stats["pretty_name"] = np.nan
        combo_stats["tags"] = np.nan
        return combo_stats

    # Rename 'contrast' -> 'condition' so we can build the same task_condition key
    contrast_meta = contrast_meta.rename(columns={"contrast": "condition"})

    # Build task_condition = f"{task}_{condition}" to match combo_stats
    contrast_meta["task_condition"] = (
        contrast_meta["task"].astype(str) + "_" + contrast_meta["condition"].astype(str)
    )

    # Normalize column names for merge
    # 'pretty name' -> 'pretty_name'
    if "pretty name" in contrast_meta.columns:
        contrast_meta = contrast_meta.rename(columns={"pretty name": "pretty_name"})
    elif "pretty_name" not in contrast_meta.columns:
        contrast_meta["pretty_name"] = np.nan

    if "tags" not in contrast_meta.columns:
        contrast_meta["tags"] = np.nan

    meta = contrast_meta[["task_condition", "pretty_name", "tags"]].drop_duplicates()

    # Debug: overlap
    n_overlap = len(
        set(combo_stats["task_condition"]).intersection(set(meta["task_condition"]))
    )
    print(
        f"  Metadata merge: {n_overlap} task_condition values matched between summary and all_contrast.tsv"
    )

    # Left-merge on task_condition
    combo_stats = combo_stats.merge(meta, on="task_condition", how="left")

    # Ensure columns exist
    if "pretty_name" not in combo_stats.columns:
        combo_stats["pretty_name"] = np.nan
    if "tags" not in combo_stats.columns:
        combo_stats["tags"] = np.nan

    return combo_stats


def _create_contrast_boxplots_fpna_fpnb_2x2(
    combo_stats: pd.DataFrame,
    group_df: pd.DataFrame,
    output_dir: str,
    top_n: int = 15,
) -> None:
    """
    2×2 figure with horizontal boxplots + jittered dots for *contrasts*.

    Layout (rows = seeds, cols = targets):
      Row 0: FPNA→DMN (left), FPNA→DAN (right)
      Row 1: FPNB→DMN (left), FPNB→DAN (right)

    For each panel:
      - Use combo_stats to select top-N significant contrasts (p<0.05) by |cohens_d|
        within seed_family×target (seed_family derived from seed: FPNA/FPNB).
      - From group_df, aggregate to subject-level and compute subject-level
        Cohen's d (beta distribution across runs per subject), and plot those.
    """
    print("\n[ALT] Creating 2×2 contrast boxplot figure (FPNA/FPNB × DMN/DAN)...")

    if combo_stats.empty or group_df.empty:
        print("  ✗ combo_stats or group_df is empty; skipping contrast boxplot 2×2.")
        return

    required_cs = {"seed", "target", "task", "condition", "task_condition", "cohens_d", "p_value"}
    required_g = {"subject", "task", "contrast", "seed", "target", "beta_contrast"}
    if not required_cs.issubset(combo_stats.columns):
        print("  ✗ Missing required columns in combo_stats; skipping contrast boxplot 2×2.")
        return
    if not required_g.issubset(group_df.columns):
        print("  ✗ Missing required columns in group_df; skipping contrast boxplot 2×2.")
        return

    cs = combo_stats.copy()
    gd = group_df.copy()

    # derive seed_family from seed (FPNA/FPNB)
    cs["seed_family"] = cs["seed"].str.extract(r"(FPN[AB])", expand=False)
    cs = cs[cs["seed_family"].isin(["FPNA", "FPNB"])].copy()
    if cs.empty:
        print("  ✗ No FPNA/FPNB rows in combo_stats; skipping contrast boxplot 2×2.")
        return

    # build task_condition on group_df to match combo_stats
    gd = gd.rename(columns={"contrast": "condition"})
    gd["task_condition"] = gd["task"].astype(str) + "_" + gd["condition"].astype(str)

    # --- NEW: compute subject-level Cohen's d per task_condition×seed×target ---
    def _cohens_d_1sample(vals: np.ndarray) -> float:
        vals = np.asarray(vals, float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            return np.nan
        m = float(vals.mean())
        s = float(vals.std(ddof=1))
        return m / s if s > 0 else 0.0

    subj_level = (
        gd.groupby(["subject", "task_condition", "seed", "target"])["beta_contrast"]
        .apply(lambda v: _cohens_d_1sample(v))
        .reset_index(name="cohens_d_subj")
    )

    # helper: select top-N contrasts for a seed_family×target from combo_stats
    def _select_top_tc(seed_family: str, target: str) -> list[str]:
        sub = cs[
            (cs["seed_family"] == seed_family)
            & (cs["target"] == target)
            & (cs["p_value"] < 0.05)
        ].copy()
        if sub.empty:
            return []
        sub["abs_d"] = sub["cohens_d"].abs()
        sub = sub.sort_values("abs_d", ascending=False).head(top_n)
        return sub["task_condition"].tolist()

    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), squeeze=False)
    fig.suptitle(
        "PPI Contrasts – Subject-wise Cohen's d (FPNA/FPNB × DMN/DAN)\n"
        "(selection: top |d| contrasts from combo_stats, p<0.05)",
        fontsize=15,
        fontweight="bold",
    )

    combos = [
        ("FPNA", "DMN", 0, 0),  # top-left
        ("FPNB", "DMN", 0, 1),  # top-right
        ("FPNA", "DAN", 1, 0),  # bottom-left
        ("FPNB", "DAN", 1, 1),  # bottom-right
    ]

    def _plot_panel(ax, seed_family: str, target: str):
        top_tc = _select_top_tc(seed_family, target)
        if not top_tc:
            ax.text(
                0.5, 0.5,
                f"No significant contrasts for {seed_family}→{target}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
            ax.grid(axis="x", alpha=0.3)
            return

        sub = subj_level[
            (subj_level["seed"].str.contains(seed_family, na=False))
            & (subj_level["target"] == target)
            & (subj_level["task_condition"].isin(top_tc))
        ].copy()

        sub = sub[np.isfinite(sub["cohens_d_subj"])]
        if sub.empty:
            ax.text(
                0.5, 0.5,
                f"No subject-level d for {seed_family}→{target}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
            ax.grid(axis="x", alpha=0.3)
            return

        # order contrasts by mean subject-level d
        order = (
            sub.groupby("task_condition")["cohens_d_subj"]
            .mean()
            .sort_values(ascending=True)
            .index.tolist()
        )
        positions = np.arange(len(order))

        plot_data = [
            sub[sub["task_condition"] == tc]["cohens_d_subj"].values for tc in order
        ]

        bp = ax.boxplot(
            plot_data,
            vert=False,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
            boxprops=dict(linewidth=1.2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#dddddd")
            patch.set_alpha(0.7)

        color = "#f94144" if target == "DMN" else "#90be6d"

        for i, tc in enumerate(order):
            vals = sub[sub["task_condition"] == tc]["cohens_d_subj"].values
            y = np.random.normal(loc=positions[i], scale=0.08, size=len(vals))
            ax.scatter(
                vals,
                y,
                s=25,
                color=color,
                edgecolors="black",
                linewidth=0.4,
                alpha=0.7,
                zorder=3,
            )

        ax.set_yticks(positions)
        ax.set_yticklabels(order, fontsize=9)
        ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
        ax.set_xlabel("Cohen's d (subject-level PPI effect)", fontsize=11)
        # NEW: constrain x-axis
        ax.set_xlim(-8.0, 6.0)
        ax.set_title(f"{seed_family}→{target}", fontsize=13, fontweight="bold")

    for seed_family, target, r, c in combos:
        _plot_panel(axes[r, c], seed_family, target)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(output_dir, "ppi_contrast_boxplots_FPNA_FPNB_DMN_DAN.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved 2×2 contrast boxplot figure to {out_path}")

def _export_top_contrast_tables_fpna_fpnb_2x2(
    combo_stats: pd.DataFrame,
    output_dir: str,
    top_n: int = 20,
) -> None:
    """
    Export tables of contrasts shown in the FPNA/FPNB × DMN/DAN 2×2 boxplot.

    Selection matches _create_contrast_boxplots_fpna_fpnb_2x2:
      - For each (seed_family, target), filter to p<0.05,
      - select top-N by |cohens_d| (group-level),
      - then label each as positive/negative by sign of cohens_d.

    The exported CSV therefore contains exactly the contrasts that appear
    in the 2×2 contrast boxplot (if top_n matches), one row per contrast.
    It also includes per-panel positive/negative ranks so you can
    report e.g. top 10 positive / top 10 negative per panel.
    """
    if combo_stats.empty:
        print("  ✗ combo_stats is empty; skipping export of top contrast tables.")
        return

    required_cs = {
        "seed", "target", "task", "condition", "task_condition",
        "mean_effect", "t_statistic", "cohens_d", "p_value",
    }
    if not required_cs.issubset(combo_stats.columns):
        print("  ✗ Missing required columns in combo_stats; skipping export of top contrast tables.")
        return

    cs = combo_stats.copy()
    cs["seed_family"] = cs["seed"].str.extract(r"(FPN[AB])", expand=False)
    cs = cs[cs["seed_family"].isin(["FPNA", "FPNB"])].copy()
    cs = cs[cs["target"].isin(["DMN", "DAN"])].copy()
    cs = cs[cs["p_value"] < 0.05].copy()
    if cs.empty:
        print("  ✗ No significant FPNA/FPNB contrasts; skipping export of top contrast tables.")
        return

    rows = []
    for seed_family in ["FPNA", "FPNB"]:
        for target in ["DMN", "DAN"]:
            sub = cs[
                (cs["seed_family"] == seed_family) &
                (cs["target"] == target)
            ].copy()
            if sub.empty:
                continue

            # match plot selection: top-N by |cohens_d|
            sub["abs_d"] = sub["cohens_d"].abs()
            sub = sub.sort_values("abs_d", ascending=False).head(top_n)
            if sub.empty:
                continue

            # within this panel subset, compute ranks separately for pos/neg
            pos = sub[sub["cohens_d"] > 0].copy()
            neg = sub[sub["cohens_d"] < 0].copy()

            pos = pos.sort_values("cohens_d", ascending=False)
            pos["pos_rank"] = np.arange(1, len(pos) + 1, dtype=float)
            neg = neg.sort_values("cohens_d", ascending=True)
            neg["neg_rank"] = np.arange(1, len(neg) + 1, dtype=float)

            merged = pd.concat([pos, neg], axis=0)

            for _, r in merged.iterrows():
                sign_label = (
                    "positive" if r["cohens_d"] > 0
                    else "negative" if r["cohens_d"] < 0
                    else "zero"
                )
                rows.append(
                    {
                        "seed_family": seed_family,
                        "target": target,
                        "panel": f"{seed_family}→{target}",
                        "sign": sign_label,
                        # keep NaNs as floats; don't cast to int
                        "pos_rank": r.get("pos_rank", np.nan),
                        "neg_rank": r.get("neg_rank", np.nan),
                        "task": r.get("task"),
                        "condition": r.get("condition"),
                        "task_condition": r["task_condition"],
                        "mean_effect": r["mean_effect"],
                        "t_statistic": r["t_statistic"],
                        "cohens_d": r["cohens_d"],
                        "p_value": r["p_value"],
                        "n_subjects": r.get("n_subjects"),
                        "pretty_name": r.get("pretty_name"),
                        "tags": r.get("tags"),
                    }
                )

    if not rows:
        print("  ✗ No rows for top contrast tables.")
        return

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(
        output_dir,
        "ppi_contrast_boxplots_FPNA_FPNB_DMN_DAN_top_contrasts.csv",
    )
    out_df.to_csv(out_path, index=False)
    print(f"  ✓ Saved top-{top_n} by |d| contrast table (matching 2×2 boxplot panels) to {out_path}")

def _create_contrast_boxplots_fpn_1x2(
    combo_stats: pd.DataFrame,
    group_df: pd.DataFrame,
    output_dir: str,
    top_n: int = 15,
) -> None:
    """
    1×2 figure after collapsing FPNA & FPNB into a single 'FPN' family.

    Panels:
      - Top:    FPN→DMN  (colored #f94144)
      - Bottom: FPN→DAN  (colored #90be6d)

    Uses subject-level Cohen's d on the x-axis.
    """
    print("\n[ALT] Creating 1×2 FPN (FPNA+FPNB) contrast boxplot figure (DMN/DAN)...")

    if combo_stats.empty or group_df.empty:
        print("  ✗ combo_stats or group_df is empty; skipping FPN contrast boxplot 1×2.")
        return

    required_cs = {"seed", "target", "task", "condition", "task_condition", "cohens_d", "p_value"}
    required_g = {"subject", "task", "contrast", "seed", "target", "beta_contrast"}
    if not required_cs.issubset(combo_stats.columns):
        print("  ✗ Missing required columns in combo_stats; skipping FPN contrast boxplot 1×2.")
        return
    if not required_g.issubset(group_df.columns):
        print("  ✗ Missing required columns in group_df; skipping FPN contrast boxplot 1×2.")
        return

    cs = combo_stats.copy()
    gd = group_df.copy()

    cs["seed_family"] = cs["seed"].str.extract(r"(FPN[AB])", expand=False)
    cs = cs[cs["seed_family"].isin(["FPNA", "FPNB"])].copy()
    if cs.empty:
        print("  ✗ No FPNA/FPNB rows in combo_stats; skipping FPN contrast boxplot 1×2.")
        return

    cs["seed_family"] = "FPN"

    gd = gd.rename(columns={"contrast": "condition"})
    gd["task_condition"] = gd["task"].astype(str) + "_" + gd["condition"].astype(str)
    gd["seed_family"] = gd["seed"].str.extract(r"(FPN[AB])", expand=False)
    gd = gd[gd["seed_family"].isin(["FPNA", "FPNB"])].copy()
    if gd.empty:
        print("  ✗ No FPNA/FPNB rows in group_df; skipping FPN contrast boxplot 1×2.")
        return

    gd["seed_family"] = "FPN"

    # --- NEW: compute subject-level Cohen's d per task_condition×seed_family×target ---
    def _cohens_d_1sample(vals: np.ndarray) -> float:
        vals = np.asarray(vals, float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            return np.nan
        m = float(vals.mean())
        s = float(vals.std(ddof=1))
        return m / s if s > 0 else 0.0

    subj_level = (
        gd.groupby(["subject", "task_condition", "seed_family", "target"])["beta_contrast"]
        .apply(lambda v: _cohens_d_1sample(v))
        .reset_index(name="cohens_d_subj")
    )

    def _select_top_tc_fpn(target: str) -> list[str]:
        sub = cs[
            (cs["seed_family"] == "FPN")
            & (cs["target"] == target)
            & (cs["p_value"] < 0.05)
        ].copy()
        if sub.empty:
            return []
        sub["abs_d"] = sub["cohens_d"].abs()
        sub = sub.sort_values("abs_d", ascending=False).head(top_n)
        return sub["task_condition"].tolist()

    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 12,
    })

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), squeeze=False)
    fig.suptitle(
        "PPI Contrasts – Subject-wise Cohen's d (FPN = FPNA+FPNB)\n"
        "(selection: top |d| contrasts from collapsed FPN combo_stats, p<0.05)",
        fontsize=15,
        fontweight="bold",
    )

    targets = ["DMN", "DAN"]

    def _plot_panel(ax, target: str):
        top_tc = _select_top_tc_fpn(target)
        if not top_tc:
            ax.text(
                0.5, 0.5,
                f"No significant contrasts for FPN→{target}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
            ax.grid(axis="x", alpha=0.3)
            return

        sub = subj_level[
            (subj_level["seed_family"] == "FPN")
            & (subj_level["target"] == target)
            & (subj_level["task_condition"].isin(top_tc))
        ].copy()

        sub = sub[np.isfinite(sub["cohens_d_subj"])]
        if sub.empty:
            ax.text(
                0.5, 0.5,
                f"No subject-level d for FPN→{target}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
            ax.grid(axis="x", alpha=0.3)
            return

        order = (
            sub.groupby("task_condition")["cohens_d_subj"]
            .mean()
            .sort_values(ascending=True)
            .index.tolist()
        )
        positions = np.arange(len(order))
        plot_data = [
            sub[sub["task_condition"] == tc]["cohens_d_subj"].values for tc in order
        ]

        bp = ax.boxplot(
            plot_data,
            vert=False,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
            boxprops=dict(linewidth=1.2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#dddddd")
            patch.set_alpha(0.7)

        color = "#f94144" if target == "DMN" else "#90be6d"

        for i, tc in enumerate(order):
            vals = sub[sub["task_condition"] == tc]["cohens_d_subj"].values
            y = np.random.normal(loc=positions[i], scale=0.08, size=len(vals))
            ax.scatter(
                vals,
                y,
                s=25,
                color=color,
                edgecolors="black",
                linewidth=0.4,
                alpha=0.7,
                zorder=3,
            )

        ax.set_yticks(positions)
        ax.set_yticklabels(order, fontsize=9)
        ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
        ax.set_xlabel("Cohen's d (subject-level PPI effect)", fontsize=11)
        # NEW: constrain x-axis to [-4, 4]
        ax.set_xlim(-4.0, 4.0)
        ax.set_title(f"FPN→{target}", fontsize=13, fontweight="bold")

    _plot_panel(axes[0, 0], "DMN")
    _plot_panel(axes[1, 0], "DAN")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(output_dir, "ppi_contrast_boxplots_FPN_DMN_DAN.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved FPN 1×2 contrast boxplot figure to {out_path}")


def _compare_fpna_fpnb_contrasts(combo_stats: pd.DataFrame, output_dir: str) -> None:
    """
    Compare FPNA vs FPNB at the contrast level.
    Aggregates per seed_family (FPNA/FPNB) × target × task_condition and
    produces a table with both sides side-by-side, plus a brief printed summary.
    """
    df = combo_stats.copy()

    # Extract FPNA/FPNB seed_family
    df["seed_family"] = df["seed"].str.extract(r"(FPN[AB])", expand=False)
    df = df[df["seed_family"].isin(["FPNA", "FPNB"])].copy()
    if df.empty:
        print("  No FPNA/FPNB seeds found in combo_stats; skipping FPNA/FPNB contrast comparison.")
        return

    # Aggregate per seed_family × target × task_condition
    grouped = (
        df.groupby(["seed_family", "target", "task_condition"])
        .agg(
            mean_effect=("mean_effect", "mean"),
            cohens_d=("cohens_d", "mean"),
            best_p_value=("p_value", "min"),
            n_contrasts=("task_condition", "size"),
        )
        .reset_index()
    )

    # Split to FPNA, FPNB and merge
    fpna = grouped[grouped["seed_family"] == "FPNA"].rename(
        columns={
            "mean_effect": "mean_effect_FPNA",
            "cohens_d": "cohens_d_FPNA",
            "best_p_value": "best_p_FPNA",
            "n_contrasts": "n_contrasts_FPNA",
        }
    ).drop(columns=["seed_family"])

    fpnb = grouped[grouped["seed_family"] == "FPNB"].rename(
        columns={
            "mean_effect": "mean_effect_FPNB",
            "cohens_d": "cohens_d_FPNB",
            "best_p_value": "best_p_FPNB",
            "n_contrasts": "n_contrasts_FPNB",
        }
    ).drop(columns=["seed_family"])

    merged = fpna.merge(fpnb, on=["target", "task_condition"], how="outer")

    # Significance + direction
    alpha = 0.05
    merged["sig_FPNA"] = merged["best_p_FPNA"] < alpha
    merged["sig_FPNB"] = merged["best_p_FPNB"] < alpha

    merged["direction_FPNA"] = np.sign(merged["cohens_d_FPNA"])
    merged["direction_FPNB"] = np.sign(merged["cohens_d_FPNB"])

    merged["shared_sig_same_dir"] = (
        merged["sig_FPNA"]
        & merged["sig_FPNB"]
        & (merged["direction_FPNA"] == merged["direction_FPNB"])
        & (merged["direction_FPNA"] != 0)
    )

    out_file = os.path.join(output_dir, "ppi_dmn_dan_FPNA_vs_FPNB_contrasts.csv")
    merged.to_csv(out_file, index=False)
    print(f"  ✓ Saved FPNA vs FPNB contrast comparison to {out_file}")

    # Brief summary per target
    print("\n=== FPNA vs FPNB contrast overlap (alpha = 0.05) ===")
    for target in sorted(merged["target"].dropna().unique()):
        sub = merged[merged["target"] == target]
        n_total = len(sub)
        n_sig_fpna = sub["sig_FPNA"].sum()
        n_sig_fpnb = sub["sig_FPNB"].sum()
        n_shared = sub["shared_sig_same_dir"].sum()

        print(f"\nTarget: {target}")
        print(f"  Task-conditions (any side): {n_total}")
        print(f"  Significant FPNA task-conditions: {n_sig_fpna}")
        print(f"  Significant FPNB task-conditions: {n_sig_fpnb}")
        print(f"  Shared significant (same direction): {n_shared}")

        if n_shared > 0:
            top_shared = (
                sub[sub["shared_sig_same_dir"]]
                .copy()
                .assign(
                    abs_d_FPNA=lambda x: x["cohens_d_FPNA"].abs(),
                    abs_d_FPNB=lambda x: x["cohens_d_FPNB"].abs(),
                    max_abs_d=lambda x: x[["abs_d_FPNA", "abs_d_FPNB"]].max(axis=1),
                )
                .sort_values("max_abs_d", ascending=False)
                .head(5)
            )
            print("  Top shared task-conditions by |d|:")
            for _, row in top_shared.iterrows():
                print(
                    f"    - {row['task_condition']}: "
                    f"d_FPNA={row['cohens_d_FPNA']:.3f}, "
                    f"d_FPNB={row['cohens_d_FPNB']:.3f}, "
                    f"p_FPNA={row['best_p_FPNA']:.3g}, "
                    f"p_FPNB={row['best_p_FPNB']:.3g}"
                )


def main():
    if len(sys.argv) > 1:
        subjects = sys.argv[1:]
    else:
        # auto-detect sub-* directories under PPI_BASE
        subjects = sorted(
            d.replace("sub-", "")
            for d in os.listdir(PPI_BASE)
            if d.startswith("sub-") and os.path.isdir(os.path.join(PPI_BASE, d))
        )

    print(f"Subjects to process: {subjects}")

    all_contrasts = []
    for subject in subjects:
        df_sub = build_contrast_level_ppi_for_subject(subject)
        if df_sub is not None:
            all_contrasts.append(df_sub)

    if all_contrasts:
        group_df = pd.concat(all_contrasts, ignore_index=True)
        group_dir = os.path.join(PPI_BASE, "group_analysis")
        os.makedirs(group_dir, exist_ok=True)

        # Save raw per-run contrast table (unchanged behavior)
        group_file = os.path.join(group_dir, "ppi_dmn_dan_contrasts_all_subjects.csv")
        group_df.to_csv(group_file, index=False)
        print(f"\n✓ Saved group-level contrast table to {group_file}")

        # Build and save summary stats with requested columns
        print("  Computing group-level summary statistics...")
        combo_stats = _build_group_contrast_stats(group_df)

        # Attach pretty_name and tags from all_contrast.tsv
        print("  Attaching contrast metadata (pretty_name, tags)...")
        combo_stats = _attach_contrast_metadata(combo_stats, ROOT_DIR)

        # NEW: create filtered table of "top" contrasts (p<0.05),
        # sorted by target, then seed, then cohens_d (descending)
        top_mask = combo_stats["p_value"] < 0.05
        top_contrasts = (
            combo_stats.loc[top_mask]
            .copy()
            .sort_values(
                by=["target", "seed", "cohens_d"],
                ascending=[True, True, False],
            )
        )

        top_file = os.path.join(group_dir, "ppi_dmn_dan_top_contrasts.csv")
        top_contrasts.to_csv(top_file, index=False)
        print(f"  ✓ Saved top-contrast table to {top_file}")

        summary_file = os.path.join(
            group_dir, "ppi_dmn_dan_contrasts_group_summary.csv"
        )
        combo_stats.to_csv(summary_file, index=False)
        print(f"  ✓ Saved group summary table to {summary_file}")

        # NEW: FPNA vs FPNB contrast comparison
        print("  Comparing FPNA vs FPNB at contrast level...")
        _compare_fpna_fpnb_contrasts(combo_stats, group_dir)

        # NEW: contrast-level boxplot figures (subject-level Cohen's d)
        print("  Creating contrast-level boxplot+dots figures...")
        _create_contrast_boxplots_fpna_fpnb_2x2(combo_stats, group_df, group_dir, top_n=15)
        _create_contrast_boxplots_fpn_1x2(combo_stats, group_df, group_dir, top_n=15)

        # NEW: export top contrasts per FPNA/FPNB × DMN/DAN panel (matching boxplot selection)
        print("  Exporting top contrast tables for boxplot panels...")
        _export_top_contrast_tables_fpna_fpnb_2x2(combo_stats, group_dir, top_n=20)
    else:
        print("\nNo contrast-level data generated.")


if __name__ == "__main__":
    main()