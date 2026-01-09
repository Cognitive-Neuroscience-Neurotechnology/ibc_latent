"""
Step 2: Build PPI contrast-level betas from condition-level PPI betas.

Input:
  - per-subject condition-level PPI results:
      /ptmp/hmueller2/Downloads/ppi_results_dmn_dan/sub-*/ppi_dmn_dan_results.csv
      (output of ppi_analysis_DMN_DAN_filtered.py)

Output:
  - per-subject contrast-level PPI results:
      /ptmp/hmueller2/Downloads/ppi_results_dmn_dan/sub-*/ppi_dmn_dan_contrasts.csv
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

PPI_BASE = "/ptmp/hmueller2/Downloads/ppi_results_dmn_dan"
# NEW: where all_contrast.tsv actually lives (same folder as this script)
DATA_INFO_DIR = os.path.join(os.path.dirname(__file__), "Data Info")

# Additional fallback locations (including your /ptmp/Downloads copy)
ALL_CONTRAST_CANDIDATES = [
    os.path.join(DATA_INFO_DIR, "all_contrast.tsv"),
    "/ptmp/hmueller2/Downloads/all_contrasts.tsv",  # note the "s" in the filename you mentioned
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


def _create_polarity_barcharts(combo_stats: pd.DataFrame, output_dir: str) -> None:
    """
    Create per-seed_target bar charts for positive vs negative effects,
    restricted to significant contrasts (p<0.05), sorted by effect size
    (Cohen's d). For each seed_target, show top 10 positive and top 10
    negative contrasts. Bars display Cohen's d, not raw beta.
    """
    # Keep only significant contrasts
    sig = combo_stats[combo_stats["p_value"] < 0.05].copy()
    if sig.empty:
        print("  No significant contrasts (p<0.05); skipping polarity bar-charts.")
        return

    seed_target_pairs = sorted(sig["seed_target"].unique())
    n_pairs = len(seed_target_pairs)
    if n_pairs == 0:
        print("  No seed_target pairs in significant contrasts; skipping plots.")
        return

    fig, axes = plt.subplots(n_pairs, 2, figsize=(16, 4 * n_pairs), squeeze=False)

    for idx, pair in enumerate(seed_target_pairs):
        pair_data = sig[sig["seed_target"] == pair]

        # Ensure we only consider rows with finite Cohen's d
        pair_data = pair_data[np.isfinite(pair_data["cohens_d"])]

        # LEFT: Top 10 positive effects by Cohen's d (largest d first)
        ax_pos = axes[idx, 0]
        pos_effects = (
            pair_data[pair_data["cohens_d"] > 0]
            .sort_values("cohens_d", ascending=False)
            .head(10)
        )

        if len(pos_effects) > 0:
            y_labels = [
                f"{row['task']}_{row['condition']}"
                for _, row in pos_effects.iterrows()
            ]
            colors = ["darkgreen"] * len(pos_effects)

            ax_pos.barh(
                range(len(pos_effects)),
                pos_effects["cohens_d"],
                color=colors,
                alpha=0.7,
                edgecolor="black",
            )
            ax_pos.set_yticks(range(len(pos_effects)))
            ax_pos.set_yticklabels(y_labels, fontsize=8)

            # Add simple significance markers
            for i, (_, row) in enumerate(pos_effects.iterrows()):
                if row["p_value"] < 0.001:
                    mark = "***"
                elif row["p_value"] < 0.01:
                    mark = "**"
                elif row["p_value"] < 0.05:
                    mark = "*"
                else:
                    continue
                ax_pos.text(
                    row["cohens_d"],
                    i,
                    f" {mark}",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )

            ax_pos.axvline(x=0, color="black", linestyle="-", linewidth=1)
            ax_pos.set_xlabel("Effect size (Cohen's d)", fontsize=10)
            ax_pos.set_title(
                f"{pair}: INCREASED Connectivity (Top 10, p<0.05)",
                fontsize=11,
                fontweight="bold",
            )
            ax_pos.grid(axis="x", alpha=0.3)
            ax_pos.set_xlim(left=0)
            ax_pos.invert_yaxis()  # largest effect at top
        else:
            ax_pos.text(
                0.5,
                0.5,
                "No positive significant effects",
                transform=ax_pos.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )
            ax_pos.set_title(
                f"{pair}: INCREASED Connectivity", fontsize=11, fontweight="bold"
            )

        # RIGHT: Top 10 negative effects by Cohen's d (most negative d first)
        ax_neg = axes[idx, 1]
        neg_effects = (
            pair_data[pair_data["cohens_d"] < 0]
            .sort_values("cohens_d", ascending=True)  # more negative d at top
            .head(10)
        )

        if len(neg_effects) > 0:
            y_labels = [
                f"{row['task']}_{row['condition']}"
                for _, row in neg_effects.iterrows()
            ]
            colors = ["darkred"] * len(neg_effects)

            ax_neg.barh(
                range(len(neg_effects)),
                neg_effects["cohens_d"],
                color=colors,
                alpha=0.7,
                edgecolor="black",
            )
            ax_neg.set_yticks(range(len(neg_effects)))
            ax_neg.set_yticklabels(y_labels, fontsize=8)

            for i, (_, row) in enumerate(neg_effects.iterrows()):
                if row["p_value"] < 0.001:
                    mark = "***"
                elif row["p_value"] < 0.01:
                    mark = "**"
                elif row["p_value"] < 0.05:
                    mark = "*"
                else:
                    continue
                ax_neg.text(
                    row["cohens_d"],
                    i,
                    f" {mark}",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )

            ax_neg.axvline(x=0, color="black", linestyle="-", linewidth=1)
            ax_neg.set_xlabel("Effect size (Cohen's d)", fontsize=10)
            ax_neg.set_title(
                f"{pair}: DECREASED Connectivity (Top 10, p<0.05)",
                fontsize=11,
                fontweight="bold",
            )
            ax_neg.grid(axis="x", alpha=0.3)
            ax_neg.set_xlim(right=0)
            ax_neg.invert_yaxis()  # most negative at top
        else:
            ax_neg.text(
                0.5,
                0.5,
                "No negative significant effects",
                transform=ax_neg.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )
            ax_neg.set_title(
                f"{pair}: DECREASED Connectivity", fontsize=11, fontweight="bold"
            )

    plt.tight_layout()
    out_path = os.path.join(output_dir, "contrast_barcharts_polarity.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved polarity bar-charts to {out_path}")


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

        # Attach pretty_name and tags from all_contrast.tsv (for BOTH summary and top table)
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

        # Create polarity bar-chart figure
        print("  Creating polarity bar-chart plots...")
        _create_polarity_barcharts(combo_stats, group_dir)
    else:
        print("\nNo contrast-level data generated.")


if __name__ == "__main__":
    main()