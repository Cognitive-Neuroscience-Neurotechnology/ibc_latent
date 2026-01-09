# filepath: /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/joint_mapping.py
"""
Joint Mapping Analysis

Integrate:
  1) Contrast × subnetwork dominance (FPNA vs FPNB)
  2) PPI task-related coupling (FPN1/FPN2 ↔ DMN/DAN)

to explore the flexible hub hypothesis via:
  A) Grouping contrasts by FPNA vs FPNB dominance and summarising PPI patterns.
  B) Correlating mean_diff_a_minus_b (FPNA–FPNB dominance) with PPI betas for
     FPN1-DMN, FPN1-DAN, FPN2-DMN, FPN2-DAN.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Paths (adjust if you move things)
PPI_BASE = "/ptmp/hmueller2/Downloads/ppi_results_dmn_dan"
CONTRAST_BASE = "/ptmp/hmueller2/Downloads/subnetwork_analysis_results"
OUT_DIR = "/ptmp/hmueller2/Downloads/joint_mapping"

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# 1. LOADERS
# ---------------------------------------------------------------------
def load_ppi_results():
    """
    Load subject-level PPI results.

    Expected per subject:
      {PPI_BASE}/sub-XX/ppi_dmn_dan_contrasts.csv
        columns: subject, task, contrast, seed, target, beta_contrast
    """
    ppi_dfs = []
    for subj_dir in sorted(os.listdir(PPI_BASE)):
        if not subj_dir.startswith("sub-"):
            continue
        sub_dir = subj_dir          # e.g. "sub-04"
        subject = sub_dir           # keep "sub-04" to match contrast files

        ppi_file = os.path.join(PPI_BASE, sub_dir, "ppi_dmn_dan_contrasts.csv")
        if not os.path.exists(ppi_file):
            continue

        df = pd.read_csv(ppi_file)

        # Enforce subject id format to match contrast files
        df["subject"] = subject

        # Build task_contrast to match fpn_subnetwork_contrast_analysis:
        # ArchiEmotional + "_" + expression_control -> ArchiEmotional_expression_control
        df["task_contrast"] = df["task"].astype(str) + "_" + df["contrast"].astype(str)

        # rename beta_contrast -> beta
        df = df.rename(columns={"beta_contrast": "beta"})
        ppi_dfs.append(df)

    if not ppi_dfs:
        raise FileNotFoundError("No PPI contrast results found under PPI_BASE")

    return pd.concat(ppi_dfs, ignore_index=True)


def load_contrast_results():
    """
    Load subject-level contrast × subnetwork dominance results.

    Expected per subject:
      {CONTRAST_BASE}/sub-XX/fpn_subnetwork_contrast_analysis.csv
        columns:
          subject, task, contrast, task_contrast,
          mean_diff_a_minus_b, cohens_d, dominant_network, p_ttest
    """
    contrast_dfs = []
    for subj_dir in sorted(os.listdir(CONTRAST_BASE)):
        if not subj_dir.startswith("sub-"):
            continue
        subject = subj_dir  # should match subject naming in PPI
        contrast_file = os.path.join(CONTRAST_BASE, subj_dir, "fpn_subnetwork_contrast_analysis.csv")
        if not os.path.exists(contrast_file):
            continue

        df = pd.read_csv(contrast_file)
        df["subject"] = subject
        # Ensure task_contrast exists and is string-typed
        df["task_contrast"] = df["task_contrast"].astype(str)
        contrast_dfs.append(df)

    if not contrast_dfs:
        raise FileNotFoundError("No contrast × subnetwork results found under CONTRAST_BASE")

    contrast_all = pd.concat(contrast_dfs, ignore_index=True)
    return contrast_all


# ---------------------------------------------------------------------
# 2. ALIGN CONTRAST DOMINANCE WITH PPI (CORE JOINT TABLE)
# ---------------------------------------------------------------------
def build_joint_table():
    """
    Build a table where each row is:
      subject × task_contrast × seed × target,
    with columns:
      - mean_diff_a_minus_b  (FPNA - FPNB dominance; from contrast_x_subn)
      - cohens_d_diff        (effect size of dominance)
      - dominant_network     ("FPNA", "FPNB", or 'none')
      - p_ttest              (from contrast dominance test)
      - PPI beta             for FPN1/FPN2 ↔ DMN/DAN
    """
    ppi_all = load_ppi_results()
    contrast_all = load_contrast_results()

    # Merge on subject × task_contrast
    # Note: we currently don't distinguish FPNA vs FPNB at this stage;
    # mean_diff_a_minus_b is a single scalar per (subject, task_contrast).
    joint = pd.merge(
        contrast_all,
        ppi_all,
        how="inner",
        on=["subject", "task_contrast", "task"],
        suffixes=("_contrast", "_ppi"),
    )

    # At this point, columns from contrast_all:
    #   subject, task, contrast, task_contrast,
    #   mean_diff_a_minus_b, cohens_d, dominant_network, p_ttest, ...
    # and from ppi_all:
    #   condition, seed, target, beta, variance, tstat, pval, ...
    #
    # We'll keep a clean subset:
    cols_keep = [
        "subject",
        "task",
        "contrast",
        "task_contrast",
        "mean_diff_a_minus_b",
        "cohens_d",           # effect size of mean_diff
        "dominant_network",
        "p_ttest",
        "condition",
        "seed",
        "target",
        "beta",
        "variance",
        "tstat",
        "pval",
    ]
    cols_keep = [c for c in cols_keep if c in joint.columns]
    joint = joint[cols_keep].copy()

    # Map seeds to FPNA/FPNB labels for clarity
    seed_map = {"FPN1": "FPNA", "FPN2": "FPNB"}
    joint["seed_family"] = joint["seed"].map(seed_map).fillna(joint["seed"])

    return joint


# ---------------------------------------------------------------------
# 3. IDEA A: GROUP CONTRASTS BY FPNA / FPNB DOMINANCE → SUMMARISE PPI
# ---------------------------------------------------------------------
def summarize_ppi_by_dominance(joint: pd.DataFrame):
    """
    For each group of contrasts that are FPNA-dominant vs FPNB-dominant,
    summarise PPI beta patterns FPNA/FPNB ↔ DMN/DAN.

    Dominant groups:
      - 'FPNA'  if dominant_network == 'FPNA'
      - 'FPNB'  if dominant_network == 'FPNB'
      - optionally 'none/other' if desired
    """
    # Keep only contrasts with a clear dominant network
    dom = joint[joint["dominant_network"].isin(["FPNA", "FPNB"])].copy()

    summaries = []

    for dom_net in ["FPNA", "FPNB"]:
        sub = dom[dom["dominant_network"] == dom_net]

        # For each PPI link (seed_family × target), summarise beta across rows
        for (seed_family, target), g in sub.groupby(["seed_family", "target"]):
            if seed_family not in ["FPNA", "FPNB"]:
                continue
            mean_beta = g["beta"].mean()
            std_beta = g["beta"].std(ddof=1)
            n = len(g)

            summaries.append(
                {
                    "dominant_network_group": dom_net,
                    "seed_family": seed_family,
                    "target": target,
                    "n_rows": n,
                    "mean_beta": mean_beta,
                    "std_beta": std_beta,
                }
            )

    summary_df = pd.DataFrame(summaries)
    return summary_df


# ---------------------------------------------------------------------
# 4. IDEA B: CORRELATE mean_diff_a_minus_b WITH PPI BETAS
# ---------------------------------------------------------------------
def correlate_mean_diff_with_ppi(joint: pd.DataFrame):
    """
    For each (seed_family, target) pair:
      - correlate mean_diff_a_minus_b with PPI beta across
        subject × task_contrast combinations.

    Returns a dataframe with one row per (seed_family, target).
    """
    results = []
    # Optionally, restrict to DMN/DAN targets only
    joint_sub = joint[joint["target"].isin(["DMN", "DAN"])].copy()

    for (seed_family, target), g in joint_sub.groupby(["seed_family", "target"]):
        if seed_family not in ["FPNA", "FPNB"]:
            continue

        # Drop any missing values
        g = g.dropna(subset=["mean_diff_a_minus_b", "beta"])
        if len(g) < 3:
            continue

        x = g["mean_diff_a_minus_b"].values
        y = g["beta"].values
        r, p = pearsonr(x, y)

        results.append(
            {
                "seed_family": seed_family,  # FPNA/FPNB
                "target": target,            # DMN/DAN
                "n_points": len(g),
                "corr_r": r,
                "p_value": p,
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------
def main():
    print("=" * 80)
    print("JOINT MAPPING: Contrast dominance × PPI coupling")
    print("=" * 80)

    print("\n[1/4] Building joint (contrast × PPI) table...")
    joint = build_joint_table()
    print(f"  Joint table rows: {len(joint)}")
    print(f"  Columns: {list(joint.columns)}")

    joint_out = os.path.join(OUT_DIR, "joint_mapping_long.csv")
    joint.to_csv(joint_out, index=False)
    print(f"  ✓ Saved long-format joint table to: {joint_out}")

    print("\n[2/4] Summarising PPI by FPNA/FPNB-dominant contrasts (Idea A)...")
    summary_a = summarize_ppi_by_dominance(joint)
    summary_a_out = os.path.join(OUT_DIR, "joint_mapping_PPI_by_dominance.csv")
    summary_a.to_csv(summary_a_out, index=False)
    print(f"  ✓ Saved PPI-by-dominance summary to: {summary_a_out}")

    print("\n[3/4] Correlating mean_diff_a_minus_b with PPI betas (Idea B)...")
    corr_b = correlate_mean_diff_with_ppi(joint)
    corr_b_out = os.path.join(OUT_DIR, "joint_mapping_mean_diff_vs_PPI_correlations.csv")
    corr_b.to_csv(corr_b_out, index=False)
    print(f"  ✓ Saved correlation summary to: {corr_b_out}")

    print("\n[4/4] SUMMARY PRINT")
    print("\nCorrelation results (Idea B):")
    if not corr_b.empty:
        for _, row in corr_b.iterrows():
            print(
                f"  {row['seed_family']}–{row['target']}: "
                f"r={row['corr_r']:.3f}, p={row['p_value']:.3g}, n={int(row['n_points'])}"
            )
    else:
        print("  No valid correlations computed (insufficient data).")

    print("\nPPI-by-dominance summary (Idea A):")
    if not summary_a.empty:
        print(summary_a.sort_values(["dominant_network_group", "seed_family", "target"]))
    else:
        print("  No PPI-by-dominance summary (check dominant_network values).")

if __name__ == "__main__":
    main()