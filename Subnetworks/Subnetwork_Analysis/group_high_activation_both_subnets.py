# filepath: /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/group_high_activation_both_subnets.py
"""
Group-level analysis: find contrasts with high activation in BOTH FPN_A and FPN_B.

Inputs:
  - Subject-level outputs from contrast_x_subn_new.py:
      /ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_activation/sub-*/fpn_subnetwork_contrast_analysis.csv

For each task_contrast:
  - Aggregate mean_fpn_a and mean_fpn_b across subjects.
  - Run 1-sample t-tests vs 0 for each subnetwork.
  - Compute Cohen's d vs 0 for each subnetwork.
  - Optionally apply simple thresholds to define "high activation in both".

Outputs (in group_analysis dir):
  - fpn_subnetwork_group_activation_both_raw.csv
      → full per-contrast group stats for FPN_A and FPN_B.
  - fpn_subnetwork_high_activation_both.csv
      → subset of contrasts with strong & significant activation in BOTH.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

# ---------- Helpers ----------

def cohens_d_1sample(values: np.ndarray) -> float:
    """Cohen's d vs 0 across subjects."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    m = float(values.mean())
    s = float(values.std(ddof=1))
    return m / s if s > 0 else 0.0


# ---------- Collect subject-level files ----------

RESULT_PATTERN = "/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_activation/sub-*/fpn_subnetwork_contrast_analysis.csv"
result_files = sorted(glob.glob(RESULT_PATTERN))

print(f"{'='*70}")
print("GROUP-LEVEL: High activation in BOTH FPN_A and FPN_B")
print(f"{'='*70}\n")

if not result_files:
    print("ERROR: No subject result files found!")
    print(f"Expected pattern: {RESULT_PATTERN}")
    raise SystemExit(1)

print(f"Found {len(result_files)} subject files:")
for f in result_files:
    subj = f.split("sub-")[1].split("/")[0]
    print(f"  - sub-{subj}")
print()

all_data = []
for file in result_files:
    df_sub = pd.read_csv(file)
    all_data.append(df_sub)

combined = pd.concat(all_data, ignore_index=True)

print(f"Total observations: {len(combined)}")
print(f"Unique task_contrasts: {combined['task_contrast'].nunique()}")
print(f"Subjects per contrast (should be ~{len(result_files)}):")
print(combined.groupby("task_contrast").size().value_counts(), "\n")

# ---------- Group-level stats per subnetwork ----------

group_rows = []
for task_contrast, g in combined.groupby("task_contrast"):
    # Per-subject mean z in each subnet
    a_vals = g["mean_fpn_a"].dropna().values
    b_vals = g["mean_fpn_b"].dropna().values
    if a_vals.size < 2 or b_vals.size < 2:
        continue

    # 1-sample t vs 0 for each subnet
    t_a, p_a = stats.ttest_1samp(a_vals, 0.0)
    t_b, p_b = stats.ttest_1samp(b_vals, 0.0)

    d_a = cohens_d_1sample(a_vals)
    d_b = cohens_d_1sample(b_vals)

    group_rows.append(
        {
            "task_contrast": task_contrast,
            "task": g["task"].iloc[0],
            "contrast": g["contrast"].iloc[0],
            "n_subjects": int(len(g)),
            # raw group stats per subnet
            "mean_fpn_a_mean": float(a_vals.mean()),
            "mean_fpn_a_std": float(a_vals.std(ddof=1)),
            "t_fpn_a_vs0": float(t_a),
            "p_fpn_a_vs0": float(p_a),
            "cohens_d_fpn_a_vs0": float(d_a),
            "mean_fpn_b_mean": float(b_vals.mean()),
            "mean_fpn_b_std": float(b_vals.std(ddof=1)),
            "t_fpn_b_vs0": float(t_b),
            "p_fpn_b_vs0": float(p_b),
            "cohens_d_fpn_b_vs0": float(d_b),
        }
    )

if not group_rows:
    print("No group rows (not enough subjects per contrast).")
    raise SystemExit(1)

group_df = pd.DataFrame(group_rows)

# ---------- Save full group table ----------

group_out_dir = "/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_activation/group_analysis"
os.makedirs(group_out_dir, exist_ok=True)

raw_path = os.path.join(group_out_dir, "fpn_subnetwork_group_activation_both_raw.csv")
group_df.to_csv(raw_path, index=False)
print(f"✓ Full group activation table saved: {raw_path}")

# ---------- Identify contrasts with high activation in BOTH subnetworks ----------

# Thresholds can be adjusted; start with fairly liberal:
#   - d > 0.5 (medium effect) for both
#   - p < 0.05 (uncorrected) for both
d_thresh = 0.5
p_thresh = 0.05

mask_high_both = (
    (group_df["cohens_d_fpn_a_vs0"] > d_thresh)
    & (group_df["cohens_d_fpn_b_vs0"] > d_thresh)
    & (group_df["p_fpn_a_vs0"] < p_thresh)
    & (group_df["p_fpn_b_vs0"] < p_thresh)
)

high_both = group_df.loc[mask_high_both].copy()

# Sort by how strong both are (e.g., min of the two ds, then sum)
if not high_both.empty:
    high_both["min_d_both"] = high_both[
        ["cohens_d_fpn_a_vs0", "cohens_d_fpn_b_vs0"]
    ].min(axis=1)
    high_both["sum_d_both"] = high_both[
        ["cohens_d_fpn_a_vs0", "cohens_d_fpn_b_vs0"]
    ].sum(axis=1)

    high_both = high_both.sort_values(
        by=["min_d_both", "sum_d_both"], ascending=[False, False]
    )

high_path = os.path.join(group_out_dir, "fpn_subnetwork_high_activation_both.csv")
high_both.to_csv(high_path, index=False)
print(f"✓ High-activation-in-BOTH table saved: {high_path}")

if not high_both.empty:
    print("\nTop 20 contrasts with high activation in BOTH FPN_A and FPN_B:")
    print(
        high_both[
            [
                "task",
                "contrast",
                "cohens_d_fpn_a_vs0",
                "cohens_d_fpn_b_vs0",
                "p_fpn_a_vs0",
                "p_fpn_b_vs0",
                "n_subjects",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
else:
    print("\nNo contrasts passed the (d>{:.2f}, p<{:.3f}) thresholds for BOTH networks.".format(
        d_thresh, p_thresh
    ))

print("\nDone.")