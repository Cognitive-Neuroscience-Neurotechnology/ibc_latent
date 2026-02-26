# filepath: /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/ppi_fpn_flexible_hub_summary.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Map PPI contrast results to cognitive domains using Cognitive Atlas tags.
Creates domain-level summary analogous to ppi_dmn_dan_top_contrasts.csv.

Input:
    - /ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/ppi_results_dmn_dan/group_analysis/ppi_dmn_dan_FPNA_vs_FPNB_contrasts.csv
    - /ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/ppi_results_dmn_dan/group_analysis/ppi_dmn_dan_FPNA_vs_FPNB_domains.csv

Output:
    - ppi_dmn_dan_FPNA_FPNB_pattern_summary.csv 
        -> Counts of FPNA/FPNB patterns (both_recouple, both_decouple, mixed, FPNA_only, FPNB_only) per target, for contrasts and domains.
    - ppi_dmn_dan_FPNA_FPNB_DMN_DAN_pattern_summary.csv 
        -> Counts of DMN–DAN relationships (recouple_DAN_decouple_DMN, both_recouple, etc.) per side (FPNA/FPNB), for contrasts and domains.
    - ppi_dmn_dan_FPNA_FPNB_patterns_per_target.png 
        -> Stacked bar chart: how many domains/contrasts fall into each FPNA/FPNB pattern for DMN, DAN, etc. 
    - ppi_dmn_dan_FPNA_FPNB_DMN_DAN_patterns.png 
        -> Bar chart: number of domains/contrasts with each DMN–DAN relationship (e.g. recouple_DAN_decouple_DMN) for FPNA vs FPNB.
"""

PPI_BASE = "/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/ppi_results_dmn_dan"
GROUP_DIR = os.path.join(PPI_BASE, "group_analysis")

ALPHA = 0.05


def _load_fpna_fpnb_tables():
    """Load FPNA vs FPNB comparison tables for contrasts and domains."""
    contrasts_path = os.path.join(GROUP_DIR, "ppi_dmn_dan_FPNA_vs_FPNB_contrasts.csv")
    domains_path = os.path.join(GROUP_DIR, "ppi_dmn_dan_FPNA_vs_FPNB_domains.csv")

    if not os.path.exists(contrasts_path):
        raise FileNotFoundError(f"Missing contrasts table: {contrasts_path}")
    if not os.path.exists(domains_path):
        raise FileNotFoundError(f"Missing domains table: {domains_path}")

    contrasts = pd.read_csv(contrasts_path)
    domains = pd.read_csv(domains_path)

    print(f"Loaded contrasts FPNA/FPNB table: {contrasts_path} ({len(contrasts)} rows)")
    print(f"Loaded domains   FPNA/FPNB table: {domains_path} ({len(domains)} rows)")

    return contrasts, domains


# ---------- HELPERS FOR PATTERN LABELING ----------


def _label_pattern(sig_a, sig_b, dir_a, dir_b):
    """
    Label pattern for two systems (e.g. FPNA/FPNB) given significance and direction.
    Returns one of:
      'both_recouple', 'both_decouple', 'mixed', 'FPNA_only', 'FPNB_only', 'none'
    where 'recouple' means d>0 and 'decouple' d<0.
    """
    if not sig_a and not sig_b:
        return "none"
    if sig_a and not sig_b:
        return "FPNA_only"
    if sig_b and not sig_a:
        return "FPNB_only"

    # Both significant
    if dir_a > 0 and dir_b > 0:
        return "both_recouple"
    if dir_a < 0 and dir_b < 0:
        return "both_decouple"
    return "mixed"


def _add_pattern_columns(df, level="domain"):
    """
    For FPNA/FPNB comparison table (contrasts or domains), add:
      - pattern_FPNA_FPNB: relationship between FPNA and FPNB for same target × item
      - sign_pattern_DMNDAN (added later on merged DMN+DAN tables)
    """
    df = df.copy()
    # Basic pattern between FPNA and FPNB for each row
    df["pattern_FPNA_FPNB"] = df.apply(
        lambda r: _label_pattern(
            r.get("sig_FPNA", False),
            r.get("sig_FPNB", False),
            r.get("direction_FPNA", 0),
            r.get("direction_FPNB", 0),
        ),
        axis=1,
    )
    return df


# ---------- SUMMARY STATS: WITHIN TARGET (DMN / DAN) ----------


def _summarize_patterns_within_target(df, id_col, table_name):
    """
    Summarize how many items of a given type (contrast/domain) show each pattern
    for each target (DMN, DAN, etc.).

    id_col: 'task_condition' for contrasts, 'domain' for domains.
    """
    print(f"\n=== SUMMARY: {table_name} patterns per target (FPNA vs FPNB) ===")

    results = []
    for target in sorted(df["target"].dropna().unique()):
        sub = df[df["target"] == target]
        counts = sub["pattern_FPNA_FPNB"].value_counts().to_dict()
        total = len(sub)
        print(f"\nTarget: {target} ({total} {id_col}s)")
        for pattern, cnt in sorted(counts.items()):
            print(f"  {pattern:15s}: {cnt:3d}")

        results.append(
            {
                "table": table_name,
                "target": target,
                "n_items": total,
                **{f"n_{k}": v for k, v in counts.items()},
            }
        )

    return pd.DataFrame(results)


def _barplot_patterns_within_target(summary_df, output_prefix):
    """
    Bar plot of counts per pattern (both_recouple, both_decouple, mixed, FPNA_only, FPNB_only)
    for each target, separately for contrasts and domains (if both present).
    """
    if summary_df.empty:
        print("No summary data for within-target patterns; skipping barplot.")
        return

    patterns_order = [
        "both_recouple",
        "both_decouple",
        "mixed",
        "FPNA_only",
        "FPNB_only",
    ]

    tables = sorted(summary_df["table"].unique())
    n_tables = len(tables)
    fig, axes = plt.subplots(1, n_tables, figsize=(6 * n_tables, 4), squeeze=False)

    for idx, table in enumerate(tables):
        ax = axes[0, idx]
        sub = summary_df[summary_df["table"] == table].copy()

        # Build matrix: rows = target, columns = patterns
        rows = []
        for _, row in sub.iterrows():
            counts = []
            for p in patterns_order:
                col_name = f"n_{p}"
                counts.append(row.get(col_name, 0))
            rows.append(counts)

        if not rows:
            continue

        mat = np.array(rows)
        targets = sub["target"].tolist()

        # Stacked bar
        x = np.arange(len(targets))
        bottom = np.zeros(len(targets))
        colors = {
            "both_recouple": "darkgreen",
            "both_decouple": "darkred",
            "mixed": "goldenrod",
            "FPNA_only": "steelblue",
            "FPNB_only": "orange",
        }

        for j, p in enumerate(patterns_order):
            ax.bar(
                x,
                mat[:, j],
                bottom=bottom,
                color=colors.get(p, "gray"),
                label=p,
                alpha=0.8,
                edgecolor="black",
            )
            bottom += mat[:, j]

        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=0)
        ax.set_ylabel("Number of items")
        ax.set_title(f"{table}: FPNA vs FPNB patterns per target")
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out = os.path.join(GROUP_DIR, f"{output_prefix}_patterns_per_target.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved pattern summary barplot to {out}")


# ---------- DMN–DAN RELATIONSHIPS FOR SAME ITEM (DOMAIN / CONTRAST) ----------


def _merge_dmn_dan(df, id_col):
    """
    For a FPNA/FPNB table with rows target × id_col, create a wide table
    with separate columns for DMN and DAN, keyed by id_col.

    Returns a dataframe with one row per id_col, containing:
      - all FPNA/FPNB stats for DMN and DAN side by side
    """
    # Split DMN / DAN (and keep other targets if present, but focus on DMN/DAN)
    dmn = df[df["target"] == "DMN"].copy()
    dan = df[df["target"] == "DAN"].copy()

    # Suffix columns to indicate target
    dmn_cols = {c: f"{c}_DMN" for c in dmn.columns if c not in [id_col, "target"]}
    dan_cols = {c: f"{c}_DAN" for c in dan.columns if c not in [id_col, "target"]}

    dmn_ren = dmn.rename(columns=dmn_cols).drop(columns=["target"])
    dan_ren = dan.rename(columns=dan_cols).drop(columns=["target"])

    merged = pd.merge(dmn_ren, dan_ren, on=id_col, how="outer")
    return merged


def _label_dmn_dan_pattern(row, side_prefix, sig_suffix="sig", dir_suffix="direction"):
    """
    Label relationship between DMN and DAN for a given side (FPNA or FPNB).
    side_prefix: 'FPNA' or 'FPNB'
    Uses columns like sig_FPNA_DMN, sig_FPNA_DAN, direction_FPNA_DMN, direction_FPNA_DAN.

    Returns one of:
      'recouple_DAN_decouple_DMN', 'decouple_DAN_recouple_DMN',
      'both_recouple', 'both_decouple', 'mixed', 'none' (if nothing sig).
    """
    sig_dmn = row.get(f"{sig_suffix}_{side_prefix}_DMN", False)
    sig_dan = row.get(f"{sig_suffix}_{side_prefix}_DAN", False)
    dir_dmn = row.get(f"{dir_suffix}_{side_prefix}_DMN", 0)
    dir_dan = row.get(f"{dir_suffix}_{side_prefix}_DAN", 0)

    if not sig_dmn and not sig_dan:
        return "none"
    if sig_dmn and not sig_dan:
        return "DMN_only"
    if sig_dan and not sig_dmn:
        return "DAN_only"

    # Both significant
    if dir_dan > 0 and dir_dmn < 0:
        return "recouple_DAN_decouple_DMN"
    if dir_dan < 0 and dir_dmn > 0:
        return "decouple_DAN_recouple_DMN"
    if dir_dan > 0 and dir_dmn > 0:
        return "both_recouple"
    if dir_dan < 0 and dir_dmn < 0:
        return "both_decouple"
    return "mixed"


def _summarize_dmn_dan_relationships(merged, id_col, table_name):
    """
    Summarize pattern relationships between DMN and DAN for same domain/contrast,
    separately for FPNA and FPNB.
    """
    # Expect columns: sig_FPNA_DMN, sig_FPNA_DAN, direction_FPNA_DMN, direction_FPNA_DAN, etc.
    # Build them if needed (for contrasts table we already have these from original CSV):
    for side in ["FPNA", "FPNB"]:
        # convert best_p_* columns into sig_* booleans if not already present
        for target in ["DMN", "DAN"]:
            best_p_col = f"best_p_{side}_{target}"
            sig_col = f"sig_{side}_{target}"
            dir_col = f"direction_{side}_{target}"
            d_col = f"cohens_d_{side}_{target}"

            if best_p_col in merged.columns and sig_col not in merged.columns:
                merged[sig_col] = merged[best_p_col] < ALPHA
            if d_col in merged.columns and dir_col not in merged.columns:
                merged[dir_col] = np.sign(merged[d_col])

        merged[f"pattern_DMN_DAN_{side}"] = merged.apply(
            lambda r: _label_dmn_dan_pattern(r, side), axis=1
        )

    print(f"\n=== DMN–DAN pattern relationships for {table_name} (per {id_col}) ===")
    results = []
    for side in ["FPNA", "FPNB"]:
        pattern_col = f"pattern_DMN_DAN_{side}"
        counts = merged[pattern_col].value_counts().to_dict()
        print(f"\nSide: {side}")
        for k, v in sorted(counts.items()):
            print(f"  {k:30s}: {v:3d}")
        results.append(
            {
                "table": table_name,
                "side": side,
                **{f"n_{k}": v for k, v in counts.items()},
            }
        )

    return pd.DataFrame(results)


# ---------- PLOTTING DMN–DAN RELATIONSHIPS ----------


def _barplot_dmn_dan_patterns(summary_df, output_prefix):
    """
    Simple barplot for the DMN–DAN relationship patterns for FPNA and FPNB.
    """
    if summary_df.empty:
        print("No DMN–DAN summary data; skipping barplot.")
        return

    patterns_all = set()
    for col in summary_df.columns:
        if col.startswith("n_"):
            patterns_all.add(col[2:])
    patterns_all = sorted(patterns_all)

    sides = sorted(summary_df["side"].unique())
    x = np.arange(len(patterns_all))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(patterns_all) * 1.2), 4))

    for i, side in enumerate(sides):
        sub = summary_df[summary_df["side"] == side]
        # We expect one row per side
        if sub.empty:
            continue
        row = sub.iloc[0]
        counts = [row.get(f"n_{p}", 0) for p in patterns_all]
        ax.bar(
            x + (i - len(sides) / 2) * width + width / 2,
            counts,
            width=width,
            label=side,
            alpha=0.8,
            edgecolor="black",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(patterns_all, rotation=45, ha="right")
    ax.set_ylabel("Number of items (domains/contrasts)")
    ax.set_title("DMN–DAN pattern relationships (FPNA vs FPNB)")
    ax.legend()

    plt.tight_layout()
    out = os.path.join(GROUP_DIR, f"{output_prefix}_dmn_dan_patterns.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved DMN–DAN pattern barplot to {out}")


# ---------- MAIN ----------


def main():
    contrasts, domains = _load_fpna_fpnb_tables()

    # 1) Within-target FPNA/FPNB patterns (both_recouple, both_decouple, mixed, etc.)
    contrasts = _add_pattern_columns(contrasts, level="contrast")
    domains = _add_pattern_columns(domains, level="domain")

    contrast_summary = _summarize_patterns_within_target(
        contrasts, id_col="task_condition", table_name="contrasts"
    )
    domain_summary = _summarize_patterns_within_target(
        domains, id_col="domain", table_name="domains"
    )

    all_summary = pd.concat([contrast_summary, domain_summary], ignore_index=True)
    summary_csv = os.path.join(GROUP_DIR, "ppi_dmn_dan_FPNA_FPNB_pattern_summary.csv")
    all_summary.to_csv(summary_csv, index=False)
    print(f"\n✓ Saved within-target pattern summary to {summary_csv}")

    _barplot_patterns_within_target(all_summary, output_prefix="ppi_dmn_dan_FPNA_FPNB")

    # 2) DMN–DAN relationships for same item (domain / contrast)

    # 2a) Domains: merge DMN+DAN and summarize
    domains_merged = _merge_dmn_dan(domains, id_col="domain")
    domains_dmn_dan_summary = _summarize_dmn_dan_relationships(
        domains_merged, id_col="domain", table_name="domains"
    )

    # 2b) Contrasts: merge DMN+DAN and summarize
    contrasts_merged = _merge_dmn_dan(contrasts, id_col="task_condition")
    contrasts_dmn_dan_summary = _summarize_dmn_dan_relationships(
        contrasts_merged, id_col="task_condition", table_name="contrasts"
    )

    dmn_dan_summary = pd.concat(
        [domains_dmn_dan_summary, contrasts_dmn_dan_summary], ignore_index=True
    )
    dmn_dan_csv = os.path.join(GROUP_DIR, "ppi_dmn_dan_FPNA_FPNB_DMN_DAN_pattern_summary.csv")
    dmn_dan_summary.to_csv(dmn_dan_csv, index=False)
    print(f"\n✓ Saved DMN–DAN pattern summary to {dmn_dan_csv}")

    _barplot_dmn_dan_patterns(dmn_dan_summary, output_prefix="ppi_dmn_dan_FPNA_FPNB")


if __name__ == "__main__":
    main()