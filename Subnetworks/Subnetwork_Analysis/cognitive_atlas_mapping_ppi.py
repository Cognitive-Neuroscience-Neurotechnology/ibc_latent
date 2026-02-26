import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

"""
Map PPI contrast results to cognitive domains using Cognitive Atlas tags.
Creates domain-level summary analogous to ppi_dmn_dan_top_contrasts.csv.

Input:
    - /ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/ppi_results_dmn_dan/group_analysis/ppi_dmn_dan_contrasts_group_summary.csv

Output:
    - /ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/ppi_results_dmn_dan/group_analysis/ppi_dmn_dan_top_domains.csv
    - /ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/ppi_results_dmn_dan/group_analysis/domain_barcharts_polarity.png
"""


def parse_tags(tag_str):
    """Convert string representation of list to actual list of tags."""
    if pd.isna(tag_str):
        return []
    try:
        tags = ast.literal_eval(tag_str)
        return tags if isinstance(tags, list) else []
    except:
        # Fallback: split by comma and clean
        return [t.strip().strip("'\"") for t in str(tag_str).strip('[]').split(',') if t.strip()]


def compute_domain_statistics(domain_df):
    """
    Compute aggregate statistics for a domain across contrasts.
    
    Returns: dict with mean_effect, cohens_d, t_statistic, p_value, n_contrasts
    """
    effects = domain_df['mean_effect'].values
    n = len(effects)
    
    if n == 0:
        return None
    
    mean_effect = float(np.mean(effects))
    
    if n == 1:
        return {
            'mean_effect': mean_effect,
            'median_effect': mean_effect,
            'std_effect': np.nan,
            'cohens_d': np.nan,
            't_statistic': np.nan,
            'p_value': np.nan,
            'n_contrasts': n,
        }
    
    # Statistics across contrasts
    median_effect = float(np.median(effects))
    std_effect = float(np.std(effects, ddof=1))
    
    # One-sample t-test against zero
    t_stat, p_val = stats.ttest_1samp(effects, 0.0)
    
    # Cohen's d = mean / std
    cohens_d = mean_effect / std_effect if std_effect > 0 else 0.0
    
    return {
        'mean_effect': mean_effect,
        'median_effect': median_effect,
        'std_effect': std_effect,
        'cohens_d': cohens_d,
        't_statistic': float(t_stat),
        'p_value': float(p_val),
        'n_contrasts': n,
    }


def _plot_polarity_for_pair(ax_pos, ax_neg, pair_data, title_prefix, xlim=4.0):
    """
    Helper to plot positive/negative domain bars for a given seed_target selection.

    POSITIVE (left axis, ax_pos):
        - x-axis: 0 .. +xlim
        - bar length = d (capped to xlim)
    NEGATIVE (right axis, ax_neg):
        - x-axis: 0 .. +xlim
        - bar length = |d| (capped to xlim)

    Out-of-range values are annotated with the true d value.
    """
    # Ensure finite d
    pair_data = pair_data[np.isfinite(pair_data['cohens_d'])]

    # POSITIVE: Top 10 by d (largest first)
    pos_effects = (
        pair_data[pair_data['cohens_d'] > 0]
        .sort_values('cohens_d', ascending=False)
        .head(10)
    )
    # NEGATIVE: Top 10 by d (most negative first)
    neg_effects = (
        pair_data[pair_data['cohens_d'] < 0]
        .sort_values('cohens_d', ascending=True)
        .head(10)
    )

    # ---- Positive panel (LEFT, ax_pos) ----
    if len(pos_effects) > 0:
        y_pos = range(len(pos_effects))
        y_labels_pos = [row['domain'] for _, row in pos_effects.iterrows()]
        colors_pos = ['darkgreen'] * len(pos_effects)

        d_vals = pos_effects['cohens_d'].values
        d_capped = np.clip(d_vals, 0, xlim)

        ax_pos.barh(
            y_pos,
            d_capped,
            color=colors_pos,
            alpha=0.7,
            edgecolor='black',
        )
        ax_pos.set_yticks(y_pos)
        ax_pos.set_yticklabels(y_labels_pos, fontsize=8)

        # Significance markers (placed near bar end)
        for i, (_, row) in enumerate(pos_effects.iterrows()):
            if row['p_value'] < 0.001:
                mark = '***'
            elif row['p_value'] < 0.01:
                mark = '**'
            elif row['p_value'] < 0.05:
                mark = '*'
            else:
                continue
            ax_pos.text(
                d_capped[i],
                i,
                f" {mark}",
                va='center',
                fontsize=8,
                fontweight='bold',
            )

        # Out-of-range labels
        for i, d_true in enumerate(d_vals):
            if d_true > xlim:
                ax_pos.text(
                    xlim + 0.05 * xlim,
                    i,
                    f"d={d_true:.2f}",
                    va='center',
                    ha='left',
                    fontsize=7,
                    color='black',
                )

        ax_pos.set_xlabel("Effect size (Cohen's d)", fontsize=10)
        ax_pos.grid(axis='x', alpha=0.3)
        ax_pos.set_xlim(left=0, right=xlim)
        ax_pos.invert_yaxis()
    else:
        ax_pos.text(
            0.5,
            0.5,
            'No positive significant domains',
            transform=ax_pos.transAxes,
            ha='center',
            va='center',
            fontsize=10,
        )
        ax_pos.set_xlim(left=0, right=xlim)

    # ---- Negative panel (RIGHT, ax_neg) ----
    if len(neg_effects) > 0:
        y_neg = range(len(neg_effects))
        y_labels_neg = [row['domain'] for _, row in neg_effects.iterrows()]
        colors_neg = ['darkred'] * len(neg_effects)

        d_vals = neg_effects['cohens_d'].values
        # Use absolute value for bar length, but keep sign for annotation
        d_abs = np.abs(d_vals)
        d_capped = np.clip(d_abs, 0, xlim)

        ax_neg.barh(
            y_neg,
            d_capped,
            color=colors_neg,
            alpha=0.7,
            edgecolor='black',
        )
        ax_neg.set_yticks(y_neg)
        ax_neg.set_yticklabels(y_labels_neg, fontsize=8)

        for i, (_, row) in enumerate(neg_effects.iterrows()):
            if row['p_value'] < 0.001:
                mark = '***'
            elif row['p_value'] < 0.01:
                mark = '**'
            elif row['p_value'] < 0.05:
                mark = '*'
            else:
                continue
            ax_neg.text(
                d_capped[i],
                i,
                f" {mark}",
                va='center',
                fontsize=8,
                fontweight='bold',
            )

        # Out-of-range labels (true d is negative)
        for i, d_true in enumerate(d_vals):
            if d_abs[i] > xlim:
                ax_neg.text(
                    xlim + 0.05 * xlim,
                    i,
                    f"d={d_true:.2f}",
                    va='center',
                    ha='left',
                    fontsize=7,
                    color='black',
                )

        ax_neg.set_xlabel("Effect size |Cohen's d|", fontsize=10)
        ax_neg.grid(axis='x', alpha=0.3)
        ax_neg.set_xlim(left=0, right=xlim)
        ax_neg.invert_yaxis()
    else:
        ax_neg.text(
            0.5,
            0.5,
            'No negative significant domains',
            transform=ax_neg.transAxes,
            ha='center',
            va='center',
            fontsize=10,
        )
        ax_neg.set_xlim(left=0, right=xlim)

    # Titles are set by caller


def _create_domain_polarity_barcharts(domain_summary: pd.DataFrame, output_dir: str) -> None:
    """
    Create separate bar charts for DMN and DAN targets:
      - Figure DMN: FPNA→DMN (top) and FPNB→DMN (bottom)
      - Figure DAN: FPNA→DAN (top) and FPNB→DAN (bottom)

    For each row:
      - LEFT panel: top 10 positive domains (p<0.05), bars from 0..+4
      - RIGHT panel: top 10 negative domains (p<0.05), bars from 0..+4 (|d|)
    """
    sig = domain_summary[domain_summary['p_value'] < 0.05].copy()
    if sig.empty:
        print("  No significant domains (p<0.05); skipping polarity bar-charts.")
        return

    def _make_target_figure(target_label: str, out_name: str):
        sub = sig[sig['target'] == target_label].copy()
        if sub.empty:
            print(f"  No significant domains for target {target_label}; skipping.")
            return

        fpna_rows = sub[sub['seed'].str.contains("FPNA", na=False)]
        fpnb_rows = sub[sub['seed'].str.contains("FPNB", na=False)]

        if fpna_rows.empty and fpnb_rows.empty:
            print(f"  No FPNA/FPNB seed rows for target {target_label}; skipping.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), squeeze=False)

        # Row 0: FPNA
        if not fpna_rows.empty:
            ax_pos = axes[0, 0]  # LEFT: positive
            ax_neg = axes[0, 1]  # RIGHT: negative
            _plot_polarity_for_pair(
                ax_pos=ax_pos,
                ax_neg=ax_neg,
                pair_data=fpna_rows,
                title_prefix=f"FPNA→{target_label}",
            )
            ax_pos.set_title(
                f"FPNA→{target_label}: INCREASED Connectivity (Top 10, p<0.05)",
                fontsize=11,
                fontweight='bold',
            )
            ax_neg.set_title(
                f"FPNA→{target_label}: DECREASED Connectivity (Top 10, p<0.05)",
                fontsize=11,
                fontweight='bold',
            )
        else:
            axes[0, 0].text(
                0.5, 0.5,
                f"No FPNA significant domains for {target_label}",
                transform=axes[0, 0].transAxes,
                ha='center',
                va='center',
            )
            axes[0, 1].axis('off')

        # Row 1: FPNB
        if not fpnb_rows.empty:
            ax_pos = axes[1, 0]  # LEFT: positive
            ax_neg = axes[1, 1]  # RIGHT: negative
            _plot_polarity_for_pair(
                ax_pos=ax_pos,
                ax_neg=ax_neg,
                pair_data=fpnb_rows,
                title_prefix=f"FPNB→{target_label}",
            )
            ax_pos.set_title(
                f"FPNB→{target_label}: INCREASED Connectivity (Top 10, p<0.05)",
                fontsize=11,
                fontweight='bold',
            )
            ax_neg.set_title(
                f"FPNB→{target_label}: DECREASED Connectivity (Top 10, p<0.05)",
                fontsize=11,
                fontweight='bold',
            )
        else:
            axes[1, 0].text(
                0.5, 0.5,
                f"No FPNB significant domains for {target_label}",
                transform=axes[1, 0].transAxes,
                ha='center',
                va='center',
            )
            axes[1, 1].axis('off')

        fig.suptitle(
            f"Cognitive Domains – {target_label} (FPNA/FPNB, p<0.05)",
            fontsize=13,
            fontweight='bold',
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(output_dir, out_name)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved domain polarity bar-charts for {target_label} to {out_path}")

    # Create DMN and DAN figures
    _make_target_figure("DMN", "domain_barcharts_polarity_DMN.png")
    _make_target_figure("DAN", "domain_barcharts_polarity_DAN.png")


def _compare_fpna_fpnb_domains(domain_summary: pd.DataFrame, output_dir: str) -> None:
    """
    Compare FPNA vs FPNB at the domain level.
    Aggregates per seed_family (FPNA/FPNB) × target × domain and
    produces a table with both sides side-by-side, plus a brief printed summary.
    """
    df = domain_summary.copy()

    df["seed_family"] = df["seed"].str.extract(r"(FPN[AB])", expand=False)
    df = df[df["seed_family"].isin(["FPNA", "FPNB"])].copy()
    if df.empty:
        print("  No FPNA/FPNB seeds found in domain_summary; skipping FPNA/FPNB domain comparison.")
        return

    grouped = (
        df.groupby(["seed_family", "target", "domain"])
        .agg(
            mean_effect=("mean_effect", "mean"),
            cohens_d=("cohens_d", "mean"),
            best_p_value=("p_value", "min"),
            n_contrasts=("n_contrasts", "sum"),
        )
        .reset_index()
    )

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

    merged = fpna.merge(fpnb, on=["target", "domain"], how="outer")

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

    out_file = os.path.join(output_dir, "ppi_dmn_dan_FPNA_vs_FPNB_domains.csv")
    merged.to_csv(out_file, index=False)
    print(f"  ✓ Saved FPNA vs FPNB domain comparison to {out_file}")

    print("\n=== FPNA vs FPNB domain overlap (alpha = 0.05) ===")
    for target in sorted(merged["target"].dropna().unique()):
        sub = merged[merged["target"] == target]
        n_total = len(sub)
        n_sig_fpna = sub["sig_FPNA"].sum()
        n_sig_fpnb = sub["sig_FPNB"].sum()
        n_shared = sub["shared_sig_same_dir"].sum()

        print(f"\nTarget: {target}")
        print(f"  Domains (any side): {n_total}")
        print(f"  Significant FPNA domains: {n_sig_fpna}")
        print(f"  Significant FPNB domains: {n_sig_fpnb}")
        print(f"  Shared significant domains (same direction): {n_shared}")

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
            print("  Top shared domains by |d|:")
            for _, row in top_shared.iterrows():
                print(
                    f"    - {row['domain']}: "
                    f"d_FPNA={row['cohens_d_FPNA']:.3f}, "
                    f"d_FPNB={row['cohens_d_FPNB']:.3f}, "
                    f"p_FPNA={row['best_p_FPNA']:.3g}, "
                    f"p_FPNB={row['best_p_FPNB']:.3g}"
                )


def _create_domain_boxplot_2x2(
    domain_summary: pd.DataFrame,
    expanded_df: pd.DataFrame,
    output_dir: str,
    top_n: int = 15,
    xlim: float = 4.0,
) -> None:
    """
    2×2 figure with horizontal boxplots + jittered dots for PPI domains.

    Now driven by the same aggregated data used in ppi_domain_boxplot_table_2x2_summary:
      - compute aggregated (seed, target, domain) table from expanded_df,
      - for each panel, select top-N domains by |cohens_d|,
      - within each panel, order domains so most negative at bottom, most positive at top,
      - plot contrast-level cohens_d values for those domains.
    """
    print("\n[ALT] Creating 2×2 domain boxplot figure (FPNA/FPNB × DMN/DAN)...")

    if expanded_df.empty:
        print("  ✗ expanded_df is empty; skipping boxplot figure.")
        return

    # Build aggregated 2×2 table (identical to CSV source)
    panel_table = _build_panel_table_2x2(expanded_df)
    if panel_table.empty:
        print("  ✗ 2×2 panel table is empty; skipping boxplot figure.")
        return

    df_exp = expanded_df[np.isfinite(expanded_df['cohens_d'])].copy()
    df_exp['seed_family'] = df_exp['seed'].str.extract(r'(FPN[AB])', expand=False)
    df_exp = df_exp[df_exp['seed_family'].isin(['FPNA', 'FPNB'])]
    df_exp = df_exp[df_exp['target'].isin(['DMN', 'DAN'])]

    combos = [
        ('FPNA', 'DMN'),
        ('FPNB', 'DMN'),
        ('FPNA', 'DAN'),
        ('FPNB', 'DAN'),
    ]

    panel_data: dict[tuple[str, str], pd.DataFrame | None] = {}
    for seed_family, target in combos:
        sub_table = panel_table[
            (panel_table['seed'] == seed_family) & (panel_table['target'] == target)
        ].copy()
        if sub_table.empty:
            panel_data[(seed_family, target)] = None
            print(f"  [ALT] No domains for {seed_family}→{target}.")
            continue

        # Select top-N by |cohens_d|
        sub_table['abs_d'] = sub_table['cohens_d'].abs()
        sub_table = sub_table.sort_values('abs_d', ascending=False).head(top_n)
        top_domains = sub_table['domain'].tolist()

        sub_exp = df_exp[
            (df_exp['seed_family'] == seed_family)
            & (df_exp['target'] == target)
            & (df_exp['domain'].isin(top_domains))
        ].copy()

        if sub_exp.empty:
            panel_data[(seed_family, target)] = None
            print(f"  [ALT] No contrast-level data for selected domains in {seed_family}→{target}.")
        else:
            panel_data[(seed_family, target)] = sub_exp
            print(
                f"  [ALT] {seed_family}→{target}: "
                f"{len(top_domains)} domains, {sub_exp['domain'].nunique()} with data, "
                f"{len(sub_exp)} contrast-domain points."
            )

    if all(v is None for v in panel_data.values()):
        print("  ✗ No panels had data; skipping boxplot figure.")
        return

    import matplotlib as mpl
    mpl.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), squeeze=False)
    fig.suptitle(
        "PPI Cognitive Domains – Cohen's d across contrasts\n"
        "(selection: top |d| domains from aggregated 2×2 table)",
        fontsize=15,
        fontweight='bold',
    )

    def _plot_panel(ax, sub: pd.DataFrame | None, seed_family: str, target: str):
        if sub is None or sub.empty:
            ax.text(
                0.5, 0.5,
                f"No data for {seed_family}→{target}",
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=11,
            )
            ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
            ax.grid(axis='x', alpha=0.3)
            return

        # Order domains by mean d so:
        #   - negative at bottom, positive at top
        #   -> sort ascending, then invert y-axis
        order = (
            sub.groupby('domain')['cohens_d']
            .mean()
            .sort_values(ascending=True)
            .index.tolist()
        )
        positions = np.arange(len(order))
        plot_data = [sub[sub['domain'] == d]['cohens_d'].values for d in order]

        bp = ax.boxplot(
            plot_data,
            vert=False,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color='black', linewidth=1.5),
            boxprops=dict(linewidth=1.2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )
        for patch in bp['boxes']:
            patch.set_facecolor('#dddddd')
            patch.set_alpha(0.7)

        color = '#f94144' if target == 'DMN' else '#90be6d'

        for i, dname in enumerate(order):
            vals = sub[sub['domain'] == dname]['cohens_d'].values
            y = np.random.normal(loc=positions[i], scale=0.08, size=len(vals))
            ax.scatter(
                vals,
                y,
                s=25,
                color=color,
                edgecolors='black',
                linewidth=0.4,
                alpha=0.7,
                zorder=3,
            )

        ax.set_yticks(positions)
        ax.set_yticklabels(order, fontsize=9)
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()  # ensures most negative at bottom, most positive at top
        ax.set_xlabel("Cohen's d (PPI effect)", fontsize=11)
        ax.set_title(f"{seed_family}→{target}", fontsize=13, fontweight='bold')

    _plot_panel(axes[0, 0], panel_data[('FPNA', 'DMN')], 'FPNA', 'DMN')
    _plot_panel(axes[0, 1], panel_data[('FPNB', 'DMN')], 'FPNB', 'DMN')
    _plot_panel(axes[1, 0], panel_data[('FPNA', 'DAN')], 'FPNA', 'DAN')
    _plot_panel(axes[1, 1], panel_data[('FPNB', 'DAN')], 'FPNB', 'DAN')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(output_dir, "ppi_domain_boxplots_DMN_DAN.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved 2×2 domain boxplot figure to {out_path}")


def _create_domain_boxplot_fpn_1x2(
    domain_summary: pd.DataFrame,
    expanded_df: pd.DataFrame,
    output_dir: str,
    top_n: int = 15,
) -> None:
    """
    1×2 boxplot+dots figure after collapsing FPNA & FPNB into a single 'FPN' family.

    Now driven by the same aggregated data used in ppi_domain_boxplot_table_1x2_summary:
      - compute aggregated (FPN, target, domain) table from expanded_df,
      - for each target, select top-N domains by |cohens_d|,
      - within each panel, order domains so most negative at bottom, most positive at top,
      - plot contrast-level cohens_d values for those domains (FPNA+FPNB).
    """
    print("\n[ALT] Creating 1×2 FPN (FPNA+FPNB) domain boxplot figure (DMN/DAN)...")

    if expanded_df.empty:
        print("  ✗ expanded_df is empty; skipping FPN boxplot figure.")
        return

    panel_table = _build_panel_table_1x2(expanded_df)
    if panel_table.empty:
        print("  ✗ 1×2 panel table is empty; skipping FPN boxplot figure.")
        return

    df_exp = expanded_df[np.isfinite(expanded_df['cohens_d'])].copy()
    df_exp['seed_family'] = df_exp['seed'].str.extract(r'(FPN[AB])', expand=False)
    df_exp = df_exp[df_exp['seed_family'].isin(['FPNA', 'FPNB'])].copy()
    df_exp = df_exp[df_exp['target'].isin(['DMN', 'DAN'])].copy()
    if df_exp.empty:
        print("  ✗ No FPNA/FPNB × DMN/DAN rows; skipping FPN boxplot figure.")
        return

    df_exp['seed_family'] = 'FPN'

    targets = ['DMN', 'DAN']
    panel_data: dict[str, pd.DataFrame | None] = {}

    for target in targets:
        sub_table = panel_table[panel_table['target'] == target].copy()
        if sub_table.empty:
            panel_data[target] = None
            print(f"  [ALT-FPN] No domains for FPN→{target}.")
            continue

        sub_table['abs_d'] = sub_table['cohens_d'].abs()
        sub_table = sub_table.sort_values('abs_d', ascending=False).head(top_n)
        top_domains = sub_table['domain'].tolist()

        sub_exp = df_exp[
            (df_exp['target'] == target)
            & (df_exp['domain'].isin(top_domains))
        ].copy()

        if sub_exp.empty:
            panel_data[target] = None
            print(f"  [ALT-FPN] No contrast-level data for selected domains in FPN→{target}.")
        else:
            panel_data[target] = sub_exp
            print(
                f"  [ALT-FPN] FPN→{target}: "
                f"{len(top_domains)} domains, {sub_exp['domain'].nunique()} with data, "
                f"{len(sub_exp)} contrast-domain points."
            )

    if all(v is None for v in panel_data.values()):
        print("  ✗ No panels had data; skipping FPN boxplot figure.")
        return

    import matplotlib as mpl
    mpl.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 12,
    })

    fig, axes = plt.subplots(2, 1, figsize=(8, 12), squeeze=False)
    fig.suptitle(
        "PPI Cognitive Domains – Cohen's d across contrasts (FPN = FPNA+FPNB)\n"
        "(selection: top |d| domains from aggregated FPN table)",
        fontsize=15,
        fontweight='bold',
    )

    def _plot_panel(ax, sub: pd.DataFrame | None, target: str):
        if sub is None or sub.empty:
            ax.text(
                0.5, 0.5,
                f"No data for FPN→{target}",
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=11,
            )
            ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
            ax.grid(axis='x', alpha=0.3)
            return

        order = (
            sub.groupby('domain')['cohens_d']
            .mean()
            .sort_values(ascending=True)
            .index.tolist()
        )
        positions = np.arange(len(order))
        plot_data = [sub[sub['domain'] == d]['cohens_d'].values for d in order]

        bp = ax.boxplot(
            plot_data,
            vert=False,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color='black', linewidth=1.5),
            boxprops=dict(linewidth=1.2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )
        for patch in bp['boxes']:
            patch.set_facecolor('#dddddd')
            patch.set_alpha(0.7)

        color = '#f94144' if target == 'DMN' else '#90be6d'

        for i, dname in enumerate(order):
            vals = sub[sub['domain'] == dname]['cohens_d'].values
            y = np.random.normal(loc=positions[i], scale=0.08, size=len(vals))
            ax.scatter(
                vals,
                y,
                s=25,
                color=color,
                edgecolors='black',
                linewidth=0.4,
                alpha=0.7,
                zorder=3,
            )

        ax.set_yticks(positions)
        ax.set_yticklabels(order, fontsize=9)
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()  # negative at bottom, positive at top
        ax.set_xlabel("Cohen's d (PPI effect)", fontsize=11)
        ax.set_title(f"FPN→{target}", fontsize=13, fontweight='bold')

    _plot_panel(axes[0, 0], panel_data['DMN'], 'DMN')
    _plot_panel(axes[1, 0], panel_data['DAN'], 'DAN')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(output_dir, "ppi_domain_boxplots_FPN_DMN_DAN.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved FPN 1×2 domain boxplot figure to {out_path}")


def _export_domain_boxplot_tables_2x2(
    domain_summary: pd.DataFrame,
    expanded_df: pd.DataFrame,
    output_dir: str,
    top_n: int = 15,
) -> None:
    print("\n[ALT] Exporting 2×2 domain boxplot summary table (FPNA/FPNB × DMN/DAN, no filtering)...")

    table_2x2 = _build_panel_table_2x2(expanded_df)
    if table_2x2.empty:
        print("  ✗ 2×2 panel table is empty; skipping export.")
        return

    out_path = os.path.join(output_dir, "ppi_domain_boxplot_table_2x2_summary.csv")
    table_2x2.to_csv(out_path, index=False)
    print(f"  ✓ Saved 2×2 boxplot summary table (all domains) to {out_path}")


def _export_domain_boxplot_tables_1x2(
    domain_summary: pd.DataFrame,
    expanded_df: pd.DataFrame,
    output_dir: str,
    top_n: int = 15,
) -> None:
    print("\n[ALT] Exporting 1×2 domain boxplot summary table (FPN = FPNA+FPNB × DMN/DAN, no filtering)...")

    table_1x2 = _build_panel_table_1x2(expanded_df)
    if table_1x2.empty:
        print("  ✗ 1×2 panel table is empty; skipping export.")
        return

    out_path = os.path.join(output_dir, "ppi_domain_boxplot_table_1x2_summary.csv")
    table_1x2.to_csv(out_path, index=False)
    print(f"  ✓ Saved 1×2 boxplot summary table (FPN collapsed) to {out_path}")

def _build_panel_table_2x2(
    expanded_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build aggregated domain table for 2×2 FPNA/FPNB × DMN/DAN layout.

    Returns one row per (seed, target, domain) with:
      seed ∈ {FPNA, FPNB}, target ∈ {DMN, DAN},
      mean_effect, cohens_d, p_value, task_conditions, n_contrasts.
    No filtering is applied here.
    """
    if expanded_df.empty:
        return pd.DataFrame()

    required_cols = {
        'seed', 'target', 'domain', 'cohens_d', 'task_condition', 'mean_effect', 'p_value'
    }
    if not required_cols.issubset(expanded_df.columns):
        return pd.DataFrame()

    df_exp = expanded_df[np.isfinite(expanded_df['cohens_d'])].copy()
    if df_exp.empty:
        return pd.DataFrame()

    df_exp['seed_family'] = df_exp['seed'].str.extract(r'(FPN[AB])', expand=False)
    df_exp = df_exp[df_exp['seed_family'].isin(['FPNA', 'FPNB'])].copy()
    df_exp = df_exp[df_exp['target'].isin(['DMN', 'DAN'])].copy()
    if df_exp.empty:
        return pd.DataFrame()

    grouped = (
        df_exp.groupby(['seed_family', 'target', 'domain'])
        .agg(
            mean_effect=('mean_effect', 'mean'),
            cohens_d=('cohens_d', 'mean'),
            p_value=('p_value', 'mean'),
            task_conditions=('task_condition', lambda x: '; '.join(sorted(set(x)))),
            n_contrasts=('task_condition', lambda x: len(set(x))),
        )
        .reset_index()
    )
    grouped = grouped.rename(columns={'seed_family': 'seed'})
    return grouped


def _build_panel_table_1x2(
    expanded_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build aggregated domain table for 1×2 FPN (FPNA+FPNB) × DMN/DAN layout.

    Returns one row per (seed='FPN', target, domain) with:
      mean_effect, cohens_d, p_value, task_conditions, n_contrasts.
    No filtering is applied here.
    """
    if expanded_df.empty:
        return pd.DataFrame()

    required_cols = {
        'seed', 'target', 'domain', 'cohens_d', 'task_condition', 'mean_effect', 'p_value'
    }
    if not required_cols.issubset(expanded_df.columns):
        return pd.DataFrame()

    df_exp = expanded_df[np.isfinite(expanded_df['cohens_d'])].copy()
    if df_exp.empty:
        return pd.DataFrame()

    df_exp['seed_family'] = df_exp['seed'].str.extract(r'(FPN[AB])', expand=False)
    df_exp = df_exp[df_exp['seed_family'].isin(['FPNA', 'FPNB'])].copy()
    df_exp = df_exp[df_exp['target'].isin(['DMN', 'DAN'])].copy()
    if df_exp.empty:
        return pd.DataFrame()

    df_exp['seed_family'] = 'FPN'

    grouped = (
        df_exp.groupby(['seed_family', 'target', 'domain'])
        .agg(
            mean_effect=('mean_effect', 'mean'),
            cohens_d=('cohens_d', 'mean'),
            p_value=('p_value', 'mean'),
            task_conditions=('task_condition', lambda x: '; '.join(sorted(set(x)))),
            n_contrasts=('task_condition', lambda x: len(set(x))),
        )
        .reset_index()
    )
    grouped = grouped.rename(columns={'seed_family': 'seed'})
    return grouped


def main():
    print("="*70)
    print("PPI CONTRAST → COGNITIVE DOMAIN MAPPING")
    print("="*70)
    
    # Paths
    ppi_base = '/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/ppi_results_dmn_dan'
    group_dir = os.path.join(ppi_base, 'group_analysis')
    summary_file = os.path.join(group_dir, 'ppi_dmn_dan_contrasts_group_summary.csv')
    
    # Load contrast-level summary
    print(f"\n[1/6] Loading contrast-level summary from:\n  {summary_file}")
    
    if not os.path.exists(summary_file):
        print(f"ERROR: File not found: {summary_file}")
        return
    
    contrast_df = pd.read_csv(summary_file)
    print(f"  Loaded {len(contrast_df)} contrast rows")
    print(f"  Columns: {list(contrast_df.columns)}")
    
    # Parse tags
    print("\n[2/6] Parsing cognitive domain tags...")
    contrast_df['parsed_tags'] = contrast_df['tags'].apply(parse_tags)
    contrast_df['n_tags'] = contrast_df['parsed_tags'].apply(len)
    
    n_with_tags = (contrast_df['n_tags'] > 0).sum()
    print(f"  Contrasts with tags: {n_with_tags} / {len(contrast_df)}")
    print(f"  Average tags per contrast: {contrast_df['n_tags'].mean():.1f}")
    
    # Expand: one row per contrast-domain pair
    print("\n[3/6] Expanding contrasts by cognitive domain...")
    expanded_rows = []
    
    for _, row in contrast_df.iterrows():
        if not isinstance(row['parsed_tags'], list) or len(row['parsed_tags']) == 0:
            continue
        
        for domain in row['parsed_tags']:
            expanded_rows.append({
                'seed': row['seed'],
                'target': row['target'],
                'seed_target': row['seed_target'],
                'domain': domain,
                'task': row['task'],
                'condition': row['condition'],
                'task_condition': row['task_condition'],
                'mean_effect': row['mean_effect'],
                'cohens_d': row['cohens_d'],
                'p_value': row['p_value'],
                'n_subjects': row['n_subjects'],
            })
    
    expanded_df = pd.DataFrame(expanded_rows)
    print(f"  Expanded to {len(expanded_df)} domain-contrast pairs")
    print(f"  Unique domains: {expanded_df['domain'].nunique()}")
    
    # Aggregate by seed × target × domain
    print("\n[4/6] Aggregating by seed × target × domain...")
    domain_stats = []
    
    for (seed, target, domain), group in expanded_df.groupby(['seed', 'target', 'domain']):
        stats_dict = compute_domain_statistics(group)
        
        if stats_dict is None:
            continue
        
        # Get representative contrasts for this domain
        top_contrasts = group.nlargest(3, 'cohens_d')['task_condition'].tolist()
        
        domain_stats.append({
            'seed': seed,
            'target': target,
            'seed_target': f"{seed}→{target}",
            'domain': domain,
            'mean_effect': stats_dict['mean_effect'],
            'median_effect': stats_dict['median_effect'],
            'std_effect': stats_dict['std_effect'],
            'cohens_d': stats_dict['cohens_d'],
            't_statistic': stats_dict['t_statistic'],
            'p_value': stats_dict['p_value'],
            'n_contrasts': stats_dict['n_contrasts'],
            'n_tasks': group['task'].nunique(),
            'example_contrasts': '; '.join(top_contrasts[:3]),
        })
    
    domain_summary = pd.DataFrame(domain_stats)
    print(f"  Created {len(domain_summary)} seed-target-domain combinations")
    
    # Filter significant domains (p<0.05)
    print("\n[5/6] Filtering and sorting significant domains...")
    top_domains = domain_summary[domain_summary['p_value'] < 0.05].copy()
    
    # Sort by target, seed, then Cohen's d (descending)
    top_domains = top_domains.sort_values(
        by=['target', 'seed', 'cohens_d'],
        ascending=[True, True, False]
    )
    
    print(f"  Significant domains (p<0.05): {len(top_domains)} / {len(domain_summary)}")
    
    # Save outputs
    output_file = os.path.join(group_dir, 'ppi_dmn_dan_top_domains.csv')
    top_domains.to_csv(output_file, index=False)
    print(f"\n✓ Saved top domains to:\n  {output_file}")
    
    # Also save full domain summary (all domains, not just significant)
    full_output = os.path.join(group_dir, 'ppi_dmn_dan_all_domains.csv')
    domain_summary.to_csv(full_output, index=False)
    print(f"✓ Saved all domains to:\n  {full_output}")
    
    # NEW: FPNA vs FPNB domain comparison
    print("\n[6/6] Comparing FPNA vs FPNB at domain level...")
    _compare_fpna_fpnb_domains(domain_summary, group_dir)

    # Create domain polarity bar-chart figure(s)
    print("\nCreating domain polarity bar-chart plots...")
    _create_domain_polarity_barcharts(domain_summary, group_dir)

    # Alternate boxplot-based domain visualization (consistent selection)
    print("\n[ALT] Creating domain-level boxplot+dots figure (DMN/DAN, FPNA/FPNB)...")
    _create_domain_boxplot_2x2(domain_summary, expanded_df, group_dir, top_n=15, xlim=4.0)

    # New FPN (FPNA+FPNB) boxplot visualization
    print("\n[ALT] Creating FPN (FPNA+FPNB) domain boxplot figure (DMN/DAN)...")
    _create_domain_boxplot_fpn_1x2(domain_summary, expanded_df, group_dir, top_n=15)

    # Export boxplot tables
    print("\n[ALT] Exporting 2×2 domain boxplot tables (FPNA/FPNB × DMN/DAN)...")
    _export_domain_boxplot_tables_2x2(domain_summary, expanded_df, group_dir, top_n=15)

    print("\n[ALT] Exporting 1×2 domain boxplot tables (FPN = FPNA+FPNB × DMN/DAN)...")
    _export_domain_boxplot_tables_1x2(domain_summary, expanded_df, group_dir, top_n=15)
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for seed_target in sorted(top_domains['seed_target'].unique()):
        st_data = top_domains[top_domains['seed_target'] == seed_target]
        print(f"\n{seed_target}:")
        print(f"  Significant domains: {len(st_data)}")
        
        if len(st_data) > 0:
            top_3 = st_data.head(3)
            print(f"  Top 3 domains by Cohen's d:")
            for idx, (_, row) in enumerate(top_3.iterrows(), 1):
                print(f"    {idx}. {row['domain']} (d={row['cohens_d']:.3f}, p={row['p_value']:.4f}, n={row['n_contrasts']})")


if __name__ == "__main__":
    main()