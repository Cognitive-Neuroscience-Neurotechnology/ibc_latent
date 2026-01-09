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
    - /ptmp/hmueller2/Downloads/ppi_results_dmn_dan/group_analysis/ppi_dmn_dan_contrasts_group_summary.csv

Output:
    - /ptmp/hmueller2/Downloads/ppi_results_dmn_dan/group_analysis/ppi_dmn_dan_top_domains.csv
    - /ptmp/hmueller2/Downloads/ppi_results_dmn_dan/group_analysis/domain_barcharts_polarity.png
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


def _create_domain_polarity_barcharts(domain_summary: pd.DataFrame, output_dir: str) -> None:
    """
    Create per-seed_target bar charts for positive vs negative domain effects,
    restricted to significant domains (p<0.05), sorted by Cohen's d.
    Show top 10 positive and top 10 negative domains for each seed_target.
    """
    # Keep only significant domains
    sig = domain_summary[domain_summary['p_value'] < 0.05].copy()
    if sig.empty:
        print("  No significant domains (p<0.05); skipping polarity bar-charts.")
        return

    seed_target_pairs = sorted(sig['seed_target'].unique())
    n_pairs = len(seed_target_pairs)
    if n_pairs == 0:
        print("  No seed_target pairs in significant domains; skipping plots.")
        return

    fig, axes = plt.subplots(n_pairs, 2, figsize=(16, 4 * n_pairs), squeeze=False)

    for idx, pair in enumerate(seed_target_pairs):
        pair_data = sig[sig['seed_target'] == pair]

        # Ensure we only consider rows with finite Cohen's d
        pair_data = pair_data[np.isfinite(pair_data['cohens_d'])]

        # LEFT: Top 10 positive effects by Cohen's d (largest d first)
        ax_pos = axes[idx, 0]
        pos_effects = (
            pair_data[pair_data['cohens_d'] > 0]
            .sort_values('cohens_d', ascending=False)
            .head(10)
        )

        if len(pos_effects) > 0:
            y_labels = [row['domain'] for _, row in pos_effects.iterrows()]
            colors = ['darkgreen'] * len(pos_effects)

            ax_pos.barh(
                range(len(pos_effects)),
                pos_effects['cohens_d'],
                color=colors,
                alpha=0.7,
                edgecolor='black',
            )
            ax_pos.set_yticks(range(len(pos_effects)))
            ax_pos.set_yticklabels(y_labels, fontsize=8)

            # Add significance markers
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
                    row['cohens_d'],
                    i,
                    f" {mark}",
                    va='center',
                    fontsize=8,
                    fontweight='bold',
                )

            ax_pos.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax_pos.set_xlabel("Effect size (Cohen's d)", fontsize=10)
            ax_pos.set_title(
                f"{pair}: INCREASED Connectivity - Cognitive Domains (Top 10, p<0.05)",
                fontsize=11,
                fontweight='bold',
            )
            ax_pos.grid(axis='x', alpha=0.3)
            ax_pos.set_xlim(left=0)
            ax_pos.invert_yaxis()  # largest effect at top
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
            ax_pos.set_title(
                f"{pair}: INCREASED Connectivity", fontsize=11, fontweight='bold'
            )

        # RIGHT: Top 10 negative effects by Cohen's d (most negative d first)
        ax_neg = axes[idx, 1]
        neg_effects = (
            pair_data[pair_data['cohens_d'] < 0]
            .sort_values('cohens_d', ascending=True)  # more negative d at top
            .head(10)
        )

        if len(neg_effects) > 0:
            y_labels = [row['domain'] for _, row in neg_effects.iterrows()]
            colors = ['darkred'] * len(neg_effects)

            ax_neg.barh(
                range(len(neg_effects)),
                neg_effects['cohens_d'],
                color=colors,
                alpha=0.7,
                edgecolor='black',
            )
            ax_neg.set_yticks(range(len(neg_effects)))
            ax_neg.set_yticklabels(y_labels, fontsize=8)

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
                    row['cohens_d'],
                    i,
                    f" {mark}",
                    va='center',
                    fontsize=8,
                    fontweight='bold',
                )

            ax_neg.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax_neg.set_xlabel("Effect size (Cohen's d)", fontsize=10)
            ax_neg.set_title(
                f"{pair}: DECREASED Connectivity - Cognitive Domains (Top 10, p<0.05)",
                fontsize=11,
                fontweight='bold',
            )
            ax_neg.grid(axis='x', alpha=0.3)
            ax_neg.set_xlim(right=0)
            ax_neg.invert_yaxis()  # most negative at top
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
            ax_neg.set_title(
                f"{pair}: DECREASED Connectivity", fontsize=11, fontweight='bold'
            )

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'domain_barcharts_polarity.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved domain polarity bar-charts to {out_path}")


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


def main():
    print("="*70)
    print("PPI CONTRAST → COGNITIVE DOMAIN MAPPING")
    print("="*70)
    
    # Paths
    ppi_base = '/ptmp/hmueller2/Downloads/ppi_results_dmn_dan'
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

    # Create domain polarity bar-chart figure
    print("\nCreating domain polarity bar-chart plots...")
    _create_domain_polarity_barcharts(domain_summary, group_dir)
    
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