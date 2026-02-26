"""
gPPI analysis to examine task-dependent connectivity between FPN subnetworks and DMN/DAN.
Analyzes how connectivity from each FPN subnetwork (as seed) to DMN and DAN changes across tasks.
OLD!! - Now using _DMN_DAN (change: minimal cleaning as using cleaned GLM data directly)
"""

# ========== 1) IMPORTS AND HELPER FUNCTIONS ==========
import sys
import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel, compute_regressor, run_glm
from nilearn.signal import clean
from scipy import stats

# Helper: robust events file lookup (try with/without dir and fall back to glob)
def find_events_file(subject, session, task, direction=None, run=None):
    base = os.path.join('/ptmp/hmueller2/2025_ibc_latent/data/ibc_raw', f'sub-{subject}', session, 'func')

    candidates = []
    if run:
        # exact candidates (with and without dir)
        if direction:
            candidates.append(os.path.join(base, f'sub-{subject}_{session}_task-{task}_dir-{direction}_run-{run}_events.tsv'))
        candidates.append(os.path.join(base, f'sub-{subject}_{session}_task-{task}_run-{run}_events.tsv'))
        # glob fallback
        candidates.extend(sorted(glob.glob(os.path.join(base, f'sub-{subject}_{session}_task-{task}_*run-{run}*_events.tsv'))))
    else:
        if direction:
            candidates.append(os.path.join(base, f'sub-{subject}_{session}_task-{task}_dir-{direction}_events.tsv'))
        candidates.append(os.path.join(base, f'sub-{subject}_{session}_task-{task}_events.tsv'))
        candidates.extend(sorted(glob.glob(os.path.join(base, f'sub-{subject}_{session}_task-{task}_*_events.tsv'))))

    for c in candidates:
        if os.path.exists(c):
            return c
    return None

# ========== 2) SETUP: LOAD SUBJECT DATA AND PATHS ==========
subject = sys.argv[1]
working_dir = '/ptmp/hmueller2/2025_ibc_latent/outputs'
base_dir = os.path.join(working_dir, 'fmriprep_out_october')
subnetwork_dir = os.path.join(working_dir, 'subnetworks_october', 'infomap', f'sub-{subject}')
output_base = os.path.join(working_dir, 'ppi_results_dmn_dan', f'sub-{subject}')
os.makedirs(output_base, exist_ok=True)

# Load FPN subnetwork masks (k=2, relabeled)
subnetwork_path = os.path.join(subnetwork_dir, f'{subject}_FPN_infomap_communities_kmeans_relabeled.dscalar.nii')
if not os.path.exists(subnetwork_path):
    print(f"Subnetwork file not found: {subnetwork_path}")
    sys.exit(1)

subnetwork_img = nib.load(subnetwork_path)
subnetwork_data = subnetwork_img.get_fdata()

# Use k=2 (row index 1)
label_vec = subnetwork_data[1, :].astype(int).squeeze() if subnetwork_data.ndim == 2 else subnetwork_data.squeeze().astype(int)
subnet1_mask = (label_vec == 1)
subnet2_mask = (label_vec == 2)

print(f"Subject: {subject}")
print(f"Subnetwork 1: {subnet1_mask.sum()} vertices")
print(f"Subnetwork 2: {subnet2_mask.sum()} vertices")
print(f"{'='*60}\n")

# Load the vertex-level network assignments (LSN parcellation LABELS to identify DMN and DAN vertices)
label_path = os.path.join(
    working_dir, 'individual_networks_october', f'sub-{subject}', 'resting_state',
    'Bipartite_PhysicalCommunities+AlgorithmicLabeling.dlabel.nii'
)

if not os.path.exists(label_path):
    print(f"Error: Network label file not found at {label_path}")
    sys.exit(1)

label_img = nib.load(label_path)
network_labels = label_img.get_fdata().squeeze().astype(int)

print(f"Network labels shape: {network_labels.shape}")
print(f"Unique network labels: {np.unique(network_labels)}")

# Get the label table from the CIFTI header to identify correct indices
label_table = label_img.header.get_axis(0).label[0]  # Get first map's labels
print("\nAvailable network labels:")
for key, (name, rgba) in label_table.items():
    print(f"  {key}: {name}")

# CHANGES MADE!!! I accidentially chose dan labels as FPN+DAN before (9,10)
# Adjust which networks are DMN and DAN based on the printed label table above:
dmn_labels = [1, 2, 3, 4]  # Default_Parietal, Default_Anterolateral, Default_Dorsolateral, Default_Retrosplenial
dan_labels = [10, 11]      # DorsalAttention, Premotor/DorsalAttentionII

dmn_mask = np.isin(network_labels, dmn_labels)
dan_mask = np.isin(network_labels, dan_labels)

# Add this diagnostic output after loading masks:
print(f"\nDiagnostic - Network vertex counts:")
print(f"  FPN (subnet1 + subnet2): {(subnet1_mask | subnet2_mask).sum()}")
print(f"  DMN (labels {dmn_labels}): {dmn_mask.sum()}. Expected about 8-12k vertices.")
print(f"  DAN (labels {dan_labels}): {dan_mask.sum()}. Expected about 3-5k vertices.")
print(f"  Overlap FPN-DMN: {((subnet1_mask | subnet2_mask) & dmn_mask).sum()}")
print(f"  Overlap FPN-DAN: {((subnet1_mask | subnet2_mask) & dan_mask).sum()}")

if dmn_mask.sum() == 0 or dan_mask.sum() == 0:
    print("Warning: DMN or DAN mask is empty. Check network label indices.")
    print("Adjust dmn_labels and dan_labels based on the label table printed above.")
    sys.exit(1)

# Track all runs for fixed-effects
all_ppi_results = []

# ========== 3) PROCESS FUNCTIONAL RUNS ==========
subject_dir = os.path.join(base_dir, f'sub-{subject}')
session_dirs = sorted(glob.glob(os.path.join(subject_dir, 'ses-*')))

for session_dir in session_dirs:
    session = os.path.basename(session_dir)
    glm_dir = os.path.join(session_dir, 'postfmriprep', 'GLM')
    if not os.path.exists(glm_dir):
        continue
    # This makes sure to only use task and not rest runs
    func_files = sorted(glob.glob(os.path.join(glm_dir, f'sub-{subject}_{session}_task-*_dir-*_*cleaned_noscrub.dtseries.nii')))
    
    for func_path in func_files:
        fname = os.path.basename(func_path)
        parts = fname.split('_')
        task = [p.split('-')[1] for p in parts if p.startswith('task-')][0]
        direction = [p.split('-')[1] for p in parts if p.startswith('dir-')][0]
        run = [p.split('-')[1] for p in parts if p.startswith('run-')]
        run = run[0] if run else None
        
        run_id = f"task-{task}_dir-{direction}_run-{run}" if run else f"task-{task}_dir-{direction}"
        
        # Load functional data
        func_img = nib.load(func_path)
        func_data = func_img.get_fdata()

        # Verify mask dimensions
        if func_data.shape[1] != len(label_vec):
            print(f"Skipping {run_id}: dimension mismatch")
            continue

        if subnet1_mask.sum() == 0 or subnet2_mask.sum() == 0:
            print("Skipping run: empty subnetwork mask")
            continue

        # ========== 3a) LOAD AND VALIDATE DATA ==========
        ax0 = func_img.header.get_axis(0)
        ax1 = func_img.header.get_axis(1)
        ts_axis = ax0 if isinstance(ax0, nib.cifti2.SeriesAxis) else ax1
        tr = float(getattr(ts_axis, "step", 2.0))
        n_scans = func_data.shape[0]
        frame_times = np.arange(n_scans) * tr

        # Load motion and events
        motion_fname = f'sub-{subject}_{session}_task-{task}_dir-{direction}_motion.txt' if not run else f'sub-{subject}_{session}_task-{task}_dir-{direction}_run-{run}_motion.txt'
        motion_path = os.path.join(session_dir, 'postfmriprep', 'regressors', motion_fname)
        onset_path = find_events_file(subject, session, task, direction=direction, run=run)

        if not os.path.exists(motion_path) or onset_path is None:
            print(f"Skipping {run_id}: missing files")
            continue

        motion = np.loadtxt(motion_path)
        if motion.ndim == 1:
            motion = motion.reshape(-1, 6) if motion.size % 6 == 0 else motion.reshape(-1, 1)
        
        # Adjust motion length
        diff = motion.shape[0] - n_scans
        if abs(diff) > 0 and abs(diff) <= 5:
            motion = motion[max(0, diff):min(motion.shape[0], motion.shape[0] - diff), :]
            if diff < 0:
                motion = np.pad(motion, ((abs(diff), 0), (0, 0)), mode='constant')
        elif abs(diff) > 5:
            print(f"Skipping {run_id}: motion mismatch")
            continue

        # ========== 3b) EXTRACT AND CLEAN SUBNETWORK TIMESERIES ==========
        ts_subnet1 = func_data[:, subnet1_mask].mean(axis=1)
        ts_subnet2 = func_data[:, subnet2_mask].mean(axis=1)

        if not np.isfinite(ts_subnet1).all() or not np.isfinite(ts_subnet2).all():
            print(f"Skipping {run_id}: NaNs in subnetwork time series")
            continue

        ts_subnet1_clean = clean(ts_subnet1.reshape(-1, 1), detrend=True, standardize='zscore_sample').ravel()
        ts_subnet2_clean = clean(ts_subnet2.reshape(-1, 1), detrend=True, standardize='zscore_sample').ravel()

        # Extract DMN and DAN timeseries from THIS RUN's functional data
        dmn_tseries_run = func_data[:, dmn_mask]  # shape: (n_scans, n_dmn_vertices)
        dan_tseries_run = func_data[:, dan_mask]  # shape: (n_scans, n_dan_vertices)
        
        # Average across vertices within each network
        dmn_ts_run = dmn_tseries_run.mean(axis=1)  # shape: (n_scans,)
        dan_ts_run = dan_tseries_run.mean(axis=1)  # shape: (n_scans,)
        
        # Clean DMN and DAN timeseries
        dmn_clean = clean(dmn_ts_run.reshape(-1, 1), detrend=True, standardize='zscore_sample').ravel()
        dan_clean = clean(dan_ts_run.reshape(-1, 1), detrend=True, standardize='zscore_sample').ravel()

        # ========== 3c) LOAD EVENTS AND SELECT CONDITIONS ==========
        events = pd.read_csv(onset_path, sep='\t')
        events['onset'] = pd.to_numeric(events['onset'], errors='coerce')
        events['duration'] = pd.to_numeric(events['duration'], errors='coerce')
        events = events.dropna(subset=['onset', 'duration']).reset_index(drop=True)

        counts = events['trial_type'].value_counts()
        min_events = 2
        selected_conditions = [c for c in counts.index if counts[c] >= min_events]

        print(f"Processing {run_id}")
        print(f"  n_scans={n_scans}, selected_conditions={len(selected_conditions)}/{len(counts)}")

        # ========== 3d) LOOP OVER SEEDS (FPN1, FPN2) AND TARGETS (DMN, DAN) ==========
        for seed_name, seed_ts in [('FPN1', ts_subnet1_clean), ('FPN2', ts_subnet2_clean)]:
            for target_name, target_ts in [('DMN', dmn_clean), ('DAN', dan_clean)]:
                
                # Build design matrix
                design_dict = {'physio': seed_ts}
                for i in range(motion.shape[1]):
                    design_dict[f'motion_{i}'] = motion[:, i]

                # Create psychological and PPI regressors
                for condition in selected_conditions:
                    condition_mask = events['trial_type'] == condition
                    onsets = events.loc[condition_mask, 'onset'].to_numpy(dtype=np.float64)
                    durations = events.loc[condition_mask, 'duration'].to_numpy(dtype=np.float64)
                    amplitudes = np.ones(len(onsets), dtype=np.float64)

                    if len(onsets) == 0:
                        continue

                    exp_condition = (onsets, durations, amplitudes)
                    psych_regressor, _ = compute_regressor(
                        exp_condition,
                        hrf_model='spm',
                        frame_times=frame_times
                    )
                    psych_regressor = psych_regressor[:, 0]
                    psych_regressor = psych_regressor - psych_regressor.mean()

                    if np.allclose(psych_regressor.std(), 0, atol=1e-8):
                        continue

                    ppi_regressor = seed_ts * psych_regressor
                    ppi_amplitude = np.max(np.abs(ppi_regressor))
                    if ppi_amplitude < 0.01:
                        continue

                    design_dict[f'psych_{condition}'] = psych_regressor
                    design_dict[f'ppi_{condition}'] = ppi_regressor

                design_matrix = pd.DataFrame(design_dict)

                # Drop low-variance columns
                stds = design_matrix.std(axis=0, ddof=0)
                keep_cols = stds[stds > 1e-8].index.tolist()
                design_matrix = design_matrix[keep_cols]
                design_matrix['constant'] = 1.0

                # ========== 3e) FIT GLM ==========
                X = design_matrix.values.astype(np.float64)
                y = target_ts.astype(np.float64).reshape(-1)

                rank = np.linalg.matrix_rank(X)
                df = X.shape[0] - rank
                
                if df <= 0:
                    print(f"  Warning: non-positive df for {seed_name}->{target_name}")
                    continue

                labels, results = run_glm(
                    y.reshape(-1, 1),
                    X,
                    noise_model='ar1',
                    bins=100
                )
                glm_result = results[labels[0]]

                theta = np.asarray(glm_result.theta, dtype=np.float64).reshape(-1)
                fitted = X @ theta
                residuals = y - fitted

                sigma2 = float(residuals @ residuals) / df
                XtX_pinv = np.linalg.pinv(X.T @ X)

                col_names = design_matrix.columns.tolist()

                # Compute PPI contrast for each condition
                for condition in selected_conditions:
                    ppi_col = f'ppi_{condition}'
                    if ppi_col not in col_names:
                        continue
                    ppi_idx = col_names.index(ppi_col)

                    beta = float(theta[ppi_idx])
                    variance = float(sigma2 * XtX_pinv[ppi_idx, ppi_idx])
                    se = np.sqrt(variance) if variance > 0 else np.inf
                    tstat = beta / se if np.isfinite(se) and se > 0 else 0.0
                    pval = 2 * (1 - stats.t.cdf(abs(tstat), df)) if np.isfinite(tstat) else 1.0

                    result_entry = {
                        'task': task,
                        'condition': condition,
                        'run_id': run_id,
                        'session': session,
                        'seed': seed_name,
                        'target': target_name,
                        'beta': beta,
                        'variance': variance,
                        'tstat': float(tstat),
                        'pval': float(pval)
                    }
                    all_ppi_results.append(result_entry)

                    output_dir = os.path.join(output_base, session, f'res_{run_id}')
                    os.makedirs(output_dir, exist_ok=True)
                    np.savez(os.path.join(output_dir, f'{seed_name}_to_{target_name}_{condition}_ppi.npz'), **result_entry)
                    print(f"    {seed_name}->{target_name} {condition}: β={beta:.4f}, t={tstat:.4f}, p={pval:.4e}")

def generate_subject_summary_report(results_df, ffx_df, subject, output_base):
    """Generate subject-specific interpretable summary report."""
    report = []
    report.append("="*80)
    report.append(f"SUBJECT {subject.upper()}: FPN→DMN/DAN PPI CONNECTIVITY ANALYSIS")
    report.append("="*80)
    report.append("")
    
    n_runs = results_df['run_id'].nunique()
    n_tasks = results_df['task'].nunique()
    n_conditions = results_df['condition'].nunique()
    
    report.append(f"Total Runs: {n_runs}")
    report.append(f"Total Tasks: {n_tasks}")
    report.append(f"Total Task Conditions: {n_conditions}")
    report.append(f"Total Run×Condition×Seed×Target Combinations: {len(results_df)}")
    report.append("")
    
    # Summary by seed→target pair
    for pair in sorted(results_df['seed'].unique() + results_df['target'].unique()):
        seeds = sorted(results_df['seed'].unique())
        targets = sorted(results_df['target'].unique())
        
    for seed in seeds:
        for target in targets:
            pair_label = f"{seed}→{target}".replace('FPN1', 'FPNA').replace('FPN2', 'FPNB')
            report.append("-" * 80)
            report.append(f"{pair_label} CONNECTIVITY")
            report.append("-" * 80)
            
            pair_data = results_df[(results_df['seed'] == seed) & (results_df['target'] == target)]
            
            if len(pair_data) == 0:
                report.append("  No data for this seed-target pair.")
                report.append("")
                continue
            
            # Overall statistics across all runs and conditions
            mean_beta = pair_data['beta'].mean()
            std_beta = pair_data['beta'].std()
            median_beta = pair_data['beta'].median()
            
            report.append(f"\nRun-Level Statistics:")
            report.append(f"  Mean β: {mean_beta:.4f} ± {std_beta:.4f}")
            report.append(f"  Median β: {median_beta:.4f}")
            report.append(f"  Number of run×condition combinations: {len(pair_data)}")
            
            # Significant effects (p < 0.05)
            sig_effects = pair_data[pair_data['pval'] < 0.05]
            report.append(f"  Significant effects (p<0.05): {len(sig_effects)} ({100*len(sig_effects)/len(pair_data):.1f}%)")
            
            # Fixed-effects summary (if available)
            if ffx_df is not None and len(ffx_df) > 0:
                ffx_pair = ffx_df[(ffx_df['seed'] == seed) & (ffx_df['target'] == target)]
                if len(ffx_pair) > 0:
                    report.append(f"\nFixed-Effects Summary (by Task):")
                    for _, row in ffx_pair.iterrows():
                        report.append(f"  {row['task']}: β={row['beta_ffx']:.4f}, SE={row['se_ffx']:.4f}, t={row['t_ffx']:.2f} (n={row['n_runs']} runs)")
            
            # Top positive effects
            report.append(f"\nTop 5 Conditions with INCREASED Connectivity:")
            top_pos = pair_data[pair_data['beta'] > 0].nlargest(5, 'beta')
            if len(top_pos) > 0:
                for idx, row in top_pos.iterrows():
                    report.append(f"  • {row['task']}_{row['condition']} [{row['run_id']}]: β={row['beta']:.4f}, t={row['tstat']:.2f}, p={row['pval']:.4f}")
            else:
                report.append("  No positive effects found.")
            
            # Top negative effects
            report.append(f"\nTop 5 Conditions with DECREASED Connectivity:")
            top_neg = pair_data[pair_data['beta'] < 0].nsmallest(5, 'beta')
            if len(top_neg) > 0:
                for idx, row in top_neg.iterrows():
                    report.append(f"  • {row['task']}_{row['condition']} [{row['run_id']}]: β={row['beta']:.4f}, t={row['tstat']:.2f}, p={row['pval']:.4f}")
            else:
                report.append("  No negative effects found.")
            
            report.append("")
    
    # Cross-seed comparison
    report.append("-" * 80)
    report.append("COMPARING FPN1 vs FPN2 (FPNA vs FPNB)")
    report.append("-" * 80)
    
    for target in targets:
        report.append(f"\n{target}:")
        fpn1_effects = results_df[(results_df['seed'] == 'FPN1') & (results_df['target'] == target)]['beta']
        fpn2_effects = results_df[(results_df['seed'] == 'FPN2') & (results_df['target'] == target)]['beta']
        
        if len(fpn1_effects) > 0 and len(fpn2_effects) > 0:
            report.append(f"  FPN1→{target}: mean={fpn1_effects.mean():.4f}, std={fpn1_effects.std():.4f}, n={len(fpn1_effects)}")
            report.append(f"  FPN2→{target}: mean={fpn2_effects.mean():.4f}, std={fpn2_effects.std():.4f}, n={len(fpn2_effects)}")
            
            # Check if we have paired data for comparison
            fpn1_data = results_df[(results_df['seed'] == 'FPN1') & (results_df['target'] == target)].set_index(['task', 'condition', 'run_id'])
            fpn2_data = results_df[(results_df['seed'] == 'FPN2') & (results_df['target'] == target)].set_index(['task', 'condition', 'run_id'])
            
            matched_indices = fpn1_data.index.intersection(fpn2_data.index)
            if len(matched_indices) > 1:
                fpn1_matched = fpn1_data.loc[matched_indices, 'beta'].values
                fpn2_matched = fpn2_data.loc[matched_indices, 'beta'].values
                
                from scipy.stats import ttest_rel
                t_stat, p_val = ttest_rel(fpn1_matched, fpn2_matched)
                report.append(f"  Paired t-test (n={len(matched_indices)} matched conditions): t={t_stat:.3f}, p={p_val:.4f}")
    
    report.append("\n" + "="*80)
    
    report_text = "\n".join(report)
    summary_file = os.path.join(output_base, 'ppi_dmn_dan_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(report_text)
    
    print(f"✓ Subject summary saved to: {summary_file}")
    return summary_file

# ========== 4) SAVE RESULTS ==========
print(f"\n{'='*60}")
print("Saving results...")
print(f"{'='*60}")

if all_ppi_results:
    results_df = pd.DataFrame(all_ppi_results)
    
    # Save per-run results
    subject_csv = os.path.join(output_base, 'ppi_dmn_dan_results.csv')
    results_df.to_csv(subject_csv, index=False)
    print(f"✓ Per-run results saved to: {subject_csv}")
    
    # Compute and save fixed-effects by task, seed, and target
    ffx_results = []
    for (task, seed, target), group in results_df.groupby(['task', 'seed', 'target']):
        if len(group) >= 2:
            weights = 1.0 / group['variance'].values
            beta_ffx = np.sum(group['beta'].values * weights) / np.sum(weights)
            var_ffx = 1.0 / np.sum(weights)
            t_ffx = beta_ffx / np.sqrt(var_ffx)
            ffx_results.append({
                'subject': subject,
                'task': task,
                'seed': seed,
                'target': target,
                'n_runs': len(group),
                'beta_ffx': beta_ffx,
                'se_ffx': np.sqrt(var_ffx),
                't_ffx': t_ffx
            })

    ffx_df = pd.DataFrame(ffx_results) if ffx_results else None
    if ffx_df is not None:
        ffx_csv = os.path.join(output_base, 'ppi_dmn_dan_ffx_summary.csv')
        ffx_df.to_csv(ffx_csv, index=False)
        print(f"✓ Fixed-effects summary saved to: {ffx_csv}")
    
    # Generate subject-specific summary report
    generate_subject_summary_report(results_df, ffx_df, subject, output_base)

print("\n✓ DMN/DAN PPI analysis complete!")