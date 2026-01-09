"""
PPI analysis to examine task-dependent connectivity between FPN subnetworks.
gPPI Version: Single GLM per run instead of one per condition
OLD!! - Now using _DMN_DAN
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

# Helper: robust events file lookup (try with/without dir and fall back to glob)
def find_events_file(subject, session, task, direction=None, run=None):
    base = os.path.join('/ptmp/hmueller2/Downloads/ibc_raw', f'sub-{subject}', session, 'func')

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

# Helper: infer masks that match functional columns
def infer_subnet_masks_for_ncols(subnetwork_data, func_ncols):
    """
    Always use k=2 clustering (row index 1) and labels 1 and 2.
    """
    label_vec = None
    # If shape is (k, n_vertices), use row 1 for k=2
    if subnetwork_data.ndim == 2:
        # Use row 1 for k=2 clustering
        if subnetwork_data.shape[0] > 1 and subnetwork_data.shape[1] == func_ncols:
            label_vec = subnetwork_data[1, :]
        elif subnetwork_data.shape[1] > 1 and subnetwork_data.shape[0] == func_ncols:
            label_vec = subnetwork_data[:, 1]
    if label_vec is None:
        # Fallback: try to squeeze and use as vector
        vec = np.asarray(subnetwork_data).squeeze()
        if vec.ndim == 1 and vec.shape[0] == func_ncols:
            label_vec = vec
    if label_vec is None:
        return None, None

    # Ensure labels are 1 and 2
    label_vec = np.asarray(label_vec)
    subnet1_mask = (label_vec == 1)
    subnet2_mask = (label_vec == 2)
    return subnet1_mask, subnet2_mask

# ========== 2) SETUP: LOAD SUBJECT DATA AND PATHS ==========
subject = sys.argv[1]
base_dir = '/ptmp/hmueller2/Downloads/fmriprep_out'
subnetwork_dir = f'/ptmp/hmueller2/Downloads/subnetworks/infomap/sub-{subject}'
output_base = f'/ptmp/hmueller2/Downloads/ppi_results/sub-{subject}'
os.makedirs(output_base, exist_ok=True)

# 1) Load FPN subnetwork masks
subnetwork_path = os.path.join(subnetwork_dir, f'{subject}_FPN_infomap_communities_kmeans_relabeled.dscalar.nii')
if not os.path.exists(subnetwork_path):
    print(f"Subnetwork file not found: {subnetwork_path}")
    sys.exit(1)

subnetwork_img = nib.load(subnetwork_path)
subnetwork_data = subnetwork_img.get_fdata()

# Delay mask creation until we know functional column count
subnet1_mask = None
subnet2_mask = None

print(f"Subject: {subject}")
print(f"{'='*60}\n")

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

    func_files = sorted(glob.glob(os.path.join(glm_dir, f'sub-{subject}_{session}_task-*_dir-*_*cleaned_noscrub.dtseries.nii')))
    
    for func_path in func_files:
        fname = os.path.basename(func_path)
        parts = fname.split('_')
        task = [p.split('-')[1] for p in parts if p.startswith('task-')][0]
        direction = [p.split('-')[1] for p in parts if p.startswith('dir-')][0]
        run = [p.split('-')[1] for p in parts if p.startswith('run-')]
        run = run[0] if run else None
        
        # Construct run_id for logging and tracking
        run_id = f"task-{task}_dir-{direction}_run-{run}" if run else f"task-{task}_dir-{direction}"
        
        # Load functional data
        func_img = nib.load(func_path)
        func_data = func_img.get_fdata()

        # Infer masks once using the first run's column count (robust to indexing)
        if subnet1_mask is None or subnet2_mask is None:
            s1, s2 = infer_subnet_masks_for_ncols(subnetwork_data, func_data.shape[1])
            if s1 is None or s2 is None:
                print(f"Skipping all runs: subnetwork map/func column mismatch (mask dims {subnetwork_data.shape} vs func cols {func_data.shape[1]})")
                break
            subnet1_mask, subnet2_mask = s1, s2
            print(f"Subnetwork 1: {subnet1_mask.sum()} vertices")
            print(f"Subnetwork 2: {subnet2_mask.sum()} vertices")
            print(f"{'='*60}\n")
        # Guard against empty masks
        if subnet1_mask.sum() == 0 or subnet2_mask.sum() == 0:
            print("Skipping run: one of the subnetworks has 0 vertices in the mask.")
            continue

        # ========== 3a) LOAD AND VALIDATE DATA ==========
        # Extract TR
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
            print(f"Skipping {run_id}: missing files (motion: {os.path.exists(motion_path)}, events: {onset_path is not None})")
            continue

        # Load motion parameters and fix columns
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

        # Skip if NaNs (e.g., empty selection)
        if not np.isfinite(ts_subnet1).all() or not np.isfinite(ts_subnet2).all():
            print(f"Skipping {run_id}: NaNs in subnetwork time series (check masks).")
            continue

        ts_subnet1_clean = clean(ts_subnet1.reshape(-1, 1), detrend=True, standardize='zscore_sample').ravel()
        ts_subnet2_clean = clean(ts_subnet2.reshape(-1, 1), detrend=True, standardize='zscore_sample').ravel()

        print(f"  ts_subnet1_clean: min={ts_subnet1_clean.min():.4f}, max={ts_subnet1_clean.max():.4f}, std={ts_subnet1_clean.std():.4f}")
        print(f"  ts_subnet2_clean: min={ts_subnet2_clean.min():.4f}, max={ts_subnet2_clean.max():.4f}, std={ts_subnet2_clean.std():.4f}")

        # ========== 3c) LOAD EVENTS AND SELECT CONDITIONS ==========
        events = pd.read_csv(onset_path, sep='\t')

        # Convert onset and duration to numeric FIRST (handle 'n/a' or other non-numeric values)
        events['onset'] = pd.to_numeric(events['onset'], errors='coerce')
        events['duration'] = pd.to_numeric(events['duration'], errors='coerce')

        # Drop rows with missing onset/duration
        events = events.dropna(subset=['onset', 'duration'])

        # Reset index after dropping rows
        events = events.reset_index(drop=True)

        # Select conditions: require >=2 events and cap to keep df positive
        counts = events['trial_type'].value_counts()
        min_events = 2 
        candidate_conds = [c for c in counts.index if counts[c] >= min_events]

        # Use all candidate conditions as contrasts instead
        selected_conditions = candidate_conds

        print(f"Processing {run_id} with gPPI")
        print(f"  n_scans={n_scans}, motion_cols={motion.shape[1]}, selected_conditions={len(selected_conditions)}/{len(counts)}")

        # ========== 3d) BUILD DESIGN MATRIX ==========
        design_dict = {
            'physio': ts_subnet1_clean,  # Main effect of seed region
        }
        # Add motion confounds
        for i in range(motion.shape[1]):
            design_dict[f'motion_{i}'] = motion[:, i]

        # Create psychological and PPI regressors for selected conditions
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

            # Only keep non-negligible regressors
            if np.allclose(psych_regressor.std(), 0, atol=1e-8):
                continue

            # **NEW: Skip PPI if amplitude is too small (causes collinearity)**
            ppi_regressor = ts_subnet1_clean * psych_regressor
            ppi_amplitude = np.max(np.abs(ppi_regressor))
            if ppi_amplitude < 0.01:  # threshold: skip very small PPIs
                print(f"  Skipping {condition}: PPI amplitude too small ({ppi_amplitude:.4e})")
                continue

            design_dict[f'psych_{condition}'] = psych_regressor
            design_dict[f'ppi_{condition}'] = ppi_regressor

            print(f"  psych_regressor for {condition}: min={psych_regressor.min():.4f}, max={psych_regressor.max():.4f}, ppi_amp={ppi_amplitude:.4e}")

        design_matrix = pd.DataFrame(design_dict)

        # Drop near-zero variance columns to prevent rank deficiency
        # let's see if i need this
        stds = design_matrix.std(axis=0, ddof=0)
        keep_cols = stds[stds > 1e-8].index.tolist()
        dropped_cols = [c for c in design_matrix.columns if c not in keep_cols]
        if dropped_cols:
            print(f"  Dropping low-variance columns: {dropped_cols}")
        design_matrix = design_matrix[keep_cols]

        # Add constant term
        design_matrix['constant'] = 1.0

        # ========== 3e) FIT GLM AND COMPUTE PPI CONTRASTS ==========
        # **NEW: Check collinearity BEFORE fitting**
        X = design_matrix.values.astype(np.float64)
        cond_num = np.linalg.cond(X)
        print(f"  Design matrix condition number: {cond_num:.2e}")
        if cond_num > 1e10:
            print(f"  WARNING: Severe collinearity detected (cond={cond_num:.2e}). Consider removing near-constant regressors.")
        
        y = ts_subnet2_clean.astype(np.float64).reshape(-1)

        # Fit GLM
        rank = np.linalg.matrix_rank(X)
        df = X.shape[0] - rank
        print(f"  Design: n={X.shape[0]}, p={X.shape[1]}, rank={rank}, df={df}")
        print(f"  Design matrix condition number: {np.linalg.cond(X):.2e}")
        if df <= 0:
            print(f"  Warning: non-positive degrees of freedom (df={df}). Skipping run {run_id}.")
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
        if rank < X.shape[1]:
            print(f"  Warning: design matrix rank deficient (rank={rank} < p={X.shape[1]}). Using pseudoinverse.")

        col_names = design_matrix.columns.tolist()

        from scipy import stats
        # Compute PPI contrast for EACH selected condition
        for condition in selected_conditions:
            ppi_col = f'ppi_{condition}'
            if ppi_col not in col_names:
                continue
            ppi_idx = col_names.index(ppi_col)
            ppi_vec = X[:, ppi_idx]
            if np.allclose(ppi_vec, 0, atol=1e-8):
                print(f"    Skipping {condition}: PPI regressor is (near) zero.")
                continue

            beta = float(np.asarray(theta[ppi_idx]).squeeze())
            variance = float(sigma2 * XtX_pinv[ppi_idx, ppi_idx])
            se = np.sqrt(variance) if variance > 0 else np.inf
            tstat = beta / se if np.isfinite(se) and se > 0 else 0.0
            pval = 2 * (1 - stats.t.cdf(abs(tstat), df)) if np.isfinite(tstat) else 1.0

            result_entry = {
                'task': task,
                'condition': condition,
                'run_id': run_id,
                'session': session,
                'beta': beta,
                'variance': variance,
                'tstat': float(tstat),
                'pval': float(pval)
            }
            all_ppi_results.append(result_entry)

            output_dir = os.path.join(output_base, session, f'res_{run_id}')
            os.makedirs(output_dir, exist_ok=True)
            np.savez(os.path.join(output_dir, f'{condition}_ppi.npz'), **result_entry)
            print(f"    {condition}: β={beta:.4f}, t={tstat:.4f}, p={pval:.4e}")

# ========== 4) FIXED-EFFECTS ACROSS RUNS ==========
print(f"\n{'='*60}")
print("Computing fixed-effects PPI across runs (per task)...")
print(f"{'='*60}")

if all_ppi_results:
    results_df = pd.DataFrame(all_ppi_results)
    
    # ========== 4a) SAVE PER-RUN RESULTS ==========
    subject_csv = os.path.join(output_base, 'task_level_ppi_results.csv')
    results_df.to_csv(subject_csv, index=False)
    print(f"✓ Per-run results saved to: {subject_csv}")
    
    # ========== 4b) COMPUTE AND SAVE FIXED-EFFECTS BY TASK ==========
    for task, group in results_df.groupby('task'):
        if len(group) < 2:
            print(f"Skipping {task}: only 1 run")
            continue
        
        # Fixed-effects: inverse-variance weighted average
        weights = 1.0 / group['variance'].values
        beta_ffx = np.sum(group['beta'].values * weights) / np.sum(weights)
        var_ffx = 1.0 / np.sum(weights)
        t_ffx = beta_ffx / np.sqrt(var_ffx)
        
        # Save fixed-effects result
        ffx_dir = os.path.join(output_base, f'res_task-{task}_dir-ffx')
        os.makedirs(ffx_dir, exist_ok=True)
        
        ffx_result = {
            'task': task,
            'n_runs': len(group),
            'beta_ffx': beta_ffx,
            'var_ffx': var_ffx,
            't_ffx': t_ffx
        }
        
        np.savez(os.path.join(ffx_dir, f'{task}_ppi_ffx.npz'), **ffx_result)
        print(f"✓ {task}: β={beta_ffx:.4f}, t={t_ffx:.4f} (n={len(group)} runs)")
    
    # ========== 4c) CREATE SUMMARY CSV WITH FIXED-EFFECTS ==========
    ffx_results = []
    for task, group in results_df.groupby('task'):
        if len(group) >= 2:
            weights = 1.0 / group['variance'].values
            beta_ffx = np.sum(group['beta'].values * weights) / np.sum(weights)
            var_ffx = 1.0 / np.sum(weights)
            t_ffx = beta_ffx / np.sqrt(var_ffx)
            ffx_results.append({
                'subject': subject,
                'task': task,
                'n_runs': len(group),
                'beta_ffx': beta_ffx,
                'se_ffx': np.sqrt(var_ffx),
                't_ffx': t_ffx
            })

    # Always write the summary file (may be empty)
    ffx_df = pd.DataFrame(ffx_results)
    ffx_csv = os.path.join(output_base, 'task_level_ppi_ffx_summary.csv')
    ffx_df.to_csv(ffx_csv, index=False)
    if ffx_df.empty:
        print("No tasks had >=2 valid runs; empty FFX summary written.")
    else:
        print(f"\n✓ Fixed-effects summary saved to: {ffx_csv}")

# ========== 5) COMPLETION ==========
print("\n✓ Task-level PPI analysis complete!")