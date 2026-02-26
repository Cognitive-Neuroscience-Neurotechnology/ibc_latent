"""
Task-level PPI analysis: Examine how subnetwork connectivity differs between tasks.
Each task gets ONE PPI value (all conditions pooled).
I don't want to use this I believe. Let's rather use ppi_analysis_gPPI.py
OLD!! - Now using _DMN_DAN
"""

import sys
import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel, compute_regressor
from nilearn.signal import clean

subject = sys.argv[1]
base_dir = '/ptmp/hmueller2/2025_ibc_latent/outputs/preprocessing/fmriprep_out'
subnetwork_dir = f'/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_derivation/infomap/sub-{subject}'
output_base = f'/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/old_versions/ppi_results/sub-{subject}'
os.makedirs(output_base, exist_ok=True)

# Load FPN subnetwork masks
subnetwork_path = os.path.join(subnetwork_dir, f'{subject}_FPN_infomap_communities_kmeans_relabeled.dscalar.nii')
if not os.path.exists(subnetwork_path):
    print(f"Subnetwork file not found: {subnetwork_path}")
    sys.exit(1)

subnetwork_img = nib.load(subnetwork_path)
subnetwork_data = subnetwork_img.get_fdata()

subnet1_mask = (subnetwork_data[1] == 1)  # Changed from [0] to [1]
subnet2_mask = (subnetwork_data[1] == 2)  # Changed from [0] to [1]


print(f"Subject: {subject}")
print(f"Subnetwork 1: {subnet1_mask.sum()} vertices")
print(f"Subnetwork 2: {subnet2_mask.sum()} vertices")
print(f"{'='*60}\n")

# Track all runs for fixed-effects
all_ppi_results = []

# ========== Process functional runs ==========
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
        
        # Load functional data
        func_img = nib.load(func_path)
        func_data = func_img.get_fdata()
        
        # Extract TR
        ax0 = func_img.header.get_axis(0)
        ax1 = func_img.header.get_axis(1)
        ts_axis = ax0 if isinstance(ax0, nib.cifti2.SeriesAxis) else ax1
        tr = float(getattr(ts_axis, "step", 2.0))
        n_scans = func_data.shape[0]
        frame_times = np.arange(n_scans) * tr
        
        # Load motion and events
        if run:
            motion_fname = f'sub-{subject}_{session}_task-{task}_dir-{direction}_run-{run}_motion.txt'
            onset_fname = f'sub-{subject}_{session}_task-{task}_dir-{direction}_run-{run}_events.tsv'
            run_id = f'task-{task}_run-{run}_dir-{direction}'
        else:
            motion_fname = f'sub-{subject}_{session}_task-{task}_dir-{direction}_motion.txt'
            onset_fname = f'sub-{subject}_{session}_task-{task}_dir-{direction}_events.tsv'
            run_id = f'task-{task}_dir-{direction}'
        
        motion_path = os.path.join(session_dir, 'postfmriprep', 'regressors', motion_fname)
        onset_path = os.path.join('/ptmp/hmueller2/2025_ibc_latent/data/ibc_raw', f'sub-{subject}', session, 'func', onset_fname)
        
        if not os.path.exists(motion_path) or not os.path.exists(onset_path):
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
        
        # ========== Extract and clean subnetwork timeseries ==========
        ts_subnet1 = func_data[:, subnet1_mask].mean(axis=1)
        ts_subnet2 = func_data[:, subnet2_mask].mean(axis=1)
        
        ts_subnet1_clean = clean(ts_subnet1.reshape(-1, 1), detrend=True, standardize=True).ravel()
        ts_subnet2_clean = clean(ts_subnet2.reshape(-1, 1), detrend=True, standardize=True).ravel()
        
        # ========== Task-level PPI: Pool ALL conditions ==========
        print(f"Processing {run_id} (task-level PPI)")
        
        events = pd.read_csv(onset_path, sep='\t')
        
        # Combine all conditions into one task regressor
        all_events = events[['onset', 'duration']].copy()
        all_events['trial_type'] = task  # Use task name as condition
        
        # Create HRF-convolved psychological regressor for entire task
        psych_regressor, _ = compute_regressor(
            all_events,
            hrf_model='spm',
            frame_times=frame_times
        )
        psych_regressor = psych_regressor[:, 0]
        
        # Demean psychological regressor
        psych_regressor = psych_regressor - psych_regressor.mean()
        
        # Create PPI regressor: interaction
        ppi_regressor = ts_subnet1_clean * psych_regressor
        
        # Build design matrix
        design_dict = {
            'ppi': ppi_regressor,
            'physio': ts_subnet1_clean,
            'psych': psych_regressor,
        }
        for i in range(motion.shape[1]):
            design_dict[f'motion_{i}'] = motion[:, i]
        
        design_matrix = pd.DataFrame(design_dict)
        
        # Run GLM on subnet2 timeseries
        y = ts_subnet2_clean.reshape(-1, 1, 1, 1)
        
        glm = FirstLevelModel(t_r=tr, drift_model='cosine', high_pass=0.01)
        glm.fit(y, design_matrices=design_matrix)
        
        # Compute PPI contrast
        ppi_contrast = glm.compute_contrast('ppi', output_type='all')
        
        # Store results for fixed-effects
        result_entry = {
            'task': task,
            'run_id': run_id,
            'session': session,
            'beta': ppi_contrast.effect_size()[0, 0, 0],
            'variance': ppi_contrast.variance()[0, 0, 0],
            'tstat': ppi_contrast.stat()[0, 0, 0],
            'pval': ppi_contrast.p_value()[0, 0, 0]
        }
        all_ppi_results.append(result_entry)
        
        # Save individual run result
        output_dir = os.path.join(output_base, session, f'res_{run_id}')
        os.makedirs(output_dir, exist_ok=True)
        np.savez(
            os.path.join(output_dir, f'{task}_task_level_ppi.npz'),
            **result_entry
        )
        print(f"  {task}: β={result_entry['beta']:.4f}, t={result_entry['tstat']:.4f}, p={result_entry['pval']:.4e}")

# ========== Fixed-effects across runs ==========
print(f"\n{'='*60}")
print("Computing fixed-effects PPI across runs (per task)...")
print(f"{'='*60}")

if all_ppi_results:
    results_df = pd.DataFrame(all_ppi_results)
    
    # Save per-subject results
    subject_csv = os.path.join(output_base, 'task_level_ppi_results.csv')
    results_df.to_csv(subject_csv, index=False)
    print(f"✓ Per-run results saved to: {subject_csv}")
    
    # Group by task only
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
    
    # Create summary CSV with fixed-effects
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
    
    if ffx_results:
        ffx_df = pd.DataFrame(ffx_results)
        ffx_csv = os.path.join(output_base, 'task_level_ppi_ffx_summary.csv')
        ffx_df.to_csv(ffx_csv, index=False)
        print(f"\n✓ Fixed-effects summary saved to: {ffx_csv}")

print("\n✓ Task-level PPI analysis complete!")