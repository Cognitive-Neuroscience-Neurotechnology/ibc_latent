"""
Mapping the MD system in every individual by using task contrasts with difficulty manipulation.

Contrasts to use:
    Option 1 (mine): 2back-0back (HcpWm), incongruent-congruent (Stroop), hard-easy (Catell), double_incongruent-double_congruent (Attention)
    Option 2 (Assem): 2back-0back (HcpWm), relational-match (HcpRelational), math-story (HcpLanguage)
"""

import os
import sys
import glob
import numpy as np
import nibabel as nib
from collections import defaultdict
import argparse
import subprocess
import tempfile
import shutil

# ============================================================================
# MD-RELATED CONTRASTS DEFINITION
# ============================================================================

MD_CONTRASTS = {
    # ASSEM 2020 contrasts
    'HcpWm': ['2back-0back'],
    'HcpRelationalR': ['relational-match'],
    #'HcpLanguage': ['math-story'],
    #'Stroop': ['incongruent-congruent'],
    'Catell': ['hard-easy'],
    'Attention': ['double_incongruent-double_congruent'],
    #'ItemRecognition': ['probe5_mem-probe1_mem'],
    #'MVEB': ['6_letters_different-2_letters_different'],
    #'MVIS': ['6_dots-2_dots'],
    #'VisualSearch': ['probe_item_four-probe_item_two'],
}

# Flatten the dictionary into a list of tuples (task, contrast)
MD_CONTRAST_LIST = []
for task, contrasts in MD_CONTRASTS.items():
    for contrast in contrasts:
        MD_CONTRAST_LIST.append((task, contrast))

DEFAULT_WB_COMMAND = '/home/hmueller2/workbench/bin_linux64/wb_command'
DEFAULT_LEFT_SURFACE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        'MSCcodebase',
        'Utilities',
        'Conte69_atlas-v2.LR.32k_fs_LR.wb',
        'Conte69.L.midthickness.32k_fs_LR.surf.gii',
    )
)
DEFAULT_RIGHT_SURFACE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        'MSCcodebase',
        'Utilities',
        'Conte69_atlas-v2.LR.32k_fs_LR.wb',
        'Conte69.R.midthickness.32k_fs_LR.surf.gii',
    )
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_contrast_map(file_path):
    """Load a CIFTI contrast map and return the data array."""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        # Data shape is typically (1, n_vertices) for scalar maps
        if data.shape[0] == 1:
            data = data[0]  # Extract the single row
        return data, img
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return None, None


def smooth_cifti_with_workbench(input_cifti, output_cifti, smoothing_fwhm,
                                wb_command=DEFAULT_WB_COMMAND,
                                left_surface=DEFAULT_LEFT_SURFACE,
                                right_surface=DEFAULT_RIGHT_SURFACE):
    """Apply geodesic smoothing with Connectome Workbench to a CIFTI file."""
    if smoothing_fwhm <= 0:
        return input_cifti

    # Resolve wb_command robustly across host/container environments.
    resolved_wb_command = None
    wb_candidates = []
    if wb_command:
        wb_candidates.append(wb_command)
    if wb_command and os.path.basename(wb_command) == wb_command:
        which_match = shutil.which(wb_command)
        if which_match:
            wb_candidates.append(which_match)
    else:
        which_match = shutil.which('wb_command')
        if which_match:
            wb_candidates.append(which_match)
    wb_candidates.extend([
        '/usr/bin/wb_command',
        '/usr/local/bin/wb_command',
        '/opt/workbench/bin_linux64/wb_command',
    ])

    for candidate in wb_candidates:
        if candidate and os.path.exists(candidate):
            resolved_wb_command = candidate
            break

    if resolved_wb_command is None:
        raise FileNotFoundError(
            "Could not find wb_command. Tried configured path and common locations. "
            "Pass a valid path with --wb-command or ensure wb_command is in PATH."
        )

    missing = []
    for path in [resolved_wb_command, left_surface, right_surface, input_cifti]:
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        raise FileNotFoundError(f"Missing required file(s) for Workbench smoothing: {missing}")

    cmd = [
        resolved_wb_command,
        '-cifti-smoothing',
        input_cifti,
        str(smoothing_fwhm),
        '0',
        'COLUMN',
        output_cifti,
        '-left-surface',
        left_surface,
        '-right-surface',
        right_surface,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Workbench smoothing failed. "
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return output_cifti


def threshold_map(map_data, threshold, label='map'):
    """Threshold a 1D map and print diagnostics about retained vertices."""
    data_min = float(np.min(map_data))
    data_max = float(np.max(map_data))
    print(f"\n{label} range before threshold: min={data_min:.3f}, max={data_max:.3f}")

    thresholded = map_data.copy()
    thresholded[thresholded < threshold] = 0
    n_suprathreshold = int(np.sum(thresholded > 0))
    pct = 100 * n_suprathreshold / len(map_data)

    print(f"Threshold: z > {threshold}")
    print(f"Suprathreshold vertices: {n_suprathreshold} ({pct:.1f}%)")

    if n_suprathreshold == 0:
        print(
            f"WARNING: No vertices survive z > {threshold}. "
            "This usually means the threshold is too high for an averaged MD map."
        )

    return thresholded, n_suprathreshold


def threshold_map_top_percent(map_data, threshold_percent, label='map'):
    """Keep only the top X percent of vertices by value and zero-out the rest."""
    if threshold_percent <= 0 or threshold_percent > 100:
        raise ValueError(f"threshold_percent must be in (0, 100], got {threshold_percent}")

    data_min = float(np.min(map_data))
    data_max = float(np.max(map_data))
    percentile = 100.0 - float(threshold_percent)
    cutoff = float(np.percentile(map_data, percentile))

    print(f"\n{label} range before threshold: min={data_min:.3f}, max={data_max:.3f}")
    print(f"Threshold mode: top {threshold_percent}% of vertices")
    print(f"Percentile cutoff ({percentile:.1f}th): {cutoff:.3f}")

    thresholded = np.zeros_like(map_data)
    keep_mask = map_data >= cutoff
    thresholded[keep_mask] = map_data[keep_mask]

    n_kept = int(np.sum(keep_mask))
    pct = 100 * n_kept / len(map_data)
    print(f"Retained vertices: {n_kept} ({pct:.1f}%)")

    return thresholded, n_kept, cutoff


def find_fixed_effects_contrasts(subject, contrast_base):
    """
    Find all fixed-effects contrast z-maps for a given subject.
    
    Returns a dict: {task: {contrast: z_map_path}}
    """
    subject_dir = os.path.join(contrast_base, f'sub-{subject}')
    
    print(f"  Looking for contrasts in: {subject_dir}")
    
    if not os.path.exists(subject_dir):
        print(f"  ERROR: Subject directory does not exist!")
        print(f"  Base directory exists: {os.path.exists(contrast_base)}")
        if os.path.exists(contrast_base):
            available = os.listdir(contrast_base)
            print(f"  Available subjects in base: {available[:5]}")
        return {}
    
    # Debug: List what's in the subject directory
    try:
        contents = os.listdir(subject_dir)
        print(f"  Subject directory contents ({len(contents)} items): {sorted(contents)}")
    except Exception as e:
        print(f"  ERROR listing subject directory: {e}")
    
    # Find all fixed-effects task directories
    glob_pattern = os.path.join(subject_dir, 'res_task-*_space-fsLR_dir-ffx')
    #print(f"  Glob pattern: {glob_pattern}")
    task_dirs = sorted(glob.glob(glob_pattern))
    #print(f"  Found {len(task_dirs)} task directories: {task_dirs}")
    
    results = {}
    
    for task_dir in task_dirs:
        # Extract task name from directory like 'res_task-HcpWm_space-fsLR_dir-ffx'
        dirname = os.path.basename(task_dir)
        # Remove 'res_task-' prefix and '_space-fsLR_dir-ffx' suffix
        task = dirname.replace('res_task-', '').replace('_space-fsLR_dir-ffx', '')
        
        z_map_dir = os.path.join(task_dir, 'z_score_maps')
        
        #print(f"    Processing task '{task}':")
        #print(f"      Task dir: {task_dir}")
        #print(f"      Z-map dir: {z_map_dir}")
        #print(f"      Z-map dir exists: {os.path.exists(z_map_dir)}")
        
        if not os.path.exists(z_map_dir):
            if os.path.exists(task_dir):
                # Check what's actually in the task directory
                try:
                    task_contents = os.listdir(task_dir)
                    print(f"      Task directory contents: {task_contents}")
                except Exception as e:
                    print(f"      Error listing task directory: {e}")
            continue
        
        # Find all z-score maps
        glob_pattern_z = os.path.join(z_map_dir, '*.dscalar.nii')
        #print(f"      Glob pattern for z-maps: {glob_pattern_z}")
        z_maps = glob.glob(glob_pattern_z)
        #print(f"      Found {len(z_maps)} z-maps")
        
        if not z_maps:
            try:
                z_contents = os.listdir(z_map_dir)
                print(f"      Z-map directory contents: {z_contents}")
            except Exception as e:
                print(f"      Error listing z-map directory: {e}")
            continue
        
        if task not in results:
            results[task] = {}
        
        for z_map_path in z_maps:
            contrast = os.path.basename(z_map_path).replace('.dscalar.nii', '')
            results[task][contrast] = z_map_path
    
    return results


def compute_md_map(subject, contrast_base, output_dir=None, save_individual=True, smoothing_fwhm=4.0,
                   threshold=None, threshold_percent=None, wb_command=DEFAULT_WB_COMMAND,
                   left_surface=DEFAULT_LEFT_SURFACE, right_surface=DEFAULT_RIGHT_SURFACE):
    """
    Compute the MD system map for a single subject by averaging MD-related contrasts.
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., '01', '02')
    contrast_base : str
        Base directory containing contrast maps
    output_dir : str, optional
        Directory to save output maps
    save_individual : bool
        Whether to save individual contrast contributions
    smoothing_fwhm : float
        FWHM of Workbench geodesic smoothing kernel in mm (0 = no smoothing)
    threshold : float, optional
        Z-score threshold for identifying reliable MD vertices.
        If provided, saves both thresholded and unthresholded maps.
    threshold_percent : float, optional
        Keep top X percent of vertices by value (e.g., 10 for top 10%).
        If provided, saves both thresholded and unthresholded maps.
    
    Returns
    -------
    md_map : np.ndarray
        Average z-score map across all available MD contrasts
    n_contrasts : int
        Number of contrasts that contributed to the map
    """
    print(f"\n{'='*60}")
    print(f"Processing subject {subject}")
    print(f"{'='*60}")
    
    # Find all available contrasts for this subject
    available_contrasts = find_fixed_effects_contrasts(subject, contrast_base)
    
    if not available_contrasts:
        print(f"No contrasts found for subject {subject}")
        return None, 0
    
    # Load MD-related contrasts
    md_maps = []
    md_info = []  # Keep track of which contrasts were used
    template_img = None
    smoothing_temp_dir = tempfile.mkdtemp(prefix=f'md_smooth_sub_{subject}_') if smoothing_fwhm > 0 else None
    smoothing_attempts = 0
    smoothing_successes = 0
    
    for task, contrast in MD_CONTRAST_LIST:
        if task in available_contrasts and contrast in available_contrasts[task]:
            z_map_path = available_contrasts[task][contrast]
            load_path = z_map_path

            if smoothing_fwhm > 0:
                smoothing_attempts += 1
                smoothed_path = os.path.join(smoothing_temp_dir, f'{task}_{contrast}_smooth.dscalar.nii')
                try:
                    smooth_cifti_with_workbench(
                        input_cifti=z_map_path,
                        output_cifti=smoothed_path,
                        smoothing_fwhm=smoothing_fwhm,
                        wb_command=wb_command,
                        left_surface=left_surface,
                        right_surface=right_surface,
                    )
                    load_path = smoothed_path
                    smoothing_successes += 1
                except Exception as e:
                    print(f"  ! Workbench smoothing failed for {task}/{contrast}, using unsmoothed map: {e}")

            data, img = load_contrast_map(load_path)
            
            if data is not None:
                md_maps.append(data)
                md_info.append((task, contrast))
                if template_img is None:
                    template_img = img
                print(f"  ✓ Loaded: {task}/{contrast}")
            else:
                print(f"  ✗ Failed to load: {task}/{contrast}")
        else:
            print(f"  - Not available: {task}/{contrast}")
    
    if not md_maps:
        print(f"No MD contrasts available for subject {subject}")
        return None, 0
    
    # Compute average MD map
    md_maps_array = np.array(md_maps)
    md_map_mean = np.mean(md_maps_array, axis=0)
    md_map_std = np.std(md_maps_array, axis=0)
    
    if smoothing_fwhm > 0:
        if smoothing_attempts == 0:
            print(f"\nSmoothing requested (FWHM={smoothing_fwhm}mm), but no maps were available to smooth.")
        elif smoothing_successes == smoothing_attempts:
            print(f"\nApplied Workbench CIFTI smoothing to input contrasts (FWHM={smoothing_fwhm}mm)")
        elif smoothing_successes == 0:
            print(
                f"\nWARNING: Smoothing requested (FWHM={smoothing_fwhm}mm) but all smoothing attempts failed. "
                "Using unsmoothed contrasts."
            )
        else:
            print(
                f"\nApplied Workbench smoothing to {smoothing_successes}/{smoothing_attempts} contrasts "
                f"(FWHM={smoothing_fwhm}mm)."
            )
    
    n_contrasts = len(md_maps)
    print(f"\nCombined {n_contrasts} MD contrasts")
    print(f"Mean z-score: {np.mean(md_map_mean):.3f} ± {np.std(md_map_mean):.3f}")
    print(f"Max z-score: {np.max(md_map_mean):.3f}")
    
    # Apply threshold if requested
    md_map_thresholded = None
    threshold_suffix = None
    if threshold_percent is not None:
        md_map_thresholded, _, cutoff = threshold_map_top_percent(
            md_map_mean,
            threshold_percent,
            label='Subject MD mean map',
        )
        threshold_suffix = f'top{threshold_percent:g}pct_cutoff{cutoff:.3f}'
    elif threshold is not None:
        md_map_thresholded, _ = threshold_map(md_map_mean, threshold, label='Subject MD mean map')
        threshold_suffix = f'z{threshold:g}'
    
    # Save outputs if requested
    if output_dir and template_img:
        subject_output_dir = os.path.join(output_dir, f'sub-{subject}')
        os.makedirs(subject_output_dir, exist_ok=True)
        
        # Save mean MD map
        mean_2d = md_map_mean.reshape(1, -1)
        brain_models_axis = template_img.header.get_axis(1)
        scalar_axis = nib.cifti2.ScalarAxis(['MD_mean'])
        new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_models_axis))
        mean_img = nib.Cifti2Image(mean_2d, header=new_header)
        mean_path = os.path.join(subject_output_dir, f'sub-{subject}_MD_mean.dscalar.nii')
        nib.save(mean_img, mean_path)
        print(f"\n✓ Saved mean MD map: {mean_path}")
        
        # Save thresholded MD map if threshold was applied
        if md_map_thresholded is not None:
            thresh_2d = md_map_thresholded.reshape(1, -1)
            scalar_axis = nib.cifti2.ScalarAxis([f'MD_mean_thresh_{threshold_suffix}'])
            new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_models_axis))
            thresh_img = nib.Cifti2Image(thresh_2d, header=new_header)
            thresh_path = os.path.join(subject_output_dir, f'sub-{subject}_MD_mean_thresh_{threshold_suffix}.dscalar.nii')
            nib.save(thresh_img, thresh_path)
            print(f"✓ Saved thresholded MD map: {thresh_path}")
        
        # Save std MD map
        std_2d = md_map_std.reshape(1, -1)
        scalar_axis = nib.cifti2.ScalarAxis(['MD_std'])
        new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_models_axis))
        std_img = nib.Cifti2Image(std_2d, header=new_header)
        std_path = os.path.join(subject_output_dir, f'sub-{subject}_MD_std.dscalar.nii')
        nib.save(std_img, std_path)
        print(f"✓ Saved std MD map: {std_path}")
        
        # Save info about which contrasts were used
        info_path = os.path.join(subject_output_dir, f'sub-{subject}_MD_contrasts.txt')
        with open(info_path, 'w') as f:
            f.write(f"MD System Mapping for subject {subject}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Number of contrasts: {n_contrasts}\n\n")
            f.write("Contrasts used:\n")
            for task, contrast in md_info:
                f.write(f"  - {task}: {contrast}\n")
        print(f"✓ Saved contrast info: {info_path}")
        
        if save_individual:
            # Save each individual contrast contribution
            individual_dir = os.path.join(subject_output_dir, 'individual_contrasts')
            os.makedirs(individual_dir, exist_ok=True)
            
            for i, (task, contrast) in enumerate(md_info):
                contrast_2d = md_maps_array[i].reshape(1, -1)
                scalar_axis = nib.cifti2.ScalarAxis([f'{task}_{contrast}'])
                new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_models_axis))
                contrast_img = nib.Cifti2Image(contrast_2d, header=new_header)
                contrast_path = os.path.join(individual_dir, f'{task}_{contrast}.dscalar.nii')
                nib.save(contrast_img, contrast_path)
            print(f"✓ Saved {n_contrasts} individual contrast maps")
    
    if smoothing_temp_dir and os.path.exists(smoothing_temp_dir):
        for temp_file in glob.glob(os.path.join(smoothing_temp_dir, '*.dscalar.nii')):
            try:
                os.remove(temp_file)
            except OSError:
                pass
        try:
            os.rmdir(smoothing_temp_dir)
        except OSError:
            pass

    return md_map_mean, n_contrasts


def compute_group_md_map(subjects, contrast_base, output_dir, smoothing_fwhm=4.0, threshold=None,
                         threshold_percent=None,
                         wb_command=DEFAULT_WB_COMMAND,
                         left_surface=DEFAULT_LEFT_SURFACE,
                         right_surface=DEFAULT_RIGHT_SURFACE):
    """
    Compute group-level MD map by averaging across subjects.

    Also computes group-average maps for each MD task contrast.
    
    Parameters
    ----------
    subjects : list
        List of subject IDs
    contrast_base : str
        Base directory containing contrast maps
    output_dir : str
        Directory to save group output
    smoothing_fwhm : float
        FWHM of Gaussian smoothing kernel in mm (0 = no smoothing)
    threshold : float, optional
        Z-score threshold for group map
    threshold_percent : float, optional
        Keep top X percent of group vertices by value
    """
    print(f"\n{'='*60}")
    print(f"Computing Group-Level MD Map")
    print(f"{'='*60}")
    
    subject_maps = []
    valid_subjects = []
    template_img = None
    contrast_group_maps = defaultdict(list)
    contrast_group_subjects = defaultdict(list)
    
    for subject in subjects:
        available_contrasts = find_fixed_effects_contrasts(subject, contrast_base)

        # Build group-average map for each MD task contrast separately.
        smoothing_temp_dir = tempfile.mkdtemp(prefix=f'md_group_contrast_smooth_sub_{subject}_') if smoothing_fwhm > 0 else None
        for task, contrast in MD_CONTRAST_LIST:
            if task not in available_contrasts or contrast not in available_contrasts[task]:
                continue

            z_map_path = available_contrasts[task][contrast]
            load_path = z_map_path

            if smoothing_fwhm > 0:
                smoothed_path = os.path.join(smoothing_temp_dir, f'{task}_{contrast}_smooth.dscalar.nii')
                try:
                    smooth_cifti_with_workbench(
                        input_cifti=z_map_path,
                        output_cifti=smoothed_path,
                        smoothing_fwhm=smoothing_fwhm,
                        wb_command=wb_command,
                        left_surface=left_surface,
                        right_surface=right_surface,
                    )
                    load_path = smoothed_path
                except Exception as e:
                    print(
                        f"  ! Group per-contrast smoothing failed for sub-{subject} {task}/{contrast}, "
                        f"using unsmoothed map: {e}"
                    )

            contrast_data, contrast_img = load_contrast_map(load_path)
            if contrast_data is None:
                continue

            contrast_key = f'{task}_{contrast}'
            contrast_group_maps[contrast_key].append(contrast_data)
            contrast_group_subjects[contrast_key].append(subject)
            if template_img is None and contrast_img is not None:
                template_img = contrast_img

        if smoothing_temp_dir and os.path.exists(smoothing_temp_dir):
            for temp_file in glob.glob(os.path.join(smoothing_temp_dir, '*.dscalar.nii')):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
            try:
                os.rmdir(smoothing_temp_dir)
            except OSError:
                pass

        md_map, n_contrasts = compute_md_map(subject, contrast_base, 
                                             output_dir=output_dir, 
                                             save_individual=False,
                                             smoothing_fwhm=smoothing_fwhm,
                                             threshold=None,
                                             threshold_percent=None,
                                             wb_command=wb_command,
                                             left_surface=left_surface,
                                             right_surface=right_surface)  # Don't threshold individual maps for group
        if md_map is not None and n_contrasts >= 2:  # Require at least 2 contrasts
            subject_maps.append(md_map)
            valid_subjects.append(subject)
            
            # Get template image if we don't have one
            if template_img is None:
                available = find_fixed_effects_contrasts(subject, contrast_base)
                for task in available:
                    for contrast in available[task]:
                        _, img = load_contrast_map(available[task][contrast])
                        if img is not None:
                            template_img = img
                            break
                    if template_img is not None:
                        break
    
    if not subject_maps:
        print("No valid subject maps found for group analysis")
        return
    
    # Compute group statistics
    subject_maps_array = np.array(subject_maps)
    group_mean = np.mean(subject_maps_array, axis=0)
    group_std = np.std(subject_maps_array, axis=0)
    group_sem = group_std / np.sqrt(len(subject_maps))
    
    # Group map is computed from already smoothed subject maps; avoid double smoothing.
    
    print(f"\nGroup analysis: {len(valid_subjects)} subjects")
    print(f"Valid subjects: {', '.join(valid_subjects)}")
    print(f"Mean group z-score: {np.mean(group_mean):.3f}")
    print(f"Max group z-score: {np.max(group_mean):.3f}")
    
    # Apply threshold if requested
    group_mean_thresholded = None
    threshold_suffix = None
    if threshold_percent is not None:
        group_mean_thresholded, _, cutoff = threshold_map_top_percent(
            group_mean,
            threshold_percent,
            label='Group MD mean map',
        )
        threshold_suffix = f'top{threshold_percent:g}pct_cutoff{cutoff:.3f}'
    elif threshold is not None:
        group_mean_thresholded, _ = threshold_map(group_mean, threshold, label='Group MD mean map')
        threshold_suffix = f'z{threshold:g}'
    
    # Save group maps
    group_dir = os.path.join(output_dir, 'group')
    os.makedirs(group_dir, exist_ok=True)
    
    if template_img:
        brain_models_axis = template_img.header.get_axis(1)
        
        # Save group mean
        mean_2d = group_mean.reshape(1, -1)
        scalar_axis = nib.cifti2.ScalarAxis(['MD_group_mean'])
        new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_models_axis))
        mean_img = nib.Cifti2Image(mean_2d, header=new_header)
        mean_path = os.path.join(group_dir, 'group_MD_mean.dscalar.nii')
        nib.save(mean_img, mean_path)
        print(f"\n✓ Saved group mean: {mean_path}")
        
        # Save group thresholded map if threshold was applied
        if group_mean_thresholded is not None:
            thresh_2d = group_mean_thresholded.reshape(1, -1)
            scalar_axis = nib.cifti2.ScalarAxis([f'MD_group_mean_thresh_{threshold_suffix}'])
            new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_models_axis))
            thresh_img = nib.Cifti2Image(thresh_2d, header=new_header)
            thresh_path = os.path.join(group_dir, f'group_MD_mean_thresh_{threshold_suffix}.dscalar.nii')
            nib.save(thresh_img, thresh_path)
            print(f"✓ Saved group thresholded mean: {thresh_path}")
        
        # Save group std
        std_2d = group_std.reshape(1, -1)
        scalar_axis = nib.cifti2.ScalarAxis(['MD_group_std'])
        new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_models_axis))
        std_img = nib.Cifti2Image(std_2d, header=new_header)
        std_path = os.path.join(group_dir, 'group_MD_std.dscalar.nii')
        nib.save(std_img, std_path)
        print(f"✓ Saved group std: {std_path}")
        
        # Save group SEM
        sem_2d = group_sem.reshape(1, -1)
        scalar_axis = nib.cifti2.ScalarAxis(['MD_group_sem'])
        new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_models_axis))
        sem_img = nib.Cifti2Image(sem_2d, header=new_header)
        sem_path = os.path.join(group_dir, 'group_MD_sem.dscalar.nii')
        nib.save(sem_img, sem_path)
        print(f"✓ Saved group SEM: {sem_path}")

        # Save group-average map for each MD task contrast.
        for task, contrast in MD_CONTRAST_LIST:
            contrast_key = f'{task}_{contrast}'
            if contrast_key not in contrast_group_maps or not contrast_group_maps[contrast_key]:
                print(f"- No data for group contrast map: {task}/{contrast}")
                continue

            contrast_array = np.array(contrast_group_maps[contrast_key])
            contrast_mean = np.mean(contrast_array, axis=0)

            contrast_2d = contrast_mean.reshape(1, -1)
            scalar_axis = nib.cifti2.ScalarAxis([f'{contrast_key}_group_mean'])
            new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_models_axis))
            contrast_img = nib.Cifti2Image(contrast_2d, header=new_header)

            contrast_out = os.path.join(group_dir, f'group_{task}_{contrast}_mean.dscalar.nii')
            nib.save(contrast_img, contrast_out)

            n_sub = len(contrast_group_subjects[contrast_key])
            print(f"✓ Saved group contrast mean ({n_sub} subj): {contrast_out}")
        
        # Save subject list
        info_path = os.path.join(group_dir, 'group_MD_info.txt')
        with open(info_path, 'w') as f:
            f.write(f"Group MD System Mapping\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Number of subjects: {len(valid_subjects)}\n\n")
            f.write("Subjects included:\n")
            for subj in valid_subjects:
                f.write(f"  - sub-{subj}\n")

            f.write("\nGroup contrast maps:\n")
            for task, contrast in MD_CONTRAST_LIST:
                contrast_key = f'{task}_{contrast}'
                n_sub = len(contrast_group_subjects.get(contrast_key, []))
                if n_sub == 0:
                    f.write(f"  - {task}/{contrast}: no data\n")
                else:
                    f.write(f"  - {task}/{contrast}: n={n_sub}\n")
        print(f"✓ Saved group info: {info_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Map the Multiple Demand system using difficulty-based contrasts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single subject
    python md_mapping.py --subject 01 --contrast-base /path/to/contrasts --output /path/to/output
    
    # With smoothing (4mm FWHM - recommended)
    python md_mapping.py --subject 01 --contrast-base /path/to/contrasts --output /path/to/output --smooth 4
    
    # Process all subjects with smoothing and group map
    python md_mapping.py --all-subjects --contrast-base /path/to/contrasts --output /path/to/output --smooth 4 --group
        """
    )
    
    parser.add_argument('--subject', type=str, help='Single subject ID (e.g., 01)')
    parser.add_argument('--subjects', nargs='+', help='List of subject IDs')
    parser.add_argument('--all-subjects', action='store_true', 
                       help='Process all available subjects (default if no subject flags are provided)')
    parser.add_argument('--contrast-base', type=str, required=True,
                       help='Base directory containing contrast maps')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for MD maps')
    parser.add_argument('--smooth', type=float, default=4.0, metavar='FWHM',
                       help='Workbench geodesic smoothing FWHM in mm (default: 4.0, set to 0 for no smoothing)')
    parser.add_argument('--wb-command', type=str, default=DEFAULT_WB_COMMAND,
                       help=f'Path to wb_command (default: {DEFAULT_WB_COMMAND})')
    parser.add_argument('--left-surface', type=str, default=DEFAULT_LEFT_SURFACE,
                       help='Path to left midthickness surface for geodesic smoothing')
    parser.add_argument('--right-surface', type=str, default=DEFAULT_RIGHT_SURFACE,
                       help='Path to right midthickness surface for geodesic smoothing')
    parser.add_argument('--threshold', type=float, default=None, metavar='Z',
                       help='Z-score threshold for identifying MD regions (e.g., 2.3 for p<0.01 one-tailed, 3.1 for p<0.001)')
    parser.add_argument('--threshold-percent', type=float, default=None, metavar='PCT',
                       help='Keep top PCT%% vertices by value (e.g., 10 for top 10%%).')
    parser.add_argument('--group', action='store_true',
                       help='Compute group-level MD map')
    parser.add_argument('--no-individual-contrasts', action='store_true',
                       help='Do not save individual contrast contributions')
    
    args = parser.parse_args()

    if args.threshold is not None and args.threshold_percent is not None:
        parser.error('Use either --threshold or --threshold-percent, not both.')

    if args.threshold_percent is not None and (args.threshold_percent <= 0 or args.threshold_percent > 100):
        parser.error('--threshold-percent must be in (0, 100].')
    
    # Determine which subjects to process
    subjects = []
    
    if args.subject:
        subjects = [args.subject]
    elif args.subjects:
        subjects = args.subjects
    else:
        subject_dirs = glob.glob(os.path.join(args.contrast_base, 'sub-*'))
        subjects = sorted([os.path.basename(d).replace('sub-', '') for d in subject_dirs])
        if args.all_subjects:
            print(f"Found {len(subjects)} subjects: {', '.join(subjects)}")
        else:
            print("No subject selection flags provided, defaulting to all available subjects.")
            print(f"Found {len(subjects)} subjects: {', '.join(subjects)}")

    if not subjects:
        parser.error(f"No subjects found under: {args.contrast_base}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process individual subjects
    for subject in subjects:
        compute_md_map(subject, args.contrast_base, args.output, 
                      save_individual=not args.no_individual_contrasts,
                      smoothing_fwhm=args.smooth,
                      threshold=args.threshold,
                      threshold_percent=args.threshold_percent,
                      wb_command=args.wb_command,
                      left_surface=args.left_surface,
                      right_surface=args.right_surface)
    
    # Compute group map if requested
    if args.group and len(subjects) > 1:
        compute_group_md_map(subjects, args.contrast_base, args.output, 
                            smoothing_fwhm=args.smooth,
                            threshold=args.threshold,
                            threshold_percent=args.threshold_percent,
                            wb_command=args.wb_command,
                            left_surface=args.left_surface,
                            right_surface=args.right_surface)
    
    print(f"\n{'='*60}")
    print("MD Mapping Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output}")
    if args.smooth > 0:
        print(f"Workbench geodesic smoothing applied: FWHM={args.smooth}mm")
    if args.threshold is not None:
        print(f"Threshold applied: z > {args.threshold}")
    if args.threshold_percent is not None:
        print(f"Threshold applied: top {args.threshold_percent}% of vertices")


if __name__ == '__main__':
    main()
