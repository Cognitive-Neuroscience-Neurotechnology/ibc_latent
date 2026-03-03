"""
Mapping the MD system in every individual by using task contrasts with difficulty manipulation.

Contrasts to use:
    HcpWm - 2back-0back
    ItemRecognition - encode5-encode1 (probe5_mem-probe1_mem, probe5_new-probe1_new?)
    MDTB - 2back_hard-easy, search_hard-easy, semantic_hard-easy, finger_complex-simple
    GoodBadUgly - dot_hard-easy
    Stroop - incongruent-congruent
    Catell - hard-easy
"""

import os
import sys
import glob
import numpy as np
import nibabel as nib
from collections import defaultdict
import argparse
import subprocess
from scipy.ndimage import gaussian_filter

# ============================================================================
# MD-RELATED CONTRASTS DEFINITION
# ============================================================================

MD_CONTRASTS = {
    'HcpWm': ['2back-0back'],
    'ItemRecognition': ['encode5-encode1'],  # Could also include probe5-probe1
    'MDTB': ['2back_hard-easy', 'search_hard-easy', 'semantic_hard-easy', 'finger_complex-simple'],
    'GoodBadUgly': ['dot_hard-easy'],
    'Stroop': ['incongruent-congruent'],
    'Catell': ['hard-easy'],
}

# Flatten the dictionary into a list of tuples (task, contrast)
MD_CONTRAST_LIST = []
for task, contrasts in MD_CONTRASTS.items():
    for contrast in contrasts:
        MD_CONTRAST_LIST.append((task, contrast))

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


def smooth_surface_map(data, smoothing_fwhm=4.0):
    """
    Apply Gaussian smoothing to surface data.
    
    Parameters
    ----------
    data : np.ndarray
        1D array of vertex values
    smoothing_fwhm : float
        Full-width at half-maximum of smoothing kernel in mm
        Typical values: 2-6 mm for surface data
    
    Returns
    -------
    smoothed_data : np.ndarray
        Smoothed data array
    """
    if smoothing_fwhm <= 0:
        return data
    
    # Convert FWHM to sigma for Gaussian kernel
    # FWHM = 2.355 * sigma for Gaussian distribution
    sigma = smoothing_fwhm / 2.355
    
    # Simple Gaussian smoothing (treats surface as a regular grid)
    # For true surface-aware smoothing, use wb_command
    smoothed = gaussian_filter(data, sigma=sigma, mode='nearest')
    
    return smoothed


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
    print(f"  Glob pattern: {glob_pattern}")
    task_dirs = sorted(glob.glob(glob_pattern))
    print(f"  Found {len(task_dirs)} task directories: {task_dirs}")
    
    results = {}
    
    for task_dir in task_dirs:
        # Extract task name from directory like 'res_task-HcpWm_space-fsLR_dir-ffx'
        dirname = os.path.basename(task_dir)
        # Remove 'res_task-' prefix and '_space-fsLR_dir-ffx' suffix
        task = dirname.replace('res_task-', '').replace('_space-fsLR_dir-ffx', '')
        
        z_map_dir = os.path.join(task_dir, 'z_score_maps')
        
        print(f"    Processing task '{task}':")
        print(f"      Task dir: {task_dir}")
        print(f"      Z-map dir: {z_map_dir}")
        print(f"      Z-map dir exists: {os.path.exists(z_map_dir)}")
        
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
        print(f"      Glob pattern for z-maps: {glob_pattern_z}")
        z_maps = glob.glob(glob_pattern_z)
        print(f"      Found {len(z_maps)} z-maps")
        
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


def compute_md_map(subject, contrast_base, output_dir=None, save_individual=True, smoothing_fwhm=4.0):
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
        FWHM of Gaussian smoothing kernel in mm (0 = no smoothing)
    
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
    
    for task, contrast in MD_CONTRAST_LIST:
        if task in available_contrasts and contrast in available_contrasts[task]:
            z_map_path = available_contrasts[task][contrast]
            data, img = load_contrast_map(z_map_path)
            
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
    
    # Apply smoothing if requested
    if smoothing_fwhm > 0:
        print(f"\nApplying Gaussian smoothing (FWHM={smoothing_fwhm}mm)...")
        md_map_mean = smooth_surface_map(md_map_mean, smoothing_fwhm)
        md_map_std = smooth_surface_map(md_map_std, smoothing_fwhm)
    
    n_contrasts = len(md_maps)
    print(f"\nCombined {n_contrasts} MD contrasts")
    print(f"Mean z-score: {np.mean(md_map_mean):.3f} ± {np.std(md_map_mean):.3f}")
    print(f"Max z-score: {np.max(md_map_mean):.3f}")
    
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
    
    return md_map_mean, n_contrasts


def compute_group_md_map(subjects, contrast_base, output_dir, smoothing_fwhm=4.0):
    """
    Compute group-level MD map by averaging across subjects.
    
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
    """
    print(f"\n{'='*60}")
    print(f"Computing Group-Level MD Map")
    print(f"{'='*60}")
    
    subject_maps = []
    valid_subjects = []
    template_img = None
    
    for subject in subjects:
        md_map, n_contrasts = compute_md_map(subject, contrast_base, 
                                             output_dir=output_dir, 
                                             save_individual=False,
                                             smoothing_fwhm=smoothing_fwhm)
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
    
    # Apply additional smoothing to group map for cleaner visualization
    if smoothing_fwhm > 0:
        print(f"\nApplying additional smoothing to group map (FWHM={smoothing_fwhm}mm)...")
        group_mean = smooth_surface_map(group_mean, smoothing_fwhm)
        group_std = smooth_surface_map(group_std, smoothing_fwhm)
        group_sem = smooth_surface_map(group_sem, smoothing_fwhm)
    
    print(f"\nGroup analysis: {len(valid_subjects)} subjects")
    print(f"Valid subjects: {', '.join(valid_subjects)}")
    print(f"Mean group z-score: {np.mean(group_mean):.3f}")
    print(f"Max group z-score: {np.max(group_mean):.3f}")
    
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
        
        # Save subject list
        info_path = os.path.join(group_dir, 'group_MD_info.txt')
        with open(info_path, 'w') as f:
            f.write(f"Group MD System Mapping\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Number of subjects: {len(valid_subjects)}\n\n")
            f.write("Subjects included:\n")
            for subj in valid_subjects:
                f.write(f"  - sub-{subj}\n")
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
                       help='Gaussian smoothing FWHM in mm (default: 4.0, set to 0 for no smoothing)')
    parser.add_argument('--group', action='store_true',
                       help='Compute group-level MD map')
    parser.add_argument('--no-individual-contrasts', action='store_true',
                       help='Do not save individual contrast contributions')
    
    args = parser.parse_args()
    
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
                      smoothing_fwhm=args.smooth)
    
    # Compute group map if requested
    if args.group and len(subjects) > 1:
        compute_group_md_map(subjects, args.contrast_base, args.output, args.smooth)
    
    print(f"\n{'='*60}")
    print("MD Mapping Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output}")
    if args.smooth > 0:
        print(f"Smoothing applied: FWHM={args.smooth}mm")


if __name__ == '__main__':
    main()
