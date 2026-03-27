"""
Parcel-based MD system mapping using HCP-MMP1 360 parcellation.

This script applies the Glasser et al. (2016) 360-parcel cortical parcellation
to MD contrast maps, providing a more abstracted and spatially-organized view
of MD system activation patterns.
"""

import os
import sys
import glob
import numpy as np
import nibabel as nib
from collections import defaultdict
import argparse
import subprocess

# Import from the main md_mapping script
from md_mapping import MD_CONTRAST_LIST, find_fixed_effects_contrasts, load_contrast_map


def get_cortical_indices_from_cifti(img):
    """Return cortical grayordinate indices (left + right cortex) from a CIFTI image."""
    brain_axis = img.header.get_axis(1)
    cortical_indices = []

    for bm in brain_axis.iter_structures():
        if len(bm) >= 2:
            structure_name = bm[0]
            data_indices = bm[1]
        else:
            raise ValueError(f"Unexpected iter_structures() entry format: {bm}")

        if 'CORTEX_LEFT' not in structure_name and 'CORTEX_RIGHT' not in structure_name:
            continue

        if isinstance(data_indices, slice):
            cortical_indices.extend(range(data_indices.start, data_indices.stop, data_indices.step or 1))
        else:
            cortical_indices.extend(np.asarray(data_indices).tolist())

    if not cortical_indices:
        raise ValueError("No cortical structures found in CIFTI brain model axis.")

    return np.array(sorted(cortical_indices), dtype=int)


def parcel_values_to_dense_scalar(parcel_values, parcellation, cortical_indices, n_grayordinates):
    """Expand parcel-level values back to dense grayordinate space for wb_view visualization."""
    unique_parcels = np.unique(parcellation)
    unique_parcels = unique_parcels[unique_parcels > 0]

    if len(parcel_values) != len(unique_parcels):
        raise ValueError(
            f"Parcel value length ({len(parcel_values)}) does not match number of parcels ({len(unique_parcels)})."
        )

    cortical_dense = np.zeros(len(parcellation), dtype=float)
    for i, parcel_id in enumerate(unique_parcels):
        cortical_dense[parcellation == parcel_id] = parcel_values[i]

    dense = np.zeros(n_grayordinates, dtype=float)
    dense[cortical_indices] = cortical_dense
    return dense


def save_parcel_dscalar(parcel_values, map_name, out_path, template_img, parcellation, cortical_indices):
    """Save parcel-level values as a dense scalar CIFTI map for Workbench."""
    brain_axis = template_img.header.get_axis(1)
    n_grayordinates = template_img.shape[1]
    dense_map = parcel_values_to_dense_scalar(parcel_values, parcellation, cortical_indices, n_grayordinates)

    dense_2d = dense_map.reshape(1, -1)
    scalar_axis = nib.cifti2.ScalarAxis([map_name])
    new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis))
    out_img = nib.Cifti2Image(dense_2d, header=new_header)
    nib.save(out_img, out_path)


def load_hcp_parcellation(parcellation_path=None):
    """
    Load the HCP-MMP1 360-parcel cortical parcellation (32k fsLR).
    
    Parameters
    ----------
    parcellation_path : str, optional
        Path to HCP parcellation dlabel.nii file (32k fsLR space).
        If None, tries standard locations.
    
    Returns
    -------
    parcellation : np.ndarray
        Parcel labels for cortical vertices (shape: 59412)
    parcel_names : list
        Names of parcels
    img : Cifti2Image
        Template CIFTI image
    """
    # Try standard locations if not provided
    if parcellation_path is None:
        potential_paths = [
            '/home/hmueller2/atlases/fsLR/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii',
            '/usr/share/workbench/resources/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii',
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                parcellation_path = path
                break
    
    if parcellation_path is None or not os.path.exists(parcellation_path):
        raise FileNotFoundError(
            "HCP 32k_fsLR parcellation file not found. Please provide path with --parcellation-path\n"
            "You can download the Glasser et al. 360-parcel atlas from: https://balsa.wustl.edu/file/pkXDZ\n"
            "File should be: Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
        )
    
    print(f"Loading HCP 360 parcellation from: {parcellation_path}")
    img = nib.load(parcellation_path)
    full_data = img.get_fdata()[0]  # Shape: (n_grayordinates,) - may include subcortex
    
    cortical_indices = get_cortical_indices_from_cifti(img)
    parcellation_data = full_data[cortical_indices]
    print(f"Extracted cortical parcellation ({len(cortical_indices)} vertices)")
    
    # Get parcel names from CIFTI label table
    label_axis = img.header.get_axis(0)
    parcel_info = label_axis.label[0]  # First map
    parcel_names = []
    parcel_keys = sorted(parcel_info.keys())
    
    for key in parcel_keys:
        if key == 0:  # Skip unlabeled vertices
            continue
        parcel_names.append(parcel_info[key][0])  # parcel_info[key] = (name, rgba)
    
    print(f"Loaded {len(parcel_names)} cortical parcels from 360-parcel atlas")
    
    return parcellation_data, parcel_names, img, cortical_indices


def extract_cortical_data(data_cifti, img):
    """
    Extract cortical vertices from CIFTI using brain model structure.
    
    Respects the actual CIFTI organization: uses the brain model axis to identify
    which grayordinates are cortical (left + right cortex), properly handling
    cortical wall and any subcortical structures.
    
    Parameters
    ----------
    data_cifti : np.ndarray
        Full CIFTI grayordinate data (shape: n_grayordinates)
    img : Cifti2Image
        CIFTI image to extract structure from
    
    Returns
    -------
    cortical_data : np.ndarray
        Cortical-only data (left cortex + right cortex)
    cortical_indices : np.ndarray
        Indices of cortical grayordinates in the original data
    """
    # Get the brain model axis (axis 1 for CIFTI scalars)
    try:
        brain_axis = img.header.get_axis(1)
    except Exception as e:
        raise ValueError(f"Could not extract brain model axis from CIFTI: {e}")
    
    cortical_indices = []
    
    # Iterate through brain models and collect cortical vertex indices.
    # nibabel versions can return either (name, data_indices, model)
    # or an older 2-item variant. Support both.
    for bm in brain_axis.iter_structures():
        if len(bm) >= 2:
            structure_name = bm[0]
            data_indices = bm[1]
        else:
            raise ValueError(f"Unexpected iter_structures() entry format: {bm}")

        # Include left and right cortex only
        if 'CORTEX_LEFT' not in structure_name and 'CORTEX_RIGHT' not in structure_name:
            continue

        if isinstance(data_indices, slice):
            cortical_indices.extend(range(data_indices.start, data_indices.stop, data_indices.step or 1))
        else:
            cortical_indices.extend(np.asarray(data_indices).tolist())
    
    if not cortical_indices:
        raise ValueError(
            "No cortical structures found in CIFTI brain model. "
            "Expected CORTEX_LEFT and/or CORTEX_RIGHT."
        )
    
    cortical_indices = np.array(sorted(cortical_indices))
    cortical_data = data_cifti[cortical_indices]
    
    print(f"  Extracted {len(cortical_indices)} cortical vertices from {len(data_cifti)} total grayordinates")
    
    return cortical_data, cortical_indices


def parcellate_map(data, parcellation):
    """
    Average a vertex-wise map within each parcel.
    
    Parameters
    ----------
    data : np.ndarray
        Vertex-wise data (1D array, shape should match parcellation)
    parcellation : np.ndarray
        Parcel labels for each vertex
    
    Returns
    -------
    parcel_values : np.ndarray
        Average value for each parcel
    """
    # Dimension check
    if len(data) != len(parcellation):
        raise ValueError(
            f"Data shape ({len(data)}) does not match parcellation shape ({len(parcellation)}). "
            f"Ensure both are in the same space (e.g., 32k fsLR cortex with {len(parcellation)} vertices)."
        )
    
    unique_parcels = np.unique(parcellation)
    unique_parcels = unique_parcels[unique_parcels > 0]  # Exclude 0 (unlabeled)
    
    parcel_values = np.zeros(len(unique_parcels))
    
    for i, parcel_id in enumerate(unique_parcels):
        mask = parcellation == parcel_id
        if np.any(mask):
            parcel_values[i] = np.mean(data[mask])
    
    return parcel_values


def compute_md_map_parcels(subject, contrast_base, parcellation, parcel_names,
                           output_dir=None, save_individual=True,
                           parcellation_template_img=None, cortical_indices=None):
    """
    Compute parcel-averaged MD system map for a single subject.
    
    Parameters
    ----------
    subject : str
        Subject ID
    contrast_base : str
        Base directory containing contrast maps
    parcellation : np.ndarray
        Parcel labels
    parcel_names : list
        Names of parcels
    output_dir : str, optional
        Output directory
    save_individual : bool
        Whether to save individual contrast contributions
    parcellation_template_img : Cifti2Image, optional
        CIFTI template used to write dense scalar parcel maps for wb_view
    cortical_indices : np.ndarray, optional
        Cortical grayordinate indices in the template CIFTI
    
    Returns
    -------
    md_parcel_values : np.ndarray
        Average MD z-scores for each parcel
    n_contrasts : int
        Number of contrasts used
    """
    print(f"\n{'='*60}")
    print(f"Processing subject {subject} (parcel-based)")
    print(f"{'='*60}")
    
    # Find all available contrasts
    available_contrasts = find_fixed_effects_contrasts(subject, contrast_base)
    
    if not available_contrasts:
        print(f"No contrasts found for subject {subject}")
        return None, 0
    
    # Load and parcellate MD-related contrasts
    parcel_arrays = []
    md_info = []
    
    for task, contrast in MD_CONTRAST_LIST:
        if task in available_contrasts and contrast in available_contrasts[task]:
            z_map_path = available_contrasts[task][contrast]
            data, img = load_contrast_map(z_map_path)
            
            if data is not None:
                # Extract cortical vertices only using CIFTI structure
                try:
                    cortical_data, _ = extract_cortical_data(data, img)
                except ValueError as e:
                    print(f"  ✗ Failed to extract cortical data for {task}/{contrast}: {e}")
                    continue
                
                # Parcellate the map
                try:
                    parcel_values = parcellate_map(cortical_data, parcellation)
                    parcel_arrays.append(parcel_values)
                    md_info.append((task, contrast))
                    print(f"  ✓ Loaded and parcellated: {task}/{contrast}")
                except ValueError as e:
                    print(f"  ✗ Parcellation error for {task}/{contrast}: {e}")
                    continue
            else:
                print(f"  ✗ Failed to load: {task}/{contrast}")
        else:
            print(f"  - Not available: {task}/{contrast}")
    
    if not parcel_arrays:
        print(f"No MD contrasts available for subject {subject}")
        return None, 0
    
    # Compute average across contrasts
    parcel_arrays = np.array(parcel_arrays)
    md_parcel_mean = np.mean(parcel_arrays, axis=0)
    md_parcel_std = np.std(parcel_arrays, axis=0)
    
    n_contrasts = len(parcel_arrays)
    print(f"\nCombined {n_contrasts} MD contrasts")
    print(f"Mean parcel z-score: {np.mean(md_parcel_mean):.3f} ± {np.std(md_parcel_mean):.3f}")
    print(f"Max parcel z-score: {np.max(md_parcel_mean):.3f}")
    
    # Save outputs if requested
    if output_dir:
        subject_output_dir = os.path.join(output_dir, f'sub-{subject}')
        os.makedirs(subject_output_dir, exist_ok=True)
        
        # Save as CSV for easy analysis
        import pandas as pd
        df = pd.DataFrame({
            'parcel_name': parcel_names,
            'md_zscore_mean': md_parcel_mean,
            'md_zscore_std': md_parcel_std,
        })
        csv_path = os.path.join(subject_output_dir, f'sub-{subject}_MD_parcels.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved parcel data: {csv_path}")
        
        # Also save as numpy array
        npz_path = os.path.join(subject_output_dir, f'sub-{subject}_MD_parcels.npz')
        np.savez(npz_path, 
                 mean=md_parcel_mean, 
                 std=md_parcel_std,
                 parcel_names=parcel_names,
                 contrasts_used=md_info)
        print(f"✓ Saved parcel array: {npz_path}")

        # Save Workbench-readable dense maps reconstructed from parcel values
        if parcellation_template_img is not None and cortical_indices is not None:
            mean_dscalar_path = os.path.join(subject_output_dir, f'sub-{subject}_MD_parcels_mean.dscalar.nii')
            save_parcel_dscalar(
                parcel_values=md_parcel_mean,
                map_name='MD_parcels_mean',
                out_path=mean_dscalar_path,
                template_img=parcellation_template_img,
                parcellation=parcellation,
                cortical_indices=cortical_indices,
            )
            print(f"✓ Saved Workbench parcel mean map: {mean_dscalar_path}")

            std_dscalar_path = os.path.join(subject_output_dir, f'sub-{subject}_MD_parcels_std.dscalar.nii')
            save_parcel_dscalar(
                parcel_values=md_parcel_std,
                map_name='MD_parcels_std',
                out_path=std_dscalar_path,
                template_img=parcellation_template_img,
                parcellation=parcellation,
                cortical_indices=cortical_indices,
            )
            print(f"✓ Saved Workbench parcel std map: {std_dscalar_path}")
        
        # Save info about which contrasts were used
        info_path = os.path.join(subject_output_dir, f'sub-{subject}_MD_parcels_info.txt')
        with open(info_path, 'w') as f:
            f.write(f"MD System Mapping (Parcel-Based) for subject {subject}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Number of parcels: {len(parcel_names)}\n")
            f.write(f"Number of contrasts: {n_contrasts}\n\n")
            f.write("Contrasts used:\n")
            for task, contrast in md_info:
                f.write(f"  - {task}: {contrast}\n")
        print(f"✓ Saved info: {info_path}")
        
        if save_individual:
            # Save each individual contrast contribution
            individual_dir = os.path.join(subject_output_dir, 'individual_contrasts')
            os.makedirs(individual_dir, exist_ok=True)
            
            for i, (task, contrast) in enumerate(md_info):
                df_ind = pd.DataFrame({
                    'parcel_name': parcel_names,
                    'zscore': parcel_arrays[i],
                })
                ind_path = os.path.join(individual_dir, f'{task}_{contrast}.csv')
                df_ind.to_csv(ind_path, index=False)
            print(f"✓ Saved {n_contrasts} individual contrast files")
    
    return md_parcel_mean, n_contrasts


def compute_group_md_map_parcels(subjects, contrast_base, parcellation, parcel_names, output_dir,
                                 parcellation_template_img=None, cortical_indices=None):
    """
    Compute group-level parcel-based MD map.
    
    Parameters
    ----------
    subjects : list
        List of subject IDs
    contrast_base : str
        Base directory containing contrast maps
    parcellation : np.ndarray
        Parcel labels
    parcel_names : list
        Names of parcels
    output_dir : str
        Output directory
    """
    print(f"\n{'='*60}")
    print(f"Computing Group-Level MD Map (Parcel-Based)")
    print(f"{'='*60}")
    
    subject_parcel_maps = []
    valid_subjects = []
    
    for subject in subjects:
        md_parcel_map, n_contrasts = compute_md_map_parcels(
            subject, contrast_base, parcellation, parcel_names,
            output_dir=output_dir, save_individual=False,
            parcellation_template_img=parcellation_template_img,
            cortical_indices=cortical_indices,
        )
        if md_parcel_map is not None and n_contrasts >= 2:
            subject_parcel_maps.append(md_parcel_map)
            valid_subjects.append(subject)
    
    if not subject_parcel_maps:
        print("No valid subject maps found for group analysis")
        return
    
    # Compute group statistics
    subject_parcel_maps = np.array(subject_parcel_maps)
    group_mean = np.mean(subject_parcel_maps, axis=0)
    group_std = np.std(subject_parcel_maps, axis=0)
    group_sem = group_std / np.sqrt(len(subject_parcel_maps))
    
    print(f"\nGroup analysis: {len(valid_subjects)} subjects")
    print(f"Valid subjects: {', '.join(valid_subjects)}")
    print(f"Mean group parcel z-score: {np.mean(group_mean):.3f}")
    print(f"Max group parcel z-score: {np.max(group_mean):.3f}")
    
    # Identify top MD parcels
    top_n = 20
    top_indices = np.argsort(group_mean)[-top_n:][::-1]
    print(f"\nTop {top_n} MD parcels:")
    for idx in top_indices:
        print(f"  {parcel_names[idx]}: z = {group_mean[idx]:.3f} ± {group_sem[idx]:.3f}")
    
    # Save group results
    group_dir = os.path.join(output_dir, 'group')
    os.makedirs(group_dir, exist_ok=True)
    
    import pandas as pd
    df_group = pd.DataFrame({
        'parcel_name': parcel_names,
        'md_zscore_mean': group_mean,
        'md_zscore_std': group_std,
        'md_zscore_sem': group_sem,
    })
    csv_path = os.path.join(group_dir, 'group_MD_parcels.csv')
    df_group.to_csv(csv_path, index=False)
    print(f"\n✓ Saved group parcel data: {csv_path}")
    
    # Save numpy array
    npz_path = os.path.join(group_dir, 'group_MD_parcels.npz')
    np.savez(npz_path,
             mean=group_mean,
             std=group_std,
             sem=group_sem,
             parcel_names=parcel_names,
             subjects=valid_subjects)
    print(f"✓ Saved group array: {npz_path}")

    if parcellation_template_img is not None and cortical_indices is not None:
        group_mean_dscalar = os.path.join(group_dir, 'group_MD_parcels_mean.dscalar.nii')
        save_parcel_dscalar(
            parcel_values=group_mean,
            map_name='MD_group_parcels_mean',
            out_path=group_mean_dscalar,
            template_img=parcellation_template_img,
            parcellation=parcellation,
            cortical_indices=cortical_indices,
        )
        print(f"✓ Saved Workbench group parcel mean map: {group_mean_dscalar}")

        group_sem_dscalar = os.path.join(group_dir, 'group_MD_parcels_sem.dscalar.nii')
        save_parcel_dscalar(
            parcel_values=group_sem,
            map_name='MD_group_parcels_sem',
            out_path=group_sem_dscalar,
            template_img=parcellation_template_img,
            parcellation=parcellation,
            cortical_indices=cortical_indices,
        )
        print(f"✓ Saved Workbench group parcel sem map: {group_sem_dscalar}")
    
    # Save subject list
    info_path = os.path.join(group_dir, 'group_MD_parcels_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"Group MD System Mapping (Parcel-Based)\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Number of parcels: {len(parcel_names)}\n")
        f.write(f"Number of subjects: {len(valid_subjects)}\n\n")
        f.write("Subjects included:\n")
        for subj in valid_subjects:
            f.write(f"  - sub-{subj}\n")
        f.write(f"\nTop {top_n} MD parcels:\n")
        for idx in top_indices:
            f.write(f"  {parcel_names[idx]}: z = {group_mean[idx]:.3f} ± {group_sem[idx]:.3f}\n")
    print(f"✓ Saved group info: {info_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Parcel-based MD system mapping using HCP-MMP1 360 parcellation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single subject
    python md_mapping_parcels.py --subject 01 --contrast-base /path/to/contrasts --output /path/to/output
    
    # Process all subjects with group map
    python md_mapping_parcels.py --all-subjects --group --contrast-base /path/to/contrasts --output /path/to/output
    
    # Specify custom parcellation file
    python md_mapping_parcels.py --all-subjects --parcellation-path /path/to/parcellation.dlabel.nii --contrast-base /path/to/contrasts --output /path/to/output
        """
    )
    
    parser.add_argument('--subject', type=str, help='Single subject ID (e.g., 01)')
    parser.add_argument('--subjects', nargs='+', help='List of subject IDs')
    parser.add_argument('--all-subjects', action='store_true',
                       help='Process all available subjects')
    parser.add_argument('--contrast-base', type=str, required=True,
                       help='Base directory containing contrast maps')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for MD parcel maps')
    parser.add_argument('--parcellation-path', type=str, default=None,
                       help='Path to HCP parcellation dlabel.nii file (auto-detected if not provided)')
    parser.add_argument('--group', action='store_true',
                       help='Compute group-level MD map')
    parser.add_argument('--no-individual-contrasts', action='store_true',
                       help='Do not save individual contrast contributions')
    
    args = parser.parse_args()
    
    # Load parcellation
    parcellation, parcel_names, template_img, cortical_indices = load_hcp_parcellation(args.parcellation_path)
    
    # Determine which subjects to process
    subjects = []
    
    if args.subject:
        subjects = [args.subject]
    elif args.subjects:
        subjects = args.subjects
    else:
        subject_dirs = glob.glob(os.path.join(args.contrast_base, 'sub-*'))
        subjects = sorted([os.path.basename(d).replace('sub-', '') for d in subject_dirs])
        print(f"Found {len(subjects)} subjects: {', '.join(subjects)}")
    
    if not subjects:
        parser.error(f"No subjects found under: {args.contrast_base}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process individual subjects
    for subject in subjects:
        compute_md_map_parcels(subject, args.contrast_base, parcellation, parcel_names,
                              output_dir=args.output,
                              save_individual=not args.no_individual_contrasts,
                              parcellation_template_img=template_img,
                              cortical_indices=cortical_indices)
    
    # Compute group map if requested
    if args.group and len(subjects) > 1:
        compute_group_md_map_parcels(subjects, args.contrast_base, parcellation, 
                                    parcel_names, args.output,
                                    parcellation_template_img=template_img,
                                    cortical_indices=cortical_indices)
    
    print(f"\n{'='*60}")
    print("Parcel-Based MD Mapping Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
