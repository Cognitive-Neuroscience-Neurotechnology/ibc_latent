"""
Count the number of tasks and contrasts for each subject in the IBC dataset.
Analyzes contrast_maps_fsLR_october directories to find unique tasks and their contrasts.
"""

import os
import glob
import pandas as pd
from pathlib import Path

# Configuration
BASE_DIR = "/ptmp/hmueller2/2025_ibc_latent/outputs/glm/contrast_maps_fsLR_october"
OUTPUT_FILE = "/home/hmueller2/ibc_code/ibc_latent/Data Info/contrast_number.csv"

def count_tasks_and_contrasts(subject_dir):
    """
    Count unique tasks and total contrasts for a single subject.
    
    Parameters:
    -----------
    subject_dir : str
        Path to subject's contrast_maps directory
    
    Returns:
    --------
    dict : {'n_tasks': int, 'n_contrasts': int, 'task_details': dict}
    """
    # Find all task directories (res_task-*_space-fsLR_dir-ffx)
    task_pattern = os.path.join(subject_dir, "res_task-*_space-fsLR_dir-ffx")
    task_dirs = glob.glob(task_pattern)
    
    task_contrast_counts = {}
    total_contrasts = 0
    
    for task_dir in task_dirs:
        # Extract task name
        task_name = os.path.basename(task_dir)
        # Parse task name from pattern: res_task-TASKNAME_space-fsLR_dir-ffx
        task_parts = task_name.split('_')
        for part in task_parts:
            if part.startswith('task-'):
                task_id = part.replace('task-', '')
                break
        else:
            continue  # Skip if no task name found
        
        # Count contrast files in z_score_maps directory
        z_score_dir = os.path.join(task_dir, 'z_score_maps')
        
        if not os.path.exists(z_score_dir):
            task_contrast_counts[task_id] = 0
            continue
        
        # Count .dscalar.nii files (contrast maps)
        contrast_files = glob.glob(os.path.join(z_score_dir, '*.dscalar.nii'))
        n_contrasts = len(contrast_files)
        
        task_contrast_counts[task_id] = n_contrasts
        total_contrasts += n_contrasts
    
    return {
        'n_tasks': len(task_contrast_counts),
        'n_contrasts': total_contrasts,
        'task_details': task_contrast_counts
    }

def main():
    """Main analysis pipeline."""
    print("="*80)
    print("COUNTING TASKS AND CONTRASTS ACROSS SUBJECTS")
    print("="*80)
    
    # Find all subject directories
    subject_pattern = os.path.join(BASE_DIR, "sub-*")
    subject_dirs = sorted(glob.glob(subject_pattern))
    
    # Filter to only directories
    subject_dirs = [d for d in subject_dirs if os.path.isdir(d)]
    
    if not subject_dirs:
        print(f"No subject directories found matching: {subject_pattern}")
        return
    
    print(f"\nFound {len(subject_dirs)} subject directories")
    print("")
    
    # Collect results
    results = []
    task_details_all = {}
    
    for subject_dir in subject_dirs:
        # Extract subject ID
        subject_name = os.path.basename(subject_dir)
        subject_id = subject_name.replace('sub-', '')
        
        print(f"Processing sub-{subject_id}...", end=" ")
        
        # Count tasks and contrasts
        counts = count_tasks_and_contrasts(subject_dir)
        
        results.append({
            'subject': subject_id,
            'n_tasks': counts['n_tasks'],
            'n_contrasts': counts['n_contrasts']
        })
        
        task_details_all[subject_id] = counts['task_details']
        
        print(f"✓ {counts['n_tasks']} tasks, {counts['n_contrasts']} contrasts")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nTotal subjects analyzed: {len(df)}")
    print(f"\nTasks per subject:")
    print(f"  Mean: {df['n_tasks'].mean():.2f}")
    print(f"  Median: {df['n_tasks'].median():.0f}")
    print(f"  Min: {df['n_tasks'].min()}")
    print(f"  Max: {df['n_tasks'].max()}")
    print(f"  Std: {df['n_tasks'].std():.2f}")
    
    print(f"\nContrasts per subject:")
    print(f"  Mean: {df['n_contrasts'].mean():.2f}")
    print(f"  Median: {df['n_contrasts'].median():.0f}")
    print(f"  Min: {df['n_contrasts'].min()}")
    print(f"  Max: {df['n_contrasts'].max()}")
    print(f"  Std: {df['n_contrasts'].std():.2f}")
    
    # Add summary row
    summary_row = pd.DataFrame([{
        'subject': 'MEAN',
        'n_tasks': df['n_tasks'].mean(),
        'n_contrasts': df['n_contrasts'].mean()
    }])
    
    df_with_summary = pd.concat([df, summary_row], ignore_index=True)
    
    # Save to CSV
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_with_summary.to_csv(OUTPUT_FILE, index=False, float_format='%.2f')
    
    print(f"\n✓ Saved results to: {OUTPUT_FILE}")
    
    # Print detailed breakdown for each subject
    print("\n" + "="*80)
    print("DETAILED BREAKDOWN BY SUBJECT")
    print("="*80)
    
    for subject_id, task_details in task_details_all.items():
        print(f"\nsub-{subject_id}:")
        if task_details:
            for task, n_contrasts in sorted(task_details.items()):
                print(f"  {task}: {n_contrasts} contrasts")
        else:
            print("  (no tasks found)")
    
    # Find unique tasks across all subjects
    all_tasks = set()
    for task_details in task_details_all.values():
        all_tasks.update(task_details.keys())
    
    print("\n" + "="*80)
    print(f"UNIQUE TASKS ACROSS ALL SUBJECTS: {len(all_tasks)}")
    print("="*80)
    for task in sorted(all_tasks):
        # Count how many subjects have this task
        n_subjects_with_task = sum(1 for td in task_details_all.values() if task in td)
        avg_contrasts = sum(td.get(task, 0) for td in task_details_all.values()) / n_subjects_with_task
        print(f"  {task}: {n_subjects_with_task} subjects, avg {avg_contrasts:.1f} contrasts")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()