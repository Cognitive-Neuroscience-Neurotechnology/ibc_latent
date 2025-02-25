import os
from task_contrasts import task_contrasts
from config_RSA import base_dir, code_dir

def count_task_contrasts(subject):
    subject_dir = os.path.join(base_dir, f'sub-{subject}')
    session_dirs = [d for d in os.listdir(subject_dir) if d.startswith('ses-')]
    
    contrast_count = 0
    unique_tasks = set()
    
    for hemisphere in ['lh', 'rh']:
        for task, contrasts in task_contrasts.items():
            for contrast in contrasts:
                file_paths = [os.path.join(subject_dir, session, f'sub-{subject}_ses-{session.split("-")[1]}_task-{task}_dir-ffx_space-fsaverage7_hemi-{hemisphere}_ZMap-{contrast}.gii') for session in session_dirs]
                
                # Check if files exist
                file_paths = [fp for fp in file_paths if os.path.exists(fp)]
                if file_paths:
                    contrast_count += 1
                    unique_tasks.add(task)
    
    return contrast_count, len(unique_tasks)

def main():
    subjects = [d.split('-')[1] for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('sub-')]
    
    subject_contrast_counts = {}
    
    for subject in subjects:
        contrast_count, unique_task_count = count_task_contrasts(subject)
        subject_contrast_counts[subject] = (contrast_count, unique_task_count)
        print(f"Subject {subject} has {contrast_count} task contrasts across {unique_task_count} unique tasks.")
    
    # Save the results to a CSV file
    output_file = os.path.join(code_dir, 'subject_contrast_counts.csv')
    with open(output_file, 'w') as f:
        f.write('subject,contrast_count,unique_task_count\n')
        for subject, (contrast_count, unique_task_count) in subject_contrast_counts.items():
            f.write(f'{subject},{contrast_count},{unique_task_count}\n')
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()