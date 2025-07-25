#!/bin/bash

#SBATCH --job-name=ibc_preproc
#SBATCH --output=/ptmp/hmueller2/pipeline_logs/output/ibc_preproc_%A_%a.out
#SBATCH --error=/ptmp/hmueller2/pipeline_logs/errors/ibc_preproc_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --array=0-12   # 13 subjects, index 0 to 12

SUBJECTS_FILE=/ptmp/hmueller2/Downloads/subjects.txt
CONTAINER=/home/rglz/containers/gfae.sif

subject=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $SUBJECTS_FILE)

sessions=($(ls -d /ptmp/hmueller2/Downloads/fmriprep_out/sub-${subject}/ses-* 2>/dev/null | xargs -n1 basename | sed 's/^ses-//'))
directions=("ap" "pa")

echo "Starting processing for subject $subject with ${#sessions[@]} sessions."
echo "Sessions: (${sessions[@]})"
echo "Tasks: (${tasks[@]})"

for sess in "${sessions[@]}"; do
  if [ -z "$sess" ]; then
    echo "No session found for subject $subject. Skipping."
    continue
  fi

  # Get tasks for this session
  task_files=(/ptmp/hmueller2/Downloads/fmriprep_out/sub-${subject}/ses-${sess}/func/*_task-*.nii.gz)
  if [ ${#task_files[@]} -eq 0 ]; then
    echo "No tasks found for subject $subject session $sess. Skipping."
    continue
  fi
  tasks=($(basename -a "${task_files[@]}" | awk -F'task-' '{print $2}' | awk -F'_' '{print $1}' | sort | uniq))
  echo "Tasks for subject $subject session $sess: (${tasks[@]})"

  for task in "${tasks[@]}"; do
    if [ -z "$task" ]; then
      echo "No task found for subject $subject session $sess. Skipping."
      continue
    fi
    for direction in "${directions[@]}"; do
      echo "---- subject=$subject session=$sess task=$task direction=$direction ----"
      bold_file="/ptmp/hmueller2/Downloads/fmriprep_out/sub-${subject}/ses-${sess}/func/sub-${subject}_ses-${sess}_task-${task}_dir-${direction}_space-fsLR_den-91k_bold.dtseries.nii"
      if [ ! -f "$bold_file" ]; then
        echo "WARNING: BOLD file not found: $bold_file"
      fi
      singularity exec --home /home/hmueller2 \
        $CONTAINER \
        bash /home/hmueller2/ibc_code/ibc_latent/Preprocessing/pipeline_preprocessing.sh "$subject" "$sess" "$task" "$direction"
      status=$?
      if [ $status -ne 0 ]; then
        echo "FAILED: subject $subject session $sess task $task direction $direction (exit code $status)"
      else
        echo "SUCCESS: subject $subject session $sess task $task direction $direction"
      fi
    done
  done
done