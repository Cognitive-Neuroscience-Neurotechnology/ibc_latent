#!/bin/bash
#SBATCH --job-name=ibc_preproc
#SBATCH --output=logs/ibc_preproc_%j.out
#SBATCH --error=logs/ibc_preproc_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

subjects=($(ls /home/hmueller2/ibc_data/sub-* | awk -F'-' '{print $NF}'))
sessions=($(ls /home/hmueller2/ibc_data/sub-${subjects[0]}/ses-* | awk -F'-' '{print $NF}'))
tasks=($(ls /home/hmueller2/ibc_data/sub-${subjects[0]}/ses-${sessions[0]}/func/*_task-*.nii.gz | awk -F'task-' '{print $2}' | awk -F'_' '{print $1}' | sort | uniq))
directions=("ap" "pa")

for subj in "${subjects[@]}"; do
  for sess in "${sessions[@]}"; do
    for task in "${tasks[@]}"; do
      for direction in "${directions[@]}"; do
        echo "Running subject $subj session $sess task $task direction $direction"
        singularity exec --home /home/hmueller2 \
          /home/rglz/containers/gfae.sif \
          bash /home/hmueller2/ibc_code/ibc_latent/Preprocessing/pipeline_preprocessing.sh "$subj" "$sess" "$task" "$direction"
      done
    done
  done
done