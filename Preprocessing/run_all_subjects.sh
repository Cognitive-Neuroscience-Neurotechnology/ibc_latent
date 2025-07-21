#!/bin/bash

subjects=($(ls /home/hmueller2/ibc_data/sub-* | awk -F'-' '{print $NF}'))
sessions=($(ls /home/hmueller2/ibc_data/sub-${subjects[0]}/ses-* | awk -F'-' '{print $NF}'))
tasks=($(ls /home/hmueller2/ibc_data/sub-${subjects[0]}/ses-${sessions[0]}/func/*_task-*.nii.gz | awk -F'task-' '{print $2}' | awk -F'_' '{print $1}' | sort | uniq))
directions=("ap" "pa")

for subj in "${subjects[@]}"; do
  for sess in "${sessions[@]}"; do
    echo "Running subject $subj session $sess"
    singularity exec --home /home/hmueller2 \
      /home/rglz/containers/gfae.sif \
      bash /home/hmueller2/ibc_code/ibc_latent/Preprocessing/pipeline_preprocessing.sh "$subj" "$sess" "$task" "$direction"
  done
done