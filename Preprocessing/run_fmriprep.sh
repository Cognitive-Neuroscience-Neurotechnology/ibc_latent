#!/bin/bash

#SBATCH --job-name=fmriprep
#SBATCH --output=/ptmp/hmueller2/fmriprep_logs/fmriprep_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-150:00:00
#SBATCH --array=0-11
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

# Define paths
CONTAINER=/ptmp/hmueller2/Downloads/fmriprep_23.2.1.sif
BIDS_DIR=/ptmp/hmueller2/Downloads/ibc_raw
OUTPUT_DIR=/ptmp/hmueller2/Downloads/fmriprep_out
WORK_DIR=/ptmp/hmueller2/Downloads/fmriprep_work
FS_LICENSE=/ptmp/hmueller2/Downloads/license.txt
TF_CACHE="/ptmp/hmueller2/templateflow"
CONFIG_FILE=/ptmp/hmueller2/Downloads/config.txt

# Fix unexpanded ${HOME} in APPTAINER_BIND
unset APPTAINER_BIND
export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,${HOME}"

# Read the subject ID from the config file
SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $CONFIG_FILE | awk '{print $1}')
echo "Start fMRIPrep for $SUBJECT (SLURM_TASK_ID: $SLURM_ARRAY_TASK_ID)"

# Run fMRIPrep inside the container
apptainer exec \
  --bind "$BIDS_DIR:/data" \
  --bind "$OUTPUT_DIR:/out" \
  --bind "$WORK_DIR:/work" \
  --bind "$FS_LICENSE:/opt/freesurfer/license.txt" \
  --bind "$TF_CACHE:$TF_CACHE" \
  "$CONTAINER" \
  bash -c "export TEMPLATEFLOW_HOME=$TF_CACHE && \
    fmriprep /data /out participant \
      --participant-label $SUBJECT \
      --fs-license-file /opt/freesurfer/license.txt \
      --output-spaces fsaverage fsLR \
      --cifti-output \
      --use-syn-sdc \
      --nthreads 8 \
      --omp-nthreads 8 \
      --mem_mb 32000 \
      --work-dir /work"
