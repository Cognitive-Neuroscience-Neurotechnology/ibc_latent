#!/bin/bash -l

#SBATCH --job-name=FPN_subnet_new
#SBATCH --output=/ptmp/hmueller2/GLM_subnet_logs/output/%A_%x_%a_%u.out
#SBATCH --error=/ptmp/hmueller2/GLM_subnet_logs/errors/%A_%x_%a_%u.err
#SBATCH --partition=compute
#SBATCH --exclusive=user
#SBATCH --array=0-7   # Adjust to number of subjects in subjects_resting.txt
#SBATCH --time=6:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT

container=/home/rglz/containers/gfae.sif

# Read subject from config file
config_file=/ptmp/hmueller2/Downloads/subjects_resting.txt
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$config_file")
subject=$(echo "$line" | awk '{print $1}')

echo "=========================================="
echo "Processing subject: sub-${subject}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="

export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2"

# Run per-subject analysis with FDR correction
echo "Running contrast_x_subn_new.py for sub-${subject}..."
srun apptainer exec ${container} python \
    /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/contrast_x_subn_new.py \
    ${subject}

if [ $? -ne 0 ]; then
    echo "ERROR: contrast_x_subn_new.py failed for sub-${subject}"
    exit 1
fi

echo "✓ Analysis complete for sub-${subject}"
exit 0

# Run with: sbatch /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/contrast_x_subn_new_SLURM.sh

####### After this script run -> aggregate_subjects_SLURM.sh #######
