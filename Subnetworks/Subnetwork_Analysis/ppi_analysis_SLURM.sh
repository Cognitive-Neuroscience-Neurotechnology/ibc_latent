#!/bin/bash -l

#SBATCH --job-name=ppi_filtered
#SBATCH --output=/ptmp/hmueller2/ppi_analysis_logs/output/%A_%x_%a_%u.out
#SBATCH --error=/ptmp/hmueller2/ppi_analysis_logs/errors/%A_%x_%a_%u.err
#SBATCH --partition=compute
#SBATCH --exclusive=user
#SBATCH --array=0-7   # Adjust to number of subjects -1
#SBATCH --time=6:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT

container=/home/rglz/containers/gfae.sif
working_dir=/ptmp/hmueller2/Downloads

# Read subject from config file
config_file=/ptmp/hmueller2/Downloads/subjects_resting.txt
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$config_file")
subject=$(echo "$line" | awk '{print $1}')
#subject="15"

echo "=========================================="
echo "Processing subject: sub-${subject}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="

export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2"

# Run PPI analysis
echo "Running PPI analysis for sub-${subject}..."
srun apptainer exec ${container} python /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/ppi_analysis_DMN_DAN_filtered.py ${subject}

if [ $? -ne 0 ]; then
    echo "ERROR: PPI analysis failed for sub-${subject}"
    exit 1
fi

echo "=========================================="
echo "✓ PPI analysis complete for sub-${subject}!"
echo "=========================================="

exit 0

# run with: sbatch /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/ppi_analysis_SLURM.sh