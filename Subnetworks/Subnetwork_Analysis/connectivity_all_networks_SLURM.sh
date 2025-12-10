#!/bin/bash -l

#SBATCH --job-name=connectivity_analysis
#SBATCH --output=/ptmp/hmueller2/subnetworks_connectivity_logs/output/%A_%x_%a_%u.out
#SBATCH --error=/ptmp/hmueller2/subnetworks_connectivity_logs/errors/%A_%x_%a_%u.err
#SBATCH --partition=thin
#SBATCH --exclusive=user
#SBATCH --array=0-7   # Adjust to number of subjects in subjects_resting.txt
#SBATCH --time=0:30:00
#SBATCH --mail-type=FAIL,TIME_LIMIT

container=/home/rglz/containers/gfae.sif
working_dir=/ptmp/hmueller2/Downloads

approach=kmeans # 'infomap' or 'kmeans'

# Create log directories if they don't exist
mkdir -p /ptmp/hmueller2/subnetworks_connectivity_logs/output
mkdir -p /ptmp/hmueller2/subnetworks_connectivity_logs/errors

# Read subject from config file
config_file=/ptmp/hmueller2/Downloads/subjects_resting.txt
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$config_file")
subject=$(echo "$line" | awk '{print $1}')

echo "Processing subject: sub-${subject}"
echo "Approach: ${approach}"

export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2"

# Run connectivity analysis for the specified approach
srun apptainer exec ${container} python /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/connectivity_all_networks.py --subject ${subject} --approach ${approach}

echo "Completed connectivity analysis for sub-${subject} using approach: ${approach}"

exit 0

# run with: sbatch /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/connectivity_all_networks_SLURM.sh