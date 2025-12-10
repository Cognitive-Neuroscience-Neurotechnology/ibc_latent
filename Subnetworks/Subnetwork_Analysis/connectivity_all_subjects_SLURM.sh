#!/bin/bash -l

#SBATCH --job-name=connectivity_analysis
#SBATCH --output=/ptmp/hmueller2/subnetworks_connectivity_logs/output/%A_%x_%a_%u.out
#SBATCH --error=/ptmp/hmueller2/subnetworks_connectivity_logs/errors/%A_%x_%a_%u.err
#SBATCH --partition=thin
#SBATCH --exclusive=user
#SBATCH --time=0:30:00
#SBATCH --mail-type=FAIL,TIME_LIMIT


container=/home/rglz/containers/gfae.sif
working_dir=/ptmp/hmueller2/Downloads


export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2"

# Run connectivity analysis across subjects for the specified approach
srun apptainer exec ${container} python /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/connectivity_all_subjects.py

echo "Completed connectivity analysis across subjects."

exit 0

# run with: sbatch /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/connectivity_all_subjects_SLURM.sh