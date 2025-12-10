#!/bin/bash -l

#SBATCH --job-name=FPN_aggregate
#SBATCH --output=/ptmp/hmueller2/GLM_subnet_logs/output/%j_%x_%u.out
#SBATCH --error=/ptmp/hmueller2/GLM_subnet_logs/errors/%j_%x_%u.err
#SBATCH --partition=compute
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

container=/home/rglz/containers/gfae.sif

export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2"

echo "=========================================="
echo "Running group-level aggregation"
echo "=========================================="

srun apptainer exec ${container} python \
    #/home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/aggregate_subjects.py

if [ $? -eq 0 ]; then
    echo "✓ Group-level aggregation complete!"
else
    echo "ERROR: Aggregation failed"
    exit 1
fi

# Run with: sbatch /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/aggregate_subjects_SLURM.sh
