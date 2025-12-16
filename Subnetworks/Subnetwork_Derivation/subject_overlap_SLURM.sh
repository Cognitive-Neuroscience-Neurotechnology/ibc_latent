#!/bin/bash -l

#SBATCH --job-name=subject_overlap
#SBATCH --output=/ptmp/hmueller2/kmeans_compare_logs/output/%j_%x_%u.out
#SBATCH --error=/ptmp/hmueller2/kmeans_compare_logs/errors/%j_%x_%u.err
#SBATCH --partition=compute
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

container=/home/rglz/containers/gfae.sif

export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2"

echo "=========================================="
echo "Running between-subject overlap analysis"
echo "=========================================="

srun apptainer exec ${container} python \
    /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Derivation/subject_overlap.py

if [ $? -eq 0 ]; then
    echo "✓ Between-subject overlap analysis complete!"
else
    echo "ERROR: Overlap analysis failed"
    exit 1
fi

# Run with: sbatch /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Derivation/subject_overlap_SLURM.sh