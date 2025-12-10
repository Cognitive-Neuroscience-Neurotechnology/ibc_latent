#!/bin/bash -l

#SBATCH --job-name=atlas
#SBATCH --output=/ptmp/hmueller2/GLM_subnet_logs/output/%j_%x_%u.out
#SBATCH --error=/ptmp/hmueller2/GLM_subnet_logs/errors/%j_%x_%u.err
#SBATCH --partition=compute
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

container=/home/rglz/containers/gfae.sif

export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2"

analysis="activation" # "activation" or "ppi"


echo "=========================================="
echo "Running cognitive atlas mapping"
echo "=========================================="

if [ "$analysis" = "ppi" ]; then
    srun apptainer exec ${container} python \
        /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/cognitive_atlas_mapping_ppi.py
else
    srun apptainer exec ${container} python \
        /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/cognitive_atlas_mapping.py
fi

if [ $? -eq 0 ]; then
    echo "✓ Cognitive atlas mapping complete!"
else
    echo "ERROR: Mapping failed"
    exit 1
fi

# Run with: sbatch /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/cognitive_atlas_mapping_SLURM.sh
