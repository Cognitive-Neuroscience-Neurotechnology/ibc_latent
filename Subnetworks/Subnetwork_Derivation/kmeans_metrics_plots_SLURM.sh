#!/bin/bash -l

#SBATCH --job-name=kmeans_metrics_aggregate
#SBATCH --output=/ptmp/hmueller2/kmeans_comms_logs/output/%A_%x_%u.out
#SBATCH --error=/ptmp/hmueller2/kmeans_comms_logs/errors/%A_%x_%u.err
#SBATCH --partition=thin
#SBATCH --time=0:30:00 
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

container=/home/rglz/containers/gfae.sif
working_dir=/ptmp/hmueller2/Downloads

echo "Aggregating k-means metrics across all subjects"

export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2"
srun apptainer exec ${container} python /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Derivation/kmeans_metrics_plots.py

echo "Completed aggregation"

exit 0

# run with: sbatch /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Derivation/kmeans_metrics_plots_SLURM.sh