#!/bin/bash -l

#SBATCH --job-name=compare_kmeans
#SBATCH --output=/ptmp/hmueller2/kmeans_compare_logs/output/%A_%x_%a_%u.out
#SBATCH --error=/ptmp/hmueller2/kmeans_compare_logs/errors/%A_%x_%a_%u.err
#SBATCH --partition=thin
#SBATCH --exclusive=user
#SBATCH --array=0-7
#SBATCH --time=1:00:00 
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

# ----------

container=/home/rglz/containers/gfae.sif
working_dir=/ptmp/hmueller2/Downloads

# Read subject from config file
config_file=/ptmp/hmueller2/Downloads/subjects_resting.txt
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$config_file")
subject=$(echo "$line" | awk '{print $1}')

echo "Processing subject: sub-${subject}"

# Set k value (default k=2, can be overridden)
k=${1:-2}
echo "Comparing k=${k} clusters"

export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2"
srun apptainer exec ${container} python /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Derivation/compare_approaches.py --subject ${subject} --k ${k}

exit 0

# Usage:
# sbatch /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Derivation/compare_approaches_SLURM.sh
# Or with specific k:
# sbatch /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Derivation/compare_approaches_SLURM.sh 3