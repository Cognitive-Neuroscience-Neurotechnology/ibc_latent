#!/bin/bash -l

#SBATCH --job-name=kmeans_com
#SBATCH --output=/ptmp/hmueller2/kmeans_comms_logs/output/%A_%x.out
#SBATCH --error=/ptmp/hmueller2/kmeans_comms_logs/errors/%A_%x.err
#SBATCH --partition=thin
#SBATCH --exclusive=user
#SBATCH --array=0-7
#SBATCH --time=24:00:00 
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
# #SBATCH --mail-type=END,FAIL,TIME_LIMIT

# ----------

container=/home/rglz/containers/gfae.sif
working_dir=/ptmp/hmueller2/Downloads

# For all subjects: Read subject from config file (same logic as get_FPN_comms_SLURM.sh)
config_file=/ptmp/hmueller2/Downloads/subjects_resting.txt
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$config_file")
subject=$(echo "$line" | awk '{print $1}')

# Single-subject override (was previously read via SLURM_ARRAY_TASK_ID and sed)
# subject=06
echo "Processing subject: sub-${subject}"

export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2"
srun apptainer exec ${container} python /home/hmueller2/ibc_code/ibc_latent/Subnetworks/kmeans_on_communities.py --subject ${subject} --dir ${working_dir}

exit 0

