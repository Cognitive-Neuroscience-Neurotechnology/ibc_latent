#!/bin/bash -l
 
#SBATCH --job-name=subn_infomap
#SBATCH --output=/ptmp/hmueller2/subnetworks_infomap_logs/output/%A_%x.out
#SBATCH --error=/ptmp/hmueller2/subnetworks_infomap_logs/errors/%A_%x.err
#SBATCH --exclusive=user
#SBATCH --cpus-per-task=8
#SBATCH --array=0-7   # 8 subjects, index 0 to 7
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

# ----------

# aradia_container=/home/rglz/containers/gfae.sif
container=/home/rglz/containers/gfae.sif

working_dir=/ptmp/hmueller2/Downloads

# Read subject, session and run from config file
config_file=/ptmp/hmueller2/Downloads/subjects_resting.txt
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$config_file")
subject=$(echo "$line" | awk '{print $1}')

echo "Processing subject: sub-${subject}"

# Then use it to run bash out of the singularity container
export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2"
srun apptainer exec ${container} python /home/hmueller2/ibc_code/ibc_latent/Subnetworks/get_FPN_communities.py --subject ${subject} --dir ${working_dir}

# Finish the script
# echo "Script completed successfully."
exit 0

