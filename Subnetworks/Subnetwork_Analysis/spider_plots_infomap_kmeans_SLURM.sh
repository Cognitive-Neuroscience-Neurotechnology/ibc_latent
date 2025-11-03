#!/bin/bash -l

#SBATCH --job-name=infomap_subnetworks
#SBATCH --output=/ptmp/hmueller2/spider_infomap_logs/output/%A_%x_%a_%u.out
#SBATCH --error=/ptmp/hmueller2/spider_infomap_logs/errors/%A_%x_%a_%u.err
#SBATCH --partition=compute
#SBATCH --exclusive=user
#SBATCH --array=0-7   # Adjust to number of subjects in subjects_resting.txt
#SBATCH --time=2:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT

container=/home/rglz/containers/gfae.sif
working_dir=/ptmp/hmueller2/Downloads

# Read subject from config file
config_file=/ptmp/hmueller2/Downloads/subjects_resting.txt
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$config_file")
subject=$(echo "$line" | awk '{print $1}')

echo "Processing subject: sub-${subject}"

export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2"

srun apptainer exec ${container} python /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/spider_plots_infomap_kmeans.py --subject ${subject}

exit 0