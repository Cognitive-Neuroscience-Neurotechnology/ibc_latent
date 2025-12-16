#!/bin/bash -l

#SBATCH --job-name=plot_wb
#SBATCH --output=/ptmp/hmueller2/plotting_brain_logs/output/%A_%x_%a_%u.out
#SBATCH --error=/ptmp/hmueller2/plotting_brain_logs/errors/%A_%x_%a_%u.err
#SBATCH --partition=thin
#SBATCH --exclusive=user
# #SBATCH --array=0-7   # Adjust to number of subjects in subjects_resting.txt
#SBATCH --time=2:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT

container=/home/rglz/containers/gfae.sif
working_dir=/ptmp/hmueller2/Downloads

script=/home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/workbench_plots_dlabel.sh
chmod +x "${script}" || true

# Read subject from config file
config_file=/ptmp/hmueller2/Downloads/subjects_resting.txt
#line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$config_file")
#subject=$(echo "$line" | awk '{print $1}')
subject="13"

echo "Processing subject: sub-${subject}"

export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2"

srun apptainer exec "${container}" bash -lc "${script} ${subject}"
exit 0

# run with: sbatch /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/workbench_plots_SLURM.sh