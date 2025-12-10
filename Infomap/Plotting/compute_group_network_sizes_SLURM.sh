#!/bin/bash
#SBATCH --job-name=group_net_sizes
#SBATCH --output=/ptmp/hmueller2/infomap_logs/output/%j_%x.out
#SBATCH --error=/ptmp/hmueller2/infomap_logs/errors/%j_%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

# ----------------------------
# Load all subjects from text file
# ----------------------------
SUBJECTS_FILE=/ptmp/hmueller2/Downloads/subjects_resting.txt
echo "---- Running group network sizes for all subjects ----"

# ----------------------------
# Apptainer + MATLAB setup
# ----------------------------
export LD_LIBRARY_PATH=/home/hmueller2/workbench/libs_linux64:/home/hmueller2/workbench/libs_linux64_software_opengl:$LD_LIBRARY_PATH
export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2"

# ----------------------------
# Run MATLAB script with subjects file path
# ----------------------------
apptainer exec \
    --bind /home/hmueller2/workbench:/mnt/workbench \
    --bind /home/hmueller2/workbench/run_wb_command.sh:/mnt/workbench/run_wb_command.sh \
    /ptmp/containers/matlab-romy-r2024b-2024-11-08-1b59d97e0135.sif \
    bash -c "export LD_LIBRARY_PATH=/mnt/workbench/libs_linux64:/mnt/workbench/libs_linux64_software_opengl:\$LD_LIBRARY_PATH; \
             matlab -nodisplay -nosplash -r \
             \"addpath(genpath('/home/hmueller2/ibc_code/ibc_latent')); \
               compute_group_network_sizes('$SUBJECTS_FILE'); \
               exit\""
        
# run with: sbatch /home/hmueller2/ibc_code/ibc_latent/Infomap/Plotting/compute_group_network_sizes_SLURM.sh 