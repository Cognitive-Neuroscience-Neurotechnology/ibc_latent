#!/bin/bash

#SBATCH --job-name=infomap_RS
#SBATCH --output=/ptmp/hmueller2/infomap_logs/output/infomap_RS_%A_%a.out
#SBATCH --error=/ptmp/hmueller2/infomap_logs/errors/infomap_RS_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=4G
# SBATCH --mail-type=END,FAIL,TIME_LIMIT

# in case of multipe subjects: 
# SBATCH --array=0-12   # 13 subjects, index 0 to 12
# SUBJECTS_FILE=/ptmp/hmueller2/Downloads/subjects.txt
# CONTAINER=/home/rglz/containers/gfae.sif
# subject=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $SUBJECTS_FILE)
subject="01"
type="resting" # or "concatenated tasks"


echo "Starting processing for subject $subject and type $type."

apptainer exec /ptmp/containers/matlab-r2024b.sif matlab -nodisplay -nosplash -r "cd('/home/hmueller2/ibc_code/ibc_latent/Infomap'); pfm_test_resting; exit"