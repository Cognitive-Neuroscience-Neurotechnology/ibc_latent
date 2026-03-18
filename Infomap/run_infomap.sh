#!/bin/bash

#SBATCH --job-name=infomap
#SBATCH --output=/ptmp/hmueller2/2025_ibc_latent/logs/output/%A_%x_%a.out
#SBATCH --error=/ptmp/hmueller2/2025_ibc_latent/logs/errors/%A_%x_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --partition=compute
#SBATCH --array=0-2    # 8 subjects, index 0 to 7
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT


SUBJECTS_FILE=/ptmp/hmueller2/Downloads/subjects_again.txt
Subject=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $SUBJECTS_FILE)

Type="resting"

echo "===================================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Subject: $Subject"
echo "Type: $Type"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $((SLURM_MEM_PER_CPU * SLURM_CPUS_PER_TASK))GB"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "===================================================================="

# Set environment variables
export LD_LIBRARY_PATH=/home/hmueller2/workbench/libs_linux64:/home/hmueller2/workbench/libs_linux64_software_opengl:$LD_LIBRARY_PATH
export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2"
export Subject
export Type

# Verify required files/directories exist
REQUIRED_DIRS=(
    "/home/hmueller2/workbench"
    "/home/hmueller2/infomap-2.8.0"
    "/home/hmueller2/ibc_code/ibc_latent/Infomap"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [[ ! -d "$dir" ]]; then
        echo "ERROR: Required directory not found: $dir"
        exit 1
    fi
done

# Verify container exists
CONTAINER="/ptmp/containers/matlab-romy-r2024b-2024-11-08-1b59d97e0135.sif"
if [[ ! -f "$CONTAINER" ]]; then
    echo "ERROR: Container not found: $CONTAINER"
    exit 1
fi

# Clean up any stale MATLAB parallel pool jobs before starting
echo "Cleaning up stale MATLAB parallel pool cache..."
rm -rf ~/.matlab/local_cluster_jobs/R2024b/* 2>/dev/null || true

# Run Infomap pipeline
echo "Starting Infomap pipeline..."
START_TIME=$(date +%s)

apptainer exec \
    --cleanenv \
    --bind /home/hmueller2/workbench:/mnt/workbench \
    --bind /home/hmueller2/workbench/run_wb_command.sh:/mnt/workbench/run_wb_command.sh \
    --bind /home/hmueller2/infomap-2.8.0/infomap:/usr/local/bin/infomap \
    "$CONTAINER" \
    bash -c "
        export LD_LIBRARY_PATH=/mnt/workbench/libs_linux64:/mnt/workbench/libs_linux64_software_opengl:\$LD_LIBRARY_PATH
        export Subject='$Subject'
        export Type='$Type'
        matlab -nodisplay -nosplash -nodesktop -batch \"addpath('/home/hmueller2/ibc_code/ibc_latent/Infomap'); pfm_resting\"
    "

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "===================================================================="
echo "End time: $(date)"
echo "Elapsed time: $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds"
echo "Exit code: $EXIT_CODE"
echo "===================================================================="

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "SUCCESS: Subject $Subject completed successfully"
else
    echo "FAILED: Subject $Subject exited with code $EXIT_CODE"
fi

exit $EXIT_CODE

# run with: sbatch Infomap/run_infomap.sh