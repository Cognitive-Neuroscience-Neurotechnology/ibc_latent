#!/bin/bash -l

#SBATCH --job-name=GLM_subnet
#SBATCH --output=/ptmp/hmueller2/GLM_subnet_logs/output/%A_%x_%a_%u.out
#SBATCH --error=/ptmp/hmueller2/GLM_subnet_logs/errors/%A_%x_%a_%u.err
#SBATCH --partition=compute
#SBATCH --exclusive=user
#SBATCH --array=0-7   # Adjust to number of subjects in subjects_resting.txt
#SBATCH --time=6:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT

container=/home/rglz/containers/gfae.sif
working_dir=/ptmp/hmueller2/Downloads

# Read subject from config file
config_file=/ptmp/hmueller2/Downloads/subjects_resting.txt
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$config_file")
subject=$(echo "$line" | awk '{print $1}')

echo "=========================================="
echo "Processing subject: sub-${subject}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="

export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2"

# Step 1: Run per-subject contrast x subnetwork analysis
echo "[Step 1/3] Running contrast_x_subnetwork.py for sub-${subject}..."
srun apptainer exec ${container} python /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/contrast_x_subnetwork.py ${subject}

if [ $? -ne 0 ]; then
    echo "ERROR: contrast_x_subnetwork.py failed for sub-${subject}"
    exit 1
fi

echo "✓ Step 1 complete for sub-${subject}"
echo ""

# Steps 2 & 3: Aggregate across subjects and plot (only run once by the first task)
# Check if this is an array job, otherwise always run aggregation
if [ -z "${SLURM_ARRAY_TASK_ID}" ] || [ ${SLURM_ARRAY_TASK_ID} -eq 0 ]; then
    
    # Wait for all other array tasks to complete their Step 1
    if [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
        echo "[Step 2/3] Waiting for all subjects to complete Step 1..."
        
        # Get the job ID of the current array job
        job_id=${SLURM_ARRAY_JOB_ID}
        
        # Poll until all array tasks are done
        max_wait=7200  # 2 hours max wait
        elapsed=0
        while [ $elapsed -lt $max_wait ]; do
            # Check if any tasks are still running
            running=$(squeue -j ${job_id} -h -t RUNNING,PENDING | wc -l)
            if [ $running -le 1 ]; then  # Only this task is running
                break
            fi
            echo "  Still waiting... ($running tasks remaining)"
            sleep 30
            elapsed=$((elapsed + 30))
        done
        
        if [ $elapsed -ge $max_wait ]; then
            echo "WARNING: Timeout waiting for other tasks. Attempting aggregation anyway..."
        fi
    fi
    
    echo "[Step 2/3] Running contrast_x_subnetwork_aggregate_subjects.py..."
    srun apptainer exec ${container} python /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/contrast_x_subnetwork_aggregate_subjects.py
    
    if [ $? -ne 0 ]; then
        echo "ERROR: contrast_x_subnetwork_aggregate_subjects.py failed"
        exit 1
    fi
    
    echo "✓ Step 2 complete"
    echo ""
    
    # Step 3: Create plots
    echo "[Step 3/3] Running contrast_x_subnetwork_plots.py..."
    srun apptainer exec ${container} python /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/contrast_x_subnetwork_plots.py
    
    if [ $? -ne 0 ]; then
        echo "ERROR: contrast_x_subnetwork_plots.py failed"
        exit 1
    fi
    
    echo "✓ Step 3 complete"
    echo ""
    echo "=========================================="
    echo "✓ ALL STEPS COMPLETE!"
    echo "=========================================="
else
    echo "Task ${SLURM_ARRAY_TASK_ID}: Step 1 complete. Steps 2-3 will be run by task 0."
fi

exit 0

# run with: sbatch /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/contrast_x_subnetwork_SLURM.sh