#!/bin/bash -l

#SBATCH --job-name=ppi_contrasts
#SBATCH --output=/ptmp/hmueller2/ppi_analysis_logs/output/%A_%x_%u.out
#SBATCH --error=/ptmp/hmueller2/ppi_analysis_logs/errors/%A_%x_%u.err
#SBATCH --partition=thin
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

container=/home/rglz/containers/gfae.sif
config_file=/ptmp/hmueller2/Downloads/subjects_resting.txt

export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2"

echo "=========================================="
echo "Running across-subjects PPI analysis..."
echo "=========================================="

# Read subjects from config file and pass as arguments
subjects=$(awk '{print $1}' "$config_file" | tr '\n' ' ')

echo "Analyzing subjects: $subjects"

# Run with subjects as arguments
#srun apptainer exec ${container} python \
    #/home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/ppi_analysis_DMN_DAN_across_subjects.py \
    #$subjects
srun apptainer exec ${container} python /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/ppi_build_contrasts_from_conditions.py


if [ $? -ne 0 ]; then
    echo "ERROR: Across-subjects analysis failed"
    exit 1
fi

echo "=========================================="
echo "✓ Across-subjects analysis complete!"
echo "=========================================="

exit 0

# Run with: sbatch /home/hmueller2/ibc_code/ibc_latent/Subnetworks/Subnetwork_Analysis/ppi_analysis_DMN_DAN_across_subjects_SLURM.sh