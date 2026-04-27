#!/bin/bash

#SBATCH --job-name=md_mapping
#SBATCH --output=/home/hmueller2/ibc_code/ibc_latent/Multiple-Demand/logs/md_mapping_%j.out
#SBATCH --error=/home/hmueller2/ibc_code/ibc_latent/Multiple-Demand/logs/md_mapping_%j.err
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4


# SLURM script for running MD mapping analysis on IBC dataset
# Submit with: sbatch Multiple-Demand/md_mapping_SLURM.sh

set -euo pipefail

echo "============================================"
echo "Multiple Demand System Mapping (SLURM)"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Set your paths here
CONTRAST_BASE="/ptmp/hmueller2/2025_ibc_latent/outputs/glm/contrast_maps_fsLR"
OUTPUT_DIR="/ptmp/hmueller2/2025_ibc_latent/outputs/md_system_new"
CONFIG_FILE="/ptmp/hmueller2/2025_ibc_latent/misc/subjects_resting.txt"

# Pipeline mode: vertex | parcels | both
# Can be overridden via env, e.g. MODE=both sbatch md_mapping_SLURM.sh
MODE="${MODE:-vertex}"

# Common runtime options
RUN_GROUP="${RUN_GROUP:-1}"                  # 1 = add --group
SAVE_INDIVIDUAL="${SAVE_INDIVIDUAL:-1}"      # 0 = add --no-individual-contrasts

# Vertex-wise options (md_mapping.py)
SMOOTH_FWHM="${SMOOTH_FWHM:-4.0}"
THRESHOLD_PERCENT="${THRESHOLD_PERCENT:-20}"   # Empty string disables percentile threshold
THRESHOLD_Z="${THRESHOLD_Z:-}"                # Optional fallback if percentile is disabled
WB_COMMAND="${WB_COMMAND:-wb_command}"
LEFT_SURFACE="${LEFT_SURFACE:-/home/hmueller2/ibc_code/ibc_latent/MSCcodebase/Utilities/Conte69_atlas-v2.LR.32k_fs_LR.wb/Conte69.L.midthickness.32k_fs_LR.surf.gii}"
RIGHT_SURFACE="${RIGHT_SURFACE:-/home/hmueller2/ibc_code/ibc_latent/MSCcodebase/Utilities/Conte69_atlas-v2.LR.32k_fs_LR.wb/Conte69.R.midthickness.32k_fs_LR.surf.gii}"

# Parcel-wise options (md_mapping_parcels.py)
PARCELLATION_PATH="${PARCELLATION_PATH:-}"    # Optional; empty -> auto-detect in script

# Output subdirectories to avoid collisions when running both
VERTEX_OUTPUT_DIR="${VERTEX_OUTPUT_DIR:-${OUTPUT_DIR}/vertex_wise}"
PARCEL_OUTPUT_DIR="${PARCEL_OUTPUT_DIR:-${OUTPUT_DIR}/parcel_based}"

# Read subjects from config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi
SUBJECTS=$(awk '{print $1}' "$CONFIG_FILE" | tr '\n' ' ')

if [[ -z "${SUBJECTS// }" ]]; then
    echo "ERROR: No subjects found in $CONFIG_FILE"
    exit 1
fi

echo "Contrast base: $CONTRAST_BASE"
echo "Output directory: $OUTPUT_DIR"
echo "Config file: $CONFIG_FILE"
echo "Subjects: $SUBJECTS"
echo "Mode: $MODE"
echo "Run group map: $RUN_GROUP"
echo "Save individual contrasts: $SAVE_INDIVIDUAL"
echo "Vertex smooth FWHM: $SMOOTH_FWHM"
if [[ -n "$THRESHOLD_PERCENT" ]]; then
    echo "Vertex threshold percent: top $THRESHOLD_PERCENT%"
elif [[ -n "$THRESHOLD_Z" ]]; then
    echo "Vertex threshold z: $THRESHOLD_Z"
else
    echo "Vertex threshold: disabled"
fi
echo ""

# Script directory (absolute path)
SCRIPT_DIR="/home/hmueller2/ibc_code/ibc_latent/Multiple-Demand"

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$VERTEX_OUTPUT_DIR"
mkdir -p "$PARCEL_OUTPUT_DIR"
mkdir -p "$SCRIPT_DIR/logs"

# Container setup (using apptainer) - Update with your container path
container=/home/rglz/containers/gfae.sif  # Change to your container path
export APPTAINER_BIND="/run,/ptmp,/tmp,/opt/ohpc,/home/hmueller2"

echo "Script directory: $SCRIPT_DIR"
echo "Working directory: $(pwd)"
echo "Container: $container"
echo ""

run_vertex() {
    local cmd=(python "$SCRIPT_DIR/md_mapping.py" --subjects $SUBJECTS --contrast-base "$CONTRAST_BASE" --output "$VERTEX_OUTPUT_DIR" --smooth "$SMOOTH_FWHM" --wb-command "$WB_COMMAND" --left-surface "$LEFT_SURFACE" --right-surface "$RIGHT_SURFACE")

    if [[ "$RUN_GROUP" == "1" ]]; then
        cmd+=(--group)
    fi
    if [[ "$SAVE_INDIVIDUAL" == "0" ]]; then
        cmd+=(--no-individual-contrasts)
    fi
    if [[ -n "$THRESHOLD_PERCENT" ]]; then
        cmd+=(--threshold-percent "$THRESHOLD_PERCENT")
    elif [[ -n "$THRESHOLD_Z" ]]; then
        cmd+=(--threshold "$THRESHOLD_Z")
    fi

    echo "============================================"
    echo "Running vertex-wise MD mapping"
    echo "============================================"
    printf 'Command: %q ' "${cmd[@]}"
    echo

    srun apptainer exec "${container}" "${cmd[@]}"
}

run_parcels() {
    local cmd=(python "$SCRIPT_DIR/md_mapping_parcels.py" --subjects $SUBJECTS --contrast-base "$CONTRAST_BASE" --output "$PARCEL_OUTPUT_DIR")

    if [[ "$RUN_GROUP" == "1" ]]; then
        cmd+=(--group)
    fi
    if [[ "$SAVE_INDIVIDUAL" == "0" ]]; then
        cmd+=(--no-individual-contrasts)
    fi
    if [[ -n "$PARCELLATION_PATH" ]]; then
        cmd+=(--parcellation-path "$PARCELLATION_PATH")
    fi

    echo "============================================"
    echo "Running parcel-based MD mapping"
    echo "============================================"
    printf 'Command: %q ' "${cmd[@]}"
    echo

    srun apptainer exec "${container}" "${cmd[@]}"
}

EXIT_CODE=0
case "$MODE" in
    vertex)
        run_vertex || EXIT_CODE=$?
        ;;
    parcels)
        run_parcels || EXIT_CODE=$?
        ;;
    both)
        run_vertex || EXIT_CODE=$?
        if [[ $EXIT_CODE -eq 0 ]]; then
            run_parcels || EXIT_CODE=$?
        fi
        ;;
    *)
        echo "ERROR: Unsupported MODE '$MODE'. Use: vertex, parcels, or both"
        EXIT_CODE=2
        ;;
esac

echo ""
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Analysis completed successfully!"
    echo "============================================"
    echo "Results root: $OUTPUT_DIR"
    echo "Vertex outputs: $VERTEX_OUTPUT_DIR"
    echo "Parcel outputs: $PARCEL_OUTPUT_DIR"
    if [[ "$MODE" == "vertex" || "$MODE" == "both" ]]; then
        echo "Workbench geodesic smoothing used (FWHM=$SMOOTH_FWHM)."
    fi
    echo ""
    echo "To visualize with Workbench scene file:"
    echo "  ./md_mapping_view.sh group $VERTEX_OUTPUT_DIR --group"
    echo ""
    echo "To compare subjects:"
    echo "  python visualize_md_maps.py compare $VERTEX_OUTPUT_DIR --output $VERTEX_OUTPUT_DIR/figures"
else
    echo "Analysis failed with exit code: $EXIT_CODE"
    echo "============================================"
fi

echo ""
echo "Job finished at: $(date)"
echo "============================================"

exit $EXIT_CODE
