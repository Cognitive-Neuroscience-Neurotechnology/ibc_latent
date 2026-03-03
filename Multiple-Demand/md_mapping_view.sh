#!/bin/bash
# Visualization script for MD maps created by md_mapping.py
# Default behavior: generate spec files for all subjects
#
# Usage (recommended):
#   ./md_mapping_view.sh <MD_OUTPUT_DIR> [--group] [--fmriprep-dir /path]
#   ./md_mapping_view.sh <MD_OUTPUT_DIR> --subject 04 [--group] [--fmriprep-dir /path]
#
# Backward-compatible usage:
#   ./md_mapping_view.sh 04 <MD_OUTPUT_DIR> [--group] [--fmriprep-dir /path]

set -euo pipefail

SHOW_GROUP=false
FMRIPREP_DIR=""
SUB=""
MD_OUTPUT_DIR=""

print_usage() {
    echo "Usage:"
    echo "  $0 <MD_OUTPUT_DIR> [--group] [--fmriprep-dir /path]"
    echo "  $0 <MD_OUTPUT_DIR> --subject <SUBJECT_ID> [--group] [--fmriprep-dir /path]"
    echo "  $0 <SUBJECT_ID> <MD_OUTPUT_DIR> [--group] [--fmriprep-dir /path]   (legacy)"
}

if [[ $# -lt 1 ]]; then
    print_usage
    exit 1
fi

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --group)
            SHOW_GROUP=true
            shift
            ;;
        --fmriprep-dir)
            FMRIPREP_DIR="$2"
            shift 2
            ;;
        --subject)
            SUB="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [[ ${#POSITIONAL[@]} -eq 1 ]]; then
    MD_OUTPUT_DIR="${POSITIONAL[0]}"
elif [[ ${#POSITIONAL[@]} -eq 2 ]]; then
    if [[ -z "$SUB" ]]; then
        SUB="${POSITIONAL[0]}"
    fi
    MD_OUTPUT_DIR="${POSITIONAL[1]}"
else
    print_usage
    exit 1
fi

if [[ ! -d "$MD_OUTPUT_DIR" ]]; then
    echo "ERROR: MD output directory does not exist: $MD_OUTPUT_DIR"
    exit 1
fi

if [[ -z "$FMRIPREP_DIR" ]]; then
    for candidate in \
        "/Users/hannahmuller/nyx_mount_ptmp/2025_ibc_latent/outputs/preprocessing/fmriprep_out" \
        "/ptmp/hmueller2/2025_ibc_latent/outputs/preprocessing/fmriprep_out" \
        "/data/fmriprep_out"; do
        if [[ -d "$candidate" ]]; then
            FMRIPREP_DIR="$candidate"
            break
        fi
    done
fi

if [[ -z "$FMRIPREP_DIR" ]]; then
    echo "ERROR: fmriprep directory not found. Please specify with --fmriprep-dir"
    exit 1
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    export PATH="/Users/hannahmuller/workbench/bin_macosxub:$PATH"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [[ -n "${WORKBENCH_DIR:-}" ]] && [[ -d "$WORKBENCH_DIR/bin_linux64" ]]; then
        export PATH="$WORKBENCH_DIR/bin_linux64:$PATH"
    elif [[ -d "/opt/workbench/bin_linux64" ]]; then
        export PATH="/opt/workbench/bin_linux64:$PATH"
    elif [[ -d "/usr/local/workbench/bin_linux64" ]]; then
        export PATH="/usr/local/workbench/bin_linux64:$PATH"
    fi
fi

if ! command -v wb_command &> /dev/null; then
    echo "ERROR: wb_command not found. Please install Connectome Workbench."
    exit 1
fi

build_subject_spec() {
    local subj="$1"
    local subject_output_dir="${MD_OUTPUT_DIR}/sub-${subj}"

    if [[ ! -d "$subject_output_dir" ]]; then
        echo "Skipping sub-${subj}: missing ${subject_output_dir}"
        return 0
    fi

    local surf_l="${FMRIPREP_DIR}/sub-${subj}/anat/sub-${subj}_hemi-L_inflated.32k_fs_LR.surf.gii"
    local surf_r="${FMRIPREP_DIR}/sub-${subj}/anat/sub-${subj}_hemi-R_inflated.32k_fs_LR.surf.gii"

    if [[ ! -f "$surf_l" || ! -f "$surf_r" ]]; then
        echo "Skipping sub-${subj}: missing inflated surfaces"
        return 0
    fi

    local md_mean="${subject_output_dir}/sub-${subj}_MD_mean.dscalar.nii"
    local md_std="${subject_output_dir}/sub-${subj}_MD_std.dscalar.nii"
    if [[ ! -f "$md_mean" ]]; then
        echo "Skipping sub-${subj}: missing MD mean map"
        return 0
    fi

    local individual_dir="${subject_output_dir}/individual_contrasts"
    local -a individual_contrasts=()
    if [[ -d "$individual_dir" ]]; then
        while IFS= read -r line; do
            individual_contrasts+=("$line")
        done < <(find "$individual_dir" -name "*.dscalar.nii" | sort)
    fi

    local -a group_maps=()
    if [[ "$SHOW_GROUP" == true ]] && [[ -d "${MD_OUTPUT_DIR}/group" ]]; then
        [[ -f "${MD_OUTPUT_DIR}/group/group_MD_mean.dscalar.nii" ]] && group_maps+=("${MD_OUTPUT_DIR}/group/group_MD_mean.dscalar.nii")
        [[ -f "${MD_OUTPUT_DIR}/group/group_MD_std.dscalar.nii" ]] && group_maps+=("${MD_OUTPUT_DIR}/group/group_MD_std.dscalar.nii")
        [[ -f "${MD_OUTPUT_DIR}/group/group_MD_sem.dscalar.nii" ]] && group_maps+=("${MD_OUTPUT_DIR}/group/group_MD_sem.dscalar.nii")
    fi

    local scene_dir="${subject_output_dir}/scenes"
    mkdir -p "$scene_dir"
    local spec_path="${scene_dir}/sub-${subj}_md_maps.spec"

    {
        echo '<?xml version="1.0" encoding="UTF-8"?>'
        echo '<CaretSpecFile Version="1.0">'
        echo '   <DataFile Structure="CortexLeft" DataFileType="SURFACE" Selected="true">'
        echo "      ${surf_l}"
        echo '   </DataFile>'
        echo '   <DataFile Structure="CortexRight" DataFileType="SURFACE" Selected="true">'
        echo "      ${surf_r}"
        echo '   </DataFile>'
        echo '   <DataFile Structure="All" DataFileType="CONNECTIVITY_DENSE_SCALAR" Selected="true">'
        echo "      ${md_mean}"
        echo '   </DataFile>'
        if [[ -f "$md_std" ]]; then
            echo '   <DataFile Structure="All" DataFileType="CONNECTIVITY_DENSE_SCALAR" Selected="true">'
            echo "      ${md_std}"
            echo '   </DataFile>'
        fi

        for contrast_file in "${individual_contrasts[@]}"; do
            echo '   <DataFile Structure="All" DataFileType="CONNECTIVITY_DENSE_SCALAR" Selected="true">'
            echo "      ${contrast_file}"
            echo '   </DataFile>'
        done

        for group_file in "${group_maps[@]}"; do
            echo '   <DataFile Structure="All" DataFileType="CONNECTIVITY_DENSE_SCALAR" Selected="true">'
            echo "      ${group_file}"
            echo '   </DataFile>'
        done

        echo '</CaretSpecFile>'
    } > "$spec_path"

    echo "✓ Created spec for sub-${subj}: $spec_path"
}

if [[ -n "$SUB" ]]; then
    build_subject_spec "$SUB"
else
    mapfile -t SUBJECT_DIRS < <(find "$MD_OUTPUT_DIR" -maxdepth 1 -type d -name 'sub-*' | sort)
    if [[ ${#SUBJECT_DIRS[@]} -eq 0 ]]; then
        echo "ERROR: No subject directories found in $MD_OUTPUT_DIR"
        exit 1
    fi

    echo "No subject specified, defaulting to all subjects (${#SUBJECT_DIRS[@]} found)."
    for subject_dir in "${SUBJECT_DIRS[@]}"; do
        subject_id="$(basename "$subject_dir" | sed 's/^sub-//')"
        build_subject_spec "$subject_id"
    done
fi

echo "Done. Open a generated spec with: wb_view -spec-file <path-to-spec>"
