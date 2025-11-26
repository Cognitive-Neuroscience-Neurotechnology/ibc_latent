studyDataDir=$1
subject=$2

base_dir=${studyDataDir}/derivatives/MSC_PFM/${subject}
pfm_subdir=${base_dir}/pfm
if [ -d "$pfm_subdir" ]; then
    PFM_dir=$pfm_subdir
else
    PFM_dir=$base_dir
fi


files=(
  ${PFM_dir}/Bipartite_PhysicalCommunities+AlgorithmicLabeling.dlabel.nii
  ${PFM_dir}/Bipartite_PhysicalCommunities+AlgorithmicLabeling_InfoMapCommunities.dlabel.nii
  ${base_dir}/concatenated_tseries.dtseries.nii
)

# source: https://gist.github.com/benkay86/98adad49994445611531acad619b9889

for file in "${files[@]}"; do
    echo "processing: $file"
    
    file_no_path=$(basename "$file")  # remove path

    #  remove extensions
    file_base=${file_no_path%.dlabel.nii}
    file_base=${file_no_path%.dtseries.nii}

    if [[ "$file" == *.dlabel.nii ]]; then
        # separate out cortex structures
        left_label_gii=${PFM_dir}/${file_base}.L.label.gii
        right_label_gii=${PFM_dir}/${file_base}.R.label.gii
        wb_command -cifti-separate \
            $file COLUMN \
            -label CORTEX_LEFT ${left_label_gii} \
            -label CORTEX_RIGHT ${right_label_gii}

        # assemble new label file
        new_label_file=${PFM_dir}/${file_base}_only_cortex.dlabel.nii
        wb_command -cifti-create-label \
            ${new_label_file} \
            -left-label ${left_label_gii} \
            -right-label ${right_label_gii}

        rm ${left_label_gii} ${right_label_gii}
    elif [[ "$file" == *.dtseries.nii ]]; then
        ext=".dtseries.nii"
        left_gii=${PFM_dir}/${file_base}.L.func.gii
        right_gii=${PFM_dir}/${file_base}.R.func.gii

        # separate out cortex structures
        wb_command -cifti-separate \
            $file COLUMN \
            -metric CORTEX_LEFT ${left_gii} \
            -metric CORTEX_RIGHT ${right_gii}
        
        # construct new dtseries
        new_dtseries_file=${PFM_dir}/${file_base}_only_cortex.dtseries.nii
        wb_command -cifti-create-dense-timeseries \
            ${new_dtseries_file} \
            -left-metric ${left_gii} \
            -right-metric ${right_gii}

        rm ${left_gii} ${right_gii}
    fi

done

