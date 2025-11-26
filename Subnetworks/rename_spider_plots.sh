#!/bin/bash
# /home/hmueller2/ibc_code/ibc_latent/Subnetworks/rename_spider_plots.sh

SPIDER_PLOTS_DIR="/ptmp/hmueller2/Downloads/subnetworks/spider_plots/"
SUBJECTS_FILE="/ptmp/hmueller2/Downloads/subjects_resting.txt"

# Read subjects from file and rename files
while IFS= read -r subject; do
    # Remove any whitespace
    subject=$(echo "$subject" | xargs)
    
    # Rename kmeans to infomap
    old_file="${SPIDER_PLOTS_DIR}sub-${subject}_FPN_kmeans_2_spider_plot.png"
    new_file="${SPIDER_PLOTS_DIR}sub-${subject}_FPN_infomap_2_spider_plot.png"
    
    if [ -f "$old_file" ]; then
        mv "$old_file" "$new_file"
        echo "Renamed: $(basename "$old_file") -> $(basename "$new_file")"
    fi
    
    # Rename kmeans_neg to infomap_neg
    old_file="${SPIDER_PLOTS_DIR}sub-${subject}_FPN_kmeans_2_spider_plot_neg.png"
    new_file="${SPIDER_PLOTS_DIR}sub-${subject}_FPN_infomap_2_spider_plot_neg.png"
    
    if [ -f "$old_file" ]; then
        mv "$old_file" "$new_file"
        echo "Renamed: $(basename "$old_file") -> $(basename "$new_file")"
    fi
done < "$SUBJECTS_FILE"

echo "All files renamed successfully!"