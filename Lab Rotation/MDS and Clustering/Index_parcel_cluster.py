import pandas as pd
import json
import os

### THIS CODE IS USED TO CREATE A FILE THAT MAPS PARCELS TO CLUSTERS ###
# The code reads the KMeans cluster results and the parcel labels for each subject
# The resulting dataframe has columns: index (original) - parcel_label - index_Xp (indices from matrices) - cluster

'''
Steps:
1. import cluster results per index from /home/hmueller2/ibc_code/ibc_output_MDS/run_06/kmeans_labels_sub-XX.csv
2. import parcel labels from indexing '/home/hmueller2/ibc_output_RA/raw/topographic_alignment/parcel_names_sub-XX.json'
3. Concetanate the two to get a File with columns: index - parcel_label - cluster
'''

subjects = [f'sub-{i:02d}' for i in range(1, 16) if i not in [3, 10]]
run = 'run_10-3d-2clusters'

# Load the FPN parcellation file
fpn_parcellation_path = '/home/hmueller2/Downloads/FPN_parcellation_cole/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt'
fpn_df = pd.read_csv(fpn_parcellation_path, sep='\t')

for subject in subjects:
    print(f'Processing {subject}')

    # Step 1: Import cluster results per index
    cluster_results_path = f'/home/hmueller2/ibc_code/ibc_output_MDS/{run}/kmeans_labels_{subject}.csv'
    cluster_df = pd.read_csv(cluster_results_path, header=None, names=['cluster'])

    # Add an 'index' column to cluster_df
    cluster_df.reset_index(inplace=True)

    # Step 2: Import parcel labels
    parcel_labels_path = f'/home/hmueller2/ibc_code/ibc_output_RA/raw/topographic_alignment/parcel_names_{subject}.json'
    with open(parcel_labels_path, 'r') as file:
        parcel_labels = json.load(file)

    # Create a DataFrame with an 'index' column and a 'parcel_label' column
    parcel_df = pd.DataFrame({
        'index': range(len(parcel_labels)),
        'parcel_label': [label + '_ROI' for label in parcel_labels]  # Add '_ROI' suffix to match GLASSERLABELNAME
    })

    # Step 3: Merge parcel_df with fpn_df to get the 'index_Xp' column
    merged_df = pd.merge(parcel_df, fpn_df[['GLASSERLABELNAME', 'KEYVALUE']], left_on='parcel_label', right_on='GLASSERLABELNAME', how='left')
    merged_df.rename(columns={'KEYVALUE': 'index_Xp'}, inplace=True)

    # Step 4: Concatenate the merged_df with cluster_df to get a file with columns: index - parcel_label - index_Xp - cluster
    result_df = pd.merge(merged_df[['index', 'parcel_label', 'index_Xp']], cluster_df, on='index')
    result_df.columns = ['index', 'parcel_label', 'index_Xp', 'cluster']

    # Create the output directory if it doesn't exist
    output_dir = f'/home/hmueller2/ibc_code/ibc_output_KMeans_onMDS/{run}'
    os.makedirs(output_dir, exist_ok=True)

    # Save the result to a CSV file
    output_path = f'{output_dir}/parcel-cluster_{subject}.csv'
    result_df.to_csv(output_path, index=False)