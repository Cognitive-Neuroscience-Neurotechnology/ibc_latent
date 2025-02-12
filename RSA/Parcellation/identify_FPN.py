# Goal: extract the parcels that belong to the Frontoparietal network. 
# Source: ColeAnticevicNetPartition (GitHub)
# SPACE:
    # Cortical: Glasser 2016 parcellation
    # Subcortical: HCP’s subcortical parcellation

import pandas as pd

# Define the path to the Excel file
file_path = '/home/hmueller2/Downloads/Cole_FPN_Parcellation/CAB-NP_v1.1_Labels-ReorderedbyNetworks.xlsx'

# Load the Excel file
df = pd.read_excel(file_path)

# Filter the DataFrame to find parcels that belong to the 'Frontoparietal' network
fpn_parcels = df[df['NETWORK'] == 'Frontoparietal']

# Print the parcels
print(fpn_parcels)