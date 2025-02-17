import pandas as pd
import numpy as np
from nibabel.freesurfer.io import read_annot

# Step 1: Load network_partition (.txt)
network_partition_path = '/home/hmueller2/Downloads/Cole_FPN_Parcellation/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt'
network_partition = pd.read_csv(network_partition_path, sep='\t')

# Step 2: Only keep GLASSERLABELNAMEs of those that are in the frontoparietal network (Networkkey = 7)
fpn_parcels = network_partition[network_partition['NETWORKKEY'] == 7]
fpn_parcels_names = fpn_parcels['GLASSERLABELNAME'].dropna().tolist()

# Step 3: Load annot_file
lh_annot_file = '/home/hmueller2/Downloads/Atlas/glasser_fsaverage/3498446/lh.HCP-MMP1.annot'
rh_annot_file = '/home/hmueller2/Downloads/Atlas/glasser_fsaverage/3498446/rh.HCP-MMP1.annot'

labels_lh, ctab_lh, names_lh = read_annot(lh_annot_file)
labels_rh, ctab_rh, names_rh = read_annot(rh_annot_file)

# Step 4: Do a vertex-to-parcel mapping for Frontoparietal parcels
vertices_lh = np.arange(len(labels_lh))
vertices_rh = np.arange(len(labels_rh))

lh_parcel_mapping = {vertex: names_lh[label] for vertex, label in zip(vertices_lh, labels_lh)}
rh_parcel_mapping = {vertex: names_rh[label] for vertex, label in zip(vertices_rh, labels_rh)}

fpn_parcels_lh_mapping = {vertex: lh_parcel_mapping[vertex] for vertex in vertices_lh if lh_parcel_mapping[vertex].decode('utf-8') in fpn_parcels_names}
fpn_parcels_rh_mapping = {vertex: rh_parcel_mapping[vertex] for vertex in vertices_rh if rh_parcel_mapping[vertex].decode('utf-8') in fpn_parcels_names}

# Print the mappings
print("Left Hemisphere Parcel Mapping (first 5 entries):")
print({k: fpn_parcels_lh_mapping[k] for k in list(fpn_parcels_lh_mapping)[:5]})
print(f"length of lh mapping: {len(fpn_parcels_lh_mapping)}")

print("Right Hemisphere Parcel Mapping (last 5 entries):")
print({k: fpn_parcels_rh_mapping[k] for k in list(fpn_parcels_rh_mapping)[-5:]})
print(f"length of rh mapping: {len(fpn_parcels_rh_mapping)}")

# Use fpn_parcels_lh_mapping & fpn_parcels_rh_mapping to extract which vertices belong to which parcels of the frontoparietal network