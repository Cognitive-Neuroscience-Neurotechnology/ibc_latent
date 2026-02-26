import nibabel as nib
import numpy as np

subject = "06"
working_dir = '/ptmp/hmueller2/2025_ibc_latent/outputs'

parc_filename = f"{working_dir}/individual_networks_october/sub-{subject}/resting_state/sub-{subject}_individual_nets_concat.ptseries.nii"

# Load the ptseries file
img = nib.load(parc_filename)
data = img.get_fdata()

print("="*80)
print("NETWORK PARCELLATION INSPECTION")
print("="*80)
print(f"Shape: {data.shape}")
print(f"Number of networks: {data.shape[1]}")

# Get the parcellation axis (contains network names)
parc_axis = img.header.get_axis(1)

print("\n" + "="*80)
print("NETWORK NAMES AND INDICES")
print("="*80)

dmn_indices = []
dan_indices = []
fpn_indices = []

for idx, parcel_name in enumerate(parc_axis.name):
    print(f"Index {idx}: {parcel_name}")
    
    name_lower = parcel_name.lower()
    
    if 'default' in name_lower:
        dmn_indices.append(idx)
    
    if 'dorsal' in name_lower and 'attention' in name_lower:
        dan_indices.append(idx)
    
    if 'frontoparietal' in name_lower or 'fronto' in name_lower:
        fpn_indices.append(idx)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"DMN indices: {dmn_indices}")
print(f"DAN indices: {dan_indices}")
print(f"FPN indices: {fpn_indices}")

print("\n" + "="*80)
print("AFTER REMOVING FPN (index 8) AND NOISE (last index):")
print("="*80)
print("Original DAN indices:", dan_indices)
if fpn_indices and fpn_indices[0] == 8:
    adjusted_dan = [idx - 1 if idx > 8 else idx for idx in dan_indices]
    print(f"Adjusted DAN indices (after deleting index 8): {adjusted_dan}")
else:
    print("No adjustment needed")

print("\n" + "="*80)
print("SUGGESTED CODE FOR label_overwrite.py:")
print("="*80)
print(f"dmn_idx = {dmn_indices}")
print(f"dan_idx = {adjusted_dan if fpn_indices and fpn_indices[0] == 8 else dan_indices}")