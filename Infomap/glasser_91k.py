from xcp_d.utils.atlas import get_atlas_cifti

atlas_file, atlas_labels_file, atlas_metadata_file = get_atlas_cifti("Glasser")

print("Atlas file:", atlas_file)
print("Labels file:", atlas_labels_file)
print("Metadata file:", atlas_metadata_file)