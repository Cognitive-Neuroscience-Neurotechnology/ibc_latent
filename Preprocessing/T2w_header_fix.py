import nibabel as nib

nii_path = "/home/hmueller2/ptmp/Downloads/ibc_raw_test/sub-01/ses-00/anat/sub-01_ses-00_acq-spc_T2w.nii.gz"
img = nib.load(nii_path)
header = img.header

# Print spatial units
header.set_xyzt_units('mm', 'sec')
print("Spatial units after fix:", header.get_xyzt_units())

nib.save(img, nii_path)