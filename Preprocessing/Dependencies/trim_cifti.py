import sys
import nibabel as nib
# Loads the CIFTI, removes the first n timepoints, and saves the result. Equivalent of what Aradia's 'fslroi.
infile = sys.argv[1] # Input CIFTI file
outfile = sys.argv[2] # Output CIFTI file
n_remove = int(sys.argv[3]) # Number of timepoints to remove from the beginning

img = nib.load(infile)
data = img.get_fdata()
trimmed = data[..., n_remove:]
img_trim = nib.Cifti2Image(trimmed, img.header, img.nifti_header)
img_trim.to_filename(outfile)