import sys
import nibabel as nib

infile = sys.argv[1]
outfile = sys.argv[2]
n_remove = int(sys.argv[3])

img = nib.load(infile)
data = img.get_fdata()
trimmed = data[..., n_remove:]
img_trim = nib.Cifti2Image(trimmed, img.header, img.nifti_header)
img_trim.to_filename(outfile)