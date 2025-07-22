import sys
import nibabel as nib

infile = sys.argv[1]
outfile = sys.argv[2]
n_remove = int(sys.argv[3])

img = nib.load(infile)
data = img.get_fdata()
trimmed = data[n_remove:, ...]

# Get axes from original image
axis0 = img.header.get_axis(0)
axis1 = img.header.get_axis(1)

# Create a new time axis (SeriesAxis) with updated start and length
if isinstance(axis0, nib.cifti2.SeriesAxis):
    new_start = axis0.start + axis0.step * n_remove
    new_axis0 = nib.cifti2.SeriesAxis(start=new_start, step=axis0.step, size=trimmed.shape[0])
else:
    # For non-SeriesAxis, just slice the axis
    new_axis0 = axis0[n_remove:]

# Create new CIFTI image with updated axes
img_trim = nib.Cifti2Image(trimmed, (new_axis0, axis1), img.nifti_header)
img_trim.to_filename(outfile)