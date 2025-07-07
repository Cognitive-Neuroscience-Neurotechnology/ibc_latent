import sys
import nibabel as nib
import numpy as np
from scipy.signal import detrend

infile = sys.argv[1]
outfile = sys.argv[2]

img = nib.load(infile)
data = img.get_fdata()
# Detrend along time axis (last axis)
data_detrend = detrend(data, axis=-1, type='linear')
img_detrend = nib.Cifti2Image(data_detrend, img.header, img.nifti_header)
img_detrend.to_filename(outfile)