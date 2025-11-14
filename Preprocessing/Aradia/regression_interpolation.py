#!/usr/bin/env python3

import sys
import nilearn
import argparse
import numpy as np
from nilearn import image as nimg
from nilearn import signal
import pandas as pd
import nibabel as nib
from scipy import linalg
import os

# import input arguments
# Create the parser
parser = argparse.ArgumentParser(description='Process some input and output files.')

# Add the arguments
parser.add_argument('-i', '--input1', type=str, help='Input file (EPI with the first 5 volumes removed.)')
parser.add_argument('-r', '--regs', type=str, help='Regressors (txt with the first 5 volumes removed.)')
parser.add_argument('-FD', '--input2', type=str, help='FD txt file.')
parser.add_argument('-TR', '--reptime', type=float, help='Repetition Time.')
parser.add_argument('-o', '--output', type=str, help='Output file')
parser.add_argument('--MC_scrub', action='store_true', help='disable motion correction scrubbing (if this flag is given, no scrub is performed).')
parser.add_argument('--low-pass', dest='low_pass', type=float, default=None,
                    help='Low-pass cutoff in Hz. Default 0.08 (resting). Use --no-low-pass to disable.')
parser.add_argument('--high-pass', dest='high_pass', type=float, default=None,
                    help='High-pass cutoff in Hz. Default 0.009.')
parser.add_argument('--no-low-pass', action='store_true',
                    help='Disable low-pass filtering.')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
in_file = args.input1
FD_txt = args.input2
regressors = args.regs
TR = args.reptime
output_file = args.output
MC_scrub = args.MC_scrub

# Resolve filter settings with sensible defaults
if args.no_low_pass:
    low_pass = None
else:
    low_pass = args.low_pass if args.low_pass is not None else 0.08
high_pass = args.high_pass if args.high_pass is not None else 0.009
print(f"Filtering configuration -> high_pass={high_pass} Hz, low_pass={low_pass if low_pass is not None else 'None'}")

# Announce intended scrubbing behavior based on flags/inputs
if MC_scrub:
    print("Motion correction scrubbing is disabled by flag (--MC_scrub).")
elif FD_txt is None or not os.path.isfile(FD_txt):
    print("FD file not provided or not found; scrubbing will be disabled.")
    MC_scrub = True
else:
    print("Motion correction scrubbing is enabled (scrubbing will be performed).")

print("Reading in files")
# epi 
epi = nib.load(in_file)
# use the original NIfTI image to get the affine (needed for saving later)
if isinstance(epi, nib.Cifti2Image):
    # CIFTI files do not use affine; store header for saving later
    header = epi.header
else:
    affine = epi.affine

# extract data from nibabel image object
np_epi = epi.get_fdata()
shape = np_epi.shape

if isinstance(epi, nib.Cifti2Image):
    # CIFTI: shape is (n_timepoints, n_locations)
    print(f"CIFTI data shape: {np_epi.shape}")
    np_epi_2d = np_epi  # shape (n_timepoints, n_locations)
    n_timepoints = shape[0]
else:
    # NIfTI: shape is (X, Y, Z, T)
    n_timepoints = shape[3]
    np_epi_2d = np_epi.reshape(-1, n_timepoints).T  # shape (n_timepoints, n_voxels)

# delete not needed variables to avoid memory issues
# del epi
# del np_epi

# Load regressors and sanitize to avoid NaN/Inf issues during QR
reg_array_raw = np.loadtxt(regressors)
reg_df = pd.DataFrame(reg_array_raw)
reg_df = reg_df.replace([np.inf, -np.inf], np.nan)
n_nan_before = int(reg_df.isna().sum().sum())
if n_nan_before:
    print(f"DEBUG: regressors NaNs before interpolate/fill: {n_nan_before}")
reg_df = reg_df.interpolate(axis=0, limit_direction="both").fillna(0.0)
stds = reg_df.std(axis=0)
keep_cols = stds > 0
n_drop = int((~keep_cols).sum())
if n_drop:
    print(f"DEBUG: dropping {n_drop} constant/zero confound columns")
regressor_array = reg_df.loc[:, keep_cols].values

# Build sample_mask
if not MC_scrub:
    # FD provided and scrubbing enabled
    FD_clean = pd.read_csv(FD_txt, sep=' ', header=None)
    print(f"FD_clean shape (rows, cols): {FD_clean.shape}")
    if FD_clean.shape[0] != n_timepoints:
        print(f"ERROR: Number of rows in FD_clean ({FD_clean.shape[0]}) does not match number of timepoints in BOLD ({n_timepoints}).")
        raise ValueError("Timepoint mismatch between FD_clean and BOLD data.")
    sample_mask = np.array(FD_clean.iloc[:, 1] == 0, dtype=bool)
else:
    # No scrubbing: keep all timepoints
    sample_mask = None
    print("DEBUG: sample_mask=None (no scrubbing)")

print(f"np_epi_2d.shape (timepoints, features): {np_epi_2d.shape}")
print(f"regressor_array.shape (timepoints, regressors): {regressor_array.shape}")

# run interpolation + regression
# not standardised bc input files should be standardised as input
# Bandpass filter settings from Gordon et al.: 0.009 Hz < f < 0.08 Hz

def clean_no_scrub(
    signals,
    runs=None,
    detrend=True,
    standardize="zscore",
    sample_mask=None,
    confounds=None,
    standardize_confounds=True,
    filter="butterworth",
    low_pass=None,
    high_pass=None,
    t_r=2.5,
    ensure_finite=False,
    extrapolate=True,
    **kwargs,
):
    """
    Exactly like nilearn.signal.clean without motion scrubbing
    """
    # Raise warning for some parameter combinations when confounds present
    if confounds is not None:
        signal._check_signal_parameters(detrend, standardize_confounds)
    # check if filter parameters are satisfied and return correct filter
    filter_type = signal._check_filter_parameters(filter, low_pass, high_pass, t_r)

    # Read confounds and signals
    signals, runs, confounds, sample_mask = signal._sanitize_inputs(
        signals, runs, confounds, sample_mask, ensure_finite
    )

    # Process each run independently
    if runs is not None:
        return signal._process_runs(
            signals,
            runs,
            detrend,
            standardize,
            confounds,
            sample_mask,
            filter_type,
            low_pass,
            high_pass,
            t_r,
        )

    # For the following steps, sample_mask should be either None or index-like

    # Generate cosine drift terms using the full length of the signals
    if filter_type == "cosine":
        confounds = signal._create_cosine_drift_terms(
            signals, confounds, high_pass, t_r
        )

    # Interpolation / censoring
    signals, confounds, sample_mask = signal._handle_scrubbed_volumes(
        signals, confounds, sample_mask, filter_type, t_r, extrapolate
    )
    # Detrend
    # Detrend and filtering should apply to confounds, if confound presents
    # keep filters orthogonal (according to Lindquist et al. (2018))
    # Restrict the signal to the orthogonal of the confounds
    original_mean_signals = signals.mean(axis=0)
    if detrend:
        signals = signal.standardize_signal(
            signals, standardize=False, detrend=detrend
        )
        if confounds is not None:
            confounds = signal.standardize_signal(
                confounds, standardize=False, detrend=detrend
            )

    # Butterworth filtering
    if filter_type == "butterworth":
        butterworth_kwargs = {
            k.replace("butterworth__", ""): v
            for k, v in kwargs.items()
            if k.startswith("butterworth__")
        }
        signals = signal.butterworth(
            signals,
            sampling_rate=1.0 / t_r,
            low_pass=low_pass,
            high_pass=high_pass,
            **butterworth_kwargs,
        )
        if confounds is not None:
            # Apply low- and high-pass filters to keep filters orthogonal
            # (according to Lindquist et al. (2018))
            confounds = signal.butterworth(
                confounds,
                sampling_rate=1.0 / t_r,
                low_pass=low_pass,
                high_pass=high_pass,
                **butterworth_kwargs,
            )

        # # apply sample_mask to remove censored volumes after signal filtering
        # if sample_mask is not None:
        #     signals, confounds = _censor_signals(
        #         signals, confounds, sample_mask
        #     )

    # Remove confounds
    if confounds is not None:
        confounds = signal.standardize_signal(
            confounds, standardize=standardize_confounds, detrend=False
        )
        if not standardize_confounds:
            # Improve numerical stability by controlling the range of
            # confounds. We don't rely on standardize_signal as it removes any
            # constant contribution to confounds.
            confound_max = np.max(np.abs(confounds), axis=0)
            confound_max[confound_max == 0] = 1
            confounds /= confound_max

        # Pivoting in qr decomposition was added in scipy 0.10
        Q, R, _ = linalg.qr(confounds, mode="economic", pivoting=True)
        Q = Q[:, np.abs(np.diag(R)) > np.finfo(np.float64).eps * 100.0]
        signals -= Q.dot(Q.T).dot(signals)

    # Standardize
    if not standardize:
        return signals

    # detect if mean is close to zero; This can obscure the scale of the signal
    # with percent signal change standardization. This should happen when the
    # data was 1. detrended 2. high pass filtered.
    filtered_mean_check = (
        np.abs(signals.mean(0)).mean() / np.abs(original_mean_signals).mean()
        < 1e-1
    )
    if standardize == "psc" and filtered_mean_check:
        # If the signal is detrended, the mean signal will be zero or close to
        # zero. If signal is high pass filtered with butterworth, the constant
        # (mean) will be removed. This is detected through checking the scale
        # difference of the original mean and filtered mean signal. When the
        # mean is too small, we have to know the original mean signal to
        # calculate the psc to avoid weird scaling.
        signals = signal.standardize_signal(
            signals + original_mean_signals,
            standardize=standardize,
            detrend=False,
        )
    else:
        signals = signal.standardize_signal(
            signals,
            standardize=standardize,
            detrend=False,
        )
    return signals






# do a try if the epi and regressors are the same length
print("Trying regression.")
try:
    # check if the number of rows in both arrays is the same
    if np_epi_2d.shape[0] != regressor_array.shape[0]:
        print(f"ERROR: Number of timepoints in BOLD ({np_epi_2d.shape[0]}) does not match number in regressors ({regressor_array.shape[0]}).")
        raise ValueError("Timepoint mismatch between BOLD and regressors.")
    
    print(f"Number of timepoints in BOLD (np_epi_2d.shape[0]): {np_epi_2d.shape[0]}")
    print(f"Number of timepoints in regressors (regressor_array.shape[0]): {regressor_array.shape[0]}")
    print(f"np_epi_2d.shape[0] (should be timepoints): {np_epi_2d.shape[0]}")
    print(f"regressor_array.shape[0] (should be timepoints): {regressor_array.shape[0]}")
    
    # only try this operation if the number of rows is identical
    #print("Running cleaning.")
    if MC_scrub==False:
        print("Running cleaning with motion scrubbing...")
        cleaned_signals = signal.clean(np_epi_2d, 
                    runs=None, 
                    detrend=True, 
                    standardize=False, 
                    sample_mask=sample_mask,
                    confounds=regressor_array,
                    standardize_confounds=False,
                    filter='butterworth', 
                    low_pass=low_pass,
                    high_pass=high_pass,
                    t_r=TR, 
                    ensure_finite=True,            # CHANGED: be robust to non-finite in signals
                    extrapolate=True)
    else:
        print("Running cleaning without motion scrubbing...")
        cleaned_signals = clean_no_scrub(np_epi_2d, 
                    runs=None, 
                    detrend=True, 
                    standardize=False, 
                    sample_mask=sample_mask,       # None when no scrubbing
                    confounds=regressor_array,
                    standardize_confounds=False,
                    filter='butterworth', 
                    low_pass=low_pass,
                    high_pass=high_pass,
                    t_r=TR, 
                    ensure_finite=True,            # CHANGED
                    extrapolate=True)

    
    print(f"Cleaned signals shape: {cleaned_signals.shape}")
    new_volumes = cleaned_signals.shape[0]

    # save output
    print("Saving output.")

    # reshape back to 4D: (130, 130, 85, 595) -> for robustness stored as variables (note, that these dimensions are correct for my lowres data)
    if isinstance(epi, nib.Cifti2Image):
        # Get axes from original image
        brain_model_axis = epi.header.get_axis(1)
        time_axis = nib.cifti2.cifti2_axes.SeriesAxis(start=0, step=TR, size=cleaned_signals.shape[0])
        cleaned_data_2d = cleaned_signals  # shape (n_timepoints, n_locations)
        cleaned_img = nib.Cifti2Image(cleaned_data_2d, (time_axis, brain_model_axis))
    else:
        cleaned_data_4d = cleaned_signals.T.reshape(shape[0], shape[1], shape[2], new_volumes)
        cleaned_img = nib.Nifti1Image(cleaned_data_4d, affine)
    nib.save(cleaned_img, output_file)
    print("Saved.")
    
except ValueError as e:
    print(e)