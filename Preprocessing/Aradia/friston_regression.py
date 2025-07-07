
"""
Code is taken from fmriFlows: https://nbviewer.org/github/miykael/fmriflows/blob/master/notebooks/03_preproc_func.ipynb
and adapted for current use.

The code was made for 6 columns but we have 8 because the last two are max displacement. 
IMPORTANT: load txt file without last two columns!

Also added a step to convert from degrees to radians


"""

import numpy as np
# from os.path import basename, abspath
import argparse

# get input arguments
parser = argparse.ArgumentParser(description='Process some input and output files.')
parser.add_argument('-im', '--input_motion', type=str, help='Input file mc_reordered.par')
parser.add_argument('-i_csf', '--input_csf', type=str, help='Input file: average csf timeseries')
parser.add_argument('-i_wm', '--input_wm', type=str, help='Input file: average WM timeseries')
parser.add_argument('-i_glob', '--input_glob', type=str, help='Input file: average global signal')
parser.add_argument('-o', '--output', type=str, help='Output file')
parser.add_argument('-o_nuis', '--output_nuis', type=str, help='Output file: all nuisance regressors.')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
in_file = args.input_motion
csf_txt = args.input_csf
wm_txt = args.input_wm
global_signal_txt = args.input_glob
out_file = args.output
out_nuisance = args.output_nuis


# Computes Friston 24-parameter model (Friston et al., 1996)
def compute_friston24(in_file, out_file, rot_type='degrees'):

    # Load raw motion parameters
    mp_raw = np.loadtxt(in_file)

    if rot_type=='degrees':
        # this should denote the unit of the input file. The script would like rotations to be in radians
        # formula: degrees * np.pi /180
        mp_input=mp_raw[:,:6]
        mp_input[:,:3]=np.radians(mp_raw[:,:3])
    elif rot_type=='radians':
        mp_input=mp_raw[:,:6]
    
    # Get motion paremter one time point before (first order difference)
    mp_minus1 = np.vstack(([0] * 6, mp_input[1:]))
    
    # Combine the two
    mp_combine = np.hstack((mp_input, mp_minus1))

    # Add the square of those parameters to allow correction of nonlinear effects
    mp_friston = np.hstack((mp_combine, mp_combine**2))

    print("Computed Friston parameters.")

    # # Save friston 24-parameter model in new txt file
    # out_file = abspath(basename(in_file).replace('.txt', 'friston24.txt'))
    np.savetxt(out_file, mp_friston, fmt='%.8e', delimiter=' ', newline='\n')
    
    return out_file


print("Computing Friston regression parameters now.")
compute_friston24(in_file, out_file, rot_type='degrees')

print("Loading other regressors now.")
# merge this file with the txt files containing wm, csf, global signal for the regressions
# csf_ts, wm_ts, global_signal
csf_ts = np.loadtxt(csf_txt)
wm_ts = np.loadtxt(wm_txt)

# re-load the friston 24 parameters
friston_motion= np.loadtxt(out_file)

print("Compiling arrays.")

# check if global signal regression was selected
if global_signal_txt == "None":
    print("No global signal extraction was chosen.")
    array_list=[csf_ts,wm_ts,friston_motion]
else:
    print("Including global signal regressors.")
    global_signal = np.loadtxt(global_signal_txt)
    array_list=[csf_ts,wm_ts,global_signal,friston_motion]

# combine all regressors to one array
nuisance_regressors = np.column_stack(array_list)

# print the first 5 for quality checking
print(nuisance_regressors[:5,:])

print("Saving regressors.")
np.savetxt(out_nuisance, nuisance_regressors,fmt='%.6f', delimiter=' ', newline='\n')
print(nuisance_regressors.shape)
print("Saved regressors. Done.")