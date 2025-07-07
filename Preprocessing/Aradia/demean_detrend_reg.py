"""
Demeans and detrends the regressor file
"""

import argparse
import pandas as pd
import numpy as np
from scipy.signal import detrend

parser = argparse.ArgumentParser(description='Process some input and output files.')
parser.add_argument('-i', '--input', type=str, help='Input file: Regressors.')
parser.add_argument('-odmdt', '--outputdmdt', type=str, help='Output file: demeaned detrended regressors')
parser.add_argument('-o', '--output', type=str, help='Output file: z-scored regressors')

# Parse the arguments
args = parser.parse_args()

in_file=args.input
out_demean=args.outputdmdt
out_file=args.output

print("Reading in regressors.")
df = pd.read_csv(in_file, sep=' ', header=None)

print("Demeaning.")
# demean each column
df_demeaned = df - df.mean()

print("Detrending.")
# detrend each column
df_detrended = pd.DataFrame(detrend(df_demeaned, axis=0))
df_detrended.to_csv(out_demean, sep=' ', header=False, index=False)


print("Z-scoring.")
# z-score each column
df_zscored = (df_detrended - df_detrended.mean()) / df_detrended.std()

print("Saving.")
# save the result to a new file
df_zscored.to_csv(out_file, sep=' ', header=False, index=False)