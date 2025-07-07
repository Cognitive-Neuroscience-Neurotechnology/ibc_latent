"""
Removes the first x volumes from a file.
"""

import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Process some input and output files.')
parser.add_argument('-i', '--input', type=str, help='Input file')
parser.add_argument('-v', '--volumes', type=int, help='Number of volumes to remove.')
parser.add_argument('-o', '--output', type=str, help='Output file')

# Parse the arguments
args = parser.parse_args()

in_file=args.input
volumes=args.volumes
out_file=args.output

print("Reading input file.")
df = pd.read_csv(in_file, sep=' ', header=None)

print("Removing first 5 volumes.")
new_df = df.drop(index=df.index[:volumes], axis=0)
print(new_df.shape)

print("Saving cleaned regressors.")
new_df.to_csv(out_file, sep=' ',header=False, index=False)