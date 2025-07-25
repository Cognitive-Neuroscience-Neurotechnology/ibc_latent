"""

Plot regressors.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Process some input and output files.')
parser.add_argument('-i', '--input', type=str, help='Input file')
parser.add_argument('-glob', '--glob', required=True, type=str, help='Global signal regression? yes or no.')
parser.add_argument('-o', '--output', type=str, help='Output file')

# Parse the arguments
args = parser.parse_args()

regressors=args.input
global_sig=args.glob
out_file=args.output

print("Loading regressors.")
df = pd.read_csv(regressors, sep=' ', header=None)

# create column names
if global_sig.lower() in ['yes', 'y']:
    column_names=['CSF', 'WM', 'Global signal', 'Rotation 1 (rad)', 'Rotation 2 (rad)', 'Rotation 3 (rad)', 'Translation 1', 'Translation 2', 'Translation 3', 'Rotation 1 (rad) t-1', 'Rotation 2 (rad) t-1', 'Rotation 3 (rad) t-1', 'Translation 1 t-1', 'Translation 2 t-1', 'Translation 3 t-1', 'Rotation 1 (rad) **2', 'Rotation 2 (rad) **2', 'Rotation 3 (rad) **2', 'Translation 1 **2', 'Translation 2 **2', 'Translation 3 **2', 'Rotation 1 (rad) t-1 **2', 'Rotation 2 (rad) t-1 **2', 'Rotation 3 (rad) t-1 **2', 'Translation 1 t-1 **2', 'Translation 2 t-1 **2', 'Translation 3 t-1 **2']
else:
    column_names=['CSF', 'WM', 'Rotation 1 (rad)', 'Rotation 2 (rad)', 'Rotation 3 (rad)', 'Translation 1', 'Translation 2', 'Translation 3', 'Rotation 1 (rad) t-1', 'Rotation 2 (rad) t-1', 'Rotation 3 (rad) t-1', 'Translation 1 t-1', 'Translation 2 t-1', 'Translation 3 t-1', 'Rotation 1 (rad) **2', 'Rotation 2 (rad) **2', 'Rotation 3 (rad) **2', 'Translation 1 **2', 'Translation 2 **2', 'Translation 3 **2', 'Rotation 1 (rad) t-1 **2', 'Rotation 2 (rad) t-1 **2', 'Rotation 3 (rad) t-1 **2', 'Translation 1 t-1 **2', 'Translation 2 t-1 **2', 'Translation 3 t-1 **2']

df.columns = column_names


print("Initialising plots.")
# plot regressors
fig, axes = plt.subplots(5, 1, figsize=(12, 15))  # 5 subplots stacked vertically

# first plot CSF, WM, Global signal
if global_sig.lower() in ['yes', 'y']:
    print("Plotting CSF, WM, Global signal.")
    axes[0].plot(df.index, df.iloc[:, :3])
    axes[0].set_title('Z-Scored Data (CSF, WM, Global Signal)')
    axes[0].legend(df.columns[:3], loc='upper right')
    axes[0].grid(True)
else:
    print("Plotting CSF, WM.")
    axes[0].plot(df.index, df.iloc[:, :2])
    axes[0].set_title('Z-Scored Data (CSF, WM)')
    axes[0].legend(df.columns[:3], loc='upper right')
    axes[0].grid(True)

# plot original motion parameters (3 rotations and 3 translations)
#print("Plotting motion parameters.")
axes[1].plot(df.index, df.iloc[:, 3:9])
axes[1].set_title('Z-Scored Data (Motion parameters)')
axes[1].legend(df.columns[3:9], loc='upper right')
axes[1].grid(True)

# plot motion parameters t-1
#print("Plotting motion parameters t-1.")
axes[2].plot(df.index, df.iloc[:, 9:15])
axes[2].set_title('Z-Scored Data (Motion parameters t-1)')
axes[2].legend(df.columns[9:15], loc='upper right')
axes[2].grid(True)

# plot motion parameters squared
#print("Plotting motion parameters squared.")
axes[3].plot(df.index, df.iloc[:, 15:21])
axes[3].set_title('Z-Scored Data (Motion parameters squared)')
axes[3].legend(df.columns[15:21], loc='upper right')
axes[3].grid(True)

# plot motion parameters t-1 squared
#print("Plotting motion parameters t-1 squared.")
axes[4].plot(df.index, df.iloc[:, 21:27])
axes[4].set_title('Z-Scored Data (Motion parameters t-1 squared)')
axes[4].legend(df.columns[21:27], loc='upper right')
axes[4].grid(True)

print("Saving file.")
plt.tight_layout()
fig.savefig(out_file)