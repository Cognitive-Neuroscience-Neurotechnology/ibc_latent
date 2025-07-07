import argparse
import numpy as np
import sys
import RR_utils as RR

# Create the parser
parser = argparse.ArgumentParser(description='Process some input and output files.')

# Add the arguments
parser.add_argument('-i', '--input1', type=str, help='Input file')
parser.add_argument('-FD', '--input2', type=float, help='FD threshold')
parser.add_argument('-o', '--output', type=str, help='Output file')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
in_file = args.input1
FD_thresh = args.input2
output_file = args.output



print("Calculating FD now.")
#und dann laufen lassen
FD_power = RR.calculate_FD_P(in_file, rot_type='degrees')

# pre-initialise contamination column
is_contam=list()

print("Calculating contamination now.")
for row in FD_power:
    
    if row > FD_thresh:
        # value in matrix = 1 if motion contaminated
        is_contam.append(1)
    else:
        # value in matrix = 0
        is_contam.append(0)

# convert to array
contam_array=np.array(is_contam)

# add column to mane array
combined_array=np.column_stack((FD_power, contam_array))

# save to txt file
print("Saving...")
# save outcomes
np.savetxt(output_file, combined_array, fmt='%.6f %d')
print("Done.")