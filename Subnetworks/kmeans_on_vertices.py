'''
(2)
kmeans clustering on all vertices of the FPN based on their connectivity profiles within the FPN.
'''

"""
Identify subnetworks using k-means like DiNicola et al., 2022 (https://doi.org/10.1152/jn.00211.2022)

1. z-score and concatenate timeseries
2. input timeseries into k-means (MATLAB) using default parameters
    vary k=10-20, chose k=6 
    (why k=6 is not mentioned in the methods, but I haven't read the whole paper so might be explained somewhere else)
3. identify networks based on "referential features" (cite Braga & Buckner, 2017) 

"""

import sys
sys.path.insert(1, '/home/hmueller2/ibc_code/ibc_latent/Preprocessing/Aradia')
import RR_utils as RR
import argparse
import os
import re



"""
Load data, z-score

"""

parser = argparse.ArgumentParser(description='a script to run k-means on the FPN')
parser.add_argument('--subject', help='do not include sub- prefix', default='p11')
parser.add_argument('--seq', help='sequenceName, e.g., task-rest_acq-lowresmb.')
#parser.add_argument('--dir', help='Base working directory, e.g. /ptmp/hmueller2/Downloads', required=True)
parser.add_argument('--half', help='Half of dataset to run across (odd or even).')
args = parser.parse_args()

sequenceName = args.seq
subject = args.subject
half = args.half

working_dir = '/ptmp/hmueller2/Downloads'
timeseries_dir = os.path.join(working_dir, "fmriprep_out", f'sub-{subject}')
network_dir = os.path.join(working_dir, "individual_networks", f'sub-{subject}')
output_dir = os.path.join(working_dir, "subnetworks", "kmeans", f'sub-{subject}')
os.makedirs(output_dir, exist_ok=True)

session_dirs=RR.get_sessions_dirs(timeseries_dir)
half_sessions = []
files_subjects=[]
print("Reading in data...")

# concatenate runs
for session in session_dirs:
    print(session)

    current_dir = os.path.join(timeseries_dir, session, 'postfmriprep', 'GLM')
    print("Current directory:", current_dir)

    pattern = re.compile(rf'sub-{subject}_{session}_task-.*_dir-.*_cleaned\.dtseries\.nii')
    run_list = RR.which_runs(current_dir, pattern)

    # construct filename
    files_per_session = []
    for run in run_list:
        # run is now the full filename (since run number is not in the filename)
        filenamebase = f'sub-{subject}_{session}_task-{sequenceName}_dir-{run}_cleaned.dtseries.nii'
        file = os.path.join(current_dir, filenamebase)
        files_per_session.append(file)
    
    # concatenate
    print("Concatenating runs...")
    output_filename = os.path.join(current_dir, f'sub-{subject}_{session}_task-{sequenceName}_concat.LR.32k.dtseries.nii')
    files_subjects.append(output_filename)
    # RR.concat_WB(files_per_session, output_filename, return_data=False)

    # extract even or odd for split-half
    session_number = int(session.split('ses-')[1])  
    if half == 'odd' and session_number % 2 != 0:
        half_sessions.append(session_number)
    elif half == 'even' and session_number % 2 == 0:
        half_sessions.append(session_number)

print("Half sessions:", half_sessions)

print("Concatenating sessions...")
files_to_concat=RR.collect_files_to_concat(files_subjects, half_sessions) # need to implement loop with halves
all_data_concat=RR.concat_files(files_to_concat)
# filename=os.path.join(network_dir, f'{half}_half', f'sub-{subject}'+ f'_{half}' + f'_{sequenceName}_concatenated_smoothed0.85_masked_32k_fsLR.dtseries.nii')
# all_data_concat = RR.load_data(filename)
n_vertices=all_data_concat.shape[1]

print("Extracting FPN data...")
FPN_file=os.path.join(network_dir, 'whole_dataset', f'FPN.dscalar.nii')
FPN_data=RR.get_network(all_data_concat, FPN_file, remove_rest=True)

# # find columns with only zeros
# print("deleting non-FPN columns...")
# zero_columns = np.where(np.all(FPN_data == 0, axis=1))[0]
# print("Columns with only zeros:", zero_columns) # should be 52214
# FPN_data = np.delete(FPN_data, zero_columns, axis=1)
# print("Shape after removing columns with only zeros:", FPN_data.shape) # should be 7198 columns

# now we have a huge np array with the timeseries for all runs from all odd sessions
# now z-score the data

print("Z-scoring data...")
z_data = RR.z_score_np(FPN_data)
print("Shape after z-scoring:", z_data.shape)

del all_data_concat, FPN_data # save memory

# k means operates on rows so we need to transpose the data
z_data = z_data.T
print("Shape after transposing:", z_data.shape)

"""
k-means
"""

n_clusters = range(1,4) # test 1 to 4 clusters

# save cluster results
# map to cifti
run = '02' # or use a valid run identifier if needed
dtseries_template = os.path.join(timeseries_dir, 'ses-04', 'postfmriprep', 'GLM', f'sub-{subject}_ses-04_task-{sequenceName}_dir-{run}_cleaned.dtseries.nii')
output_filename = os.path.join(output_dir, f'sub-{subject}_{half}_FPN_kmeans_clusters.dtseries.nii')

cluster_results, inertia, silhouette_scores, kmeans_list = RR.kmeans_standard(n_clusters, z_data, save_to_file=True, remap_to_verts=False, filename=output_filename, mask_file=FPN_file, dtseries_template=dtseries_template)


"""
Plot results: Elbow method
"""

filename=os.path.join(output_dir, f'sub-{subject}_{half}_elbow_method.png')
RR.elbow_plot(n_clusters, inertia, filename)

"""
Plot results: Silhouette score

"""

filename=os.path.join(output_dir, f'sub-{subject}_{half}_silhouette_score.png')
RR.silhouette_plot(silhouette_scores, filename)



"""
Plot results: entropy, BIC, n in smallest class
"""


for k in n_clusters:
    print('')
    print("Number of clusters:", k)
    labels = cluster_results[k-1,:]

    entropy = RR.compute_entropy(labels)
    bic = RR.compute_bic(kmeans_list[k-1], z_data)
    smallest_size = RR.smallest_cluster_size(labels)

    print("Entropy:", entropy)
    print("BIC:", bic)
    print("Smallest Cluster Size:", smallest_size)