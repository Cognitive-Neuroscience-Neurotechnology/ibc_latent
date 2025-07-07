import os
import numpy as np
import nibabel as nib
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.axes import Axes
import sys
import re
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import subprocess
import pandas as pd
from scipy.stats import mode
import scipy.ndimage as ndi
import multiprocessing as mp
from scipy.spatial.distance import pdist, squareform
#import layer_analysis 
from collections import defaultdict
from itertools import combinations



script_dir='/ptmp/hmueller2/Downloads/fmriprep_out' ### change this to your script directory!!  

"""
Directory where scripts are located
"""




"""
Reliability functions (mostly from Matilde Vaghi)

"""

def get_sessions_dirs(input_dir:str):
    """
    Get session directories in a given input directory. Session directories are identified by starting with "ses-".

    Args:
        input_dir (str): input directory
    
    Returns:
        list: list of session directories
    
    Example:
        get_sessions_dirs('/ptmp/hmueller2/Downloads/fmriprep_out/sub-01')
        # Output: ['ses-00', 'ses-03', 'ses-04', 'ses-05', 'ses-06', 'ses-07', ...]
    
    """
    session_dirs = sorted([d for d in os.listdir(input_dir) if d.startswith('ses-') and os.path.isdir(os.path.join(input_dir, d))])
    print("Found sessions:", session_dirs)
    return session_dirs

def collect_files_to_concat(list_files_sub:list, list_sess_to_concat:list):
    """
    Collect files to concatenate based on a list of the session numbers.

    Args:
        list_files_sub (list): list of files to concatenate (all sessions)
        list_sess_to_concat (list): list of session numbers

    Returns:
        list: list of files to concatenate

    Example:
        collect_files_to_concat(['ses-3/file_01.nii', 'ses-4/file_01.nii', 'ses-5/file_01.nii', 'ses-6/file_01.nii', 'ses-7/file_01.nii'], [3, 5, 7])
        # Output: ['ses-3/file_01.nii', 'ses-5/file_01.nii', 'ses-7/file_01.nii']
    
    """

    files_to_concat=[]
    for ses in list_sess_to_concat:
        session_search="ses-"+str(ses)
        # prob need to go into ses folder here !!!!!
        for file_name in list_files_sub:
                if session_search in file_name:
                    files_to_concat.append(file_name)
                
    return(files_to_concat)      

def concat_files(list_files_to_concat:list): 
    """
    Concatenate files in a list. Files have to be nifti or txt files.

    Args:
        list_files_to_concat (list): list of files to concatenate
    
    Returns:
        np.array: concatenated data

    Example:
        concat_files(['ses-3/file_01.nii', 'ses-5/file_01.nii', 'ses-7/file_01.nii'])
        # Output: np.array of concatenated data

    """


    temp =[]
    
    for file in list_files_to_concat:
        file_extension = os.path.splitext(file)[1]
        if file_extension == '.nii':
            # print("reading this file:", file)
            
            # file_name=os.path.join(timeseries_dir, file)
            file_img=nib.load(file)
            file_data = file_img.get_fdata()
            temp.append(file_data)
        elif file_extension=='.txt':
            # print("reading this file:", file)
            file_data = np.loadtxt(file, dtype=float, encoding='utf-8')
            temp.append(file_data)
        else:
            raise ValueError("The file extension is not '.nii' or '.txt'.")
        
    all_data_concat = np.concatenate(temp, axis=0)

    # if len(temp) > 0 and all(arr.shape[1:] == temp[0].shape[1:] for arr in temp):
    #     all_data_concat = np.concatenate(temp, axis=0)
    # else:
    #     raise ValueError("Inconsistent dimensions or empty list for concatenation.")
    
    del temp #bc the arrays are very memory intensive
    return(all_data_concat)
   
def r_to_z(r:np.ndarray|pd.DataFrame):
    """
    Convert correlation values to z-scores.

    Args:
        r (np.array): correlation values

    Returns:
        np.array: z-scores
    
    
    """

    # clip in case too close to one
    # EPSILON = 1e-10
    # r = np.clip(r, -1 + EPSILON, 1 - EPSILON)

    # fisher transform
    if isinstance(r, np.ndarray):
        z=0.5*np.log((1.0+r)/(1.0-r))
        z[np.where(np.isinf(z))]=0
        z[np.where(np.isnan(z))]=0

    elif isinstance(r, pd.DataFrame):
        z = 0.5 * np.log((1.0 + r) / (1.0 - r))
        z = z.replace([np.inf, -np.inf], np.nan)

    return z

def z_to_r(z:np.ndarray):
    """
    Convert z-scores to correlation values.

    Args:
        z (np.array): z-scores

    Returns:
        np.array: correlation values
    """

    # inverse transform
    return (np.exp(2.0*z) - 1)/(np.exp(2.0*z) + 1)

def corr_matrix(timeseries_np:np.ndarray, get_upp_mtx=False):

    """
    Compute the correlation matrix for a given timeseries. With option to extract only the upper matrix 

    Args:
        timeseries_np (np.array): timeseries data
        get_upp_mtx (bool): whether to extract only the upper matrix (default: False)
    
    Returns:
        np.array: correlation matrix or upper matrix

    Example:
        corr_matrix(timeseries_np, get_upp_mtx=True)
        # Output: upper matrix of the correlation matrix as np.array
    
    """

    #Compute correlation matrix and get upper matrix
    masked_data=np.ma.masked_invalid(timeseries_np)
    mtx_correlation_np = np.ma.corrcoef(masked_data, rowvar=False)
    mtx_correlation_np = np.ma.filled(mtx_correlation_np, np.nan)

    #print('Shape correlation mtx:',mtx_correlation.shape)
    if get_upp_mtx:
        upp_mtx = mtx_correlation_np[np.triu_indices(mtx_correlation_np.shape[0], k = 1)]
        # triu indices is a function made just for the extraction of the upper triangle
        return(upp_mtx)
    else:
        return(mtx_correlation_np)

def get_upp_mtx_corr(timeseries_parcels:np.ndarray): 

    """
    Compute the upper matrix of the correlation matrix for a given timeseries. 
    Only to preserve backwards compatibility despite new function corr_matrix.

    Args:
        timeseries_parcels (np.array): timeseries data

    Returns:
        np.array: upper matrix of the correlation matrix
    
    """

    # for backwards compatibility
    upp_mtx=corr_matrix(timeseries_parcels, get_upp_mtx="Yes")
    return(upp_mtx)

def get_upp_mtx_corr_z_transformed(timeseries_parcels:np.ndarray): 

    """
    Compute the upper matrix of the correlation matrix for a given timeseries and z-transform it.

    Args:
        timeseries_parcels (np.array): timeseries data

    Returns:
        np.array: z-transformed upper matrix of the correlation matrix
    """



    #Get number of NAN values 
    #print('NaN values:', timeseries_parcels.isna().sum().sum())
    #Compute correlation matrix and get upper matrix
    mtx_correlation_np = np.corrcoef(timeseries_parcels, rowvar=False)
    
    #print('Shape correlation mtx:',mtx_correlation.shape)
    upp_mtx = mtx_correlation_np[np.triu_indices(mtx_correlation_np.shape[0], k = 1)]
    
    #R to z transform
    upp_mtx_z_transformed = r_to_z(upp_mtx)
    return(upp_mtx_z_transformed)

def get_portion_data(data:np.ndarray, TR:float, split:float):

    """
    Get the start and end indices for a specified amount of data from a timeseries.

    Args:
        data (np.array): timeseries data
        TR (float): repetition time (seconds)
        split (float): split time in minutes (e.g., 1 min, 5 min)

    Returns:
        dict: dictionary with start and end indices of each segment of data
    
    Example:
        get_portion_data(data, TR=1.04, split=5)
        # Output: {0: [0, 150], 1: [0, 300], 2: [0, 450], 3: [0, 600], 4: [0, 750], 5: [0, 900], ... }

    """

    dict_start_end={}
    #Take incrementally higher number of data from data
    # 235 is the lenght of a run. -> step
    # TR is 1.49 therefore 235 corresponds to increment of 5.8 minutes
    
    # TR 1.04 s -> 5 min (300s) -> 288.46 time points
    # round down to make this more conservative
    # step=math.floor(300/TR)
    # print(step)

    # for 1 min
    split_s=split*60
    step=math.floor(split_s/TR)
    # print(step)


    vector_boundaries = np.arange(0, len(data), step)

    for nloop, d in enumerate(vector_boundaries): 
        if  vector_boundaries[nloop] == vector_boundaries[-1]:
            #end = vector_boundaries[nloop] + step
            pass
        else:
            start = vector_boundaries[0]
            end =  vector_boundaries[nloop + 1]
            dict_start_end[nloop]=[start,end] # indented
        
    return dict_start_end


"""
Layer reliability functions
"""
def clean_region_name(name):
    """
    Remove the prefix (everything before the first '_') and the suffix (everything after the last '_').
    TODO: optimize if prefix contains multiple underscores (as is the case for large networks)

    Args:
        name (str): region name
    
    Returns:
        str: cleaned region name

    Example:
        clean_region_name('2_L_p24pr_57')
        # Output: 'L_p24pr'
    
    """
    parts = name.split("_")
    if len(parts) > 2:
        return "_".join(parts[1:-1])  # Remove first and last part
    return name

def extract_parcels(all_data_concat:np.ndarray|pd.DataFrame, coverage_file:str, remove_all_zero=False, verbose=False):
    """

    Extracts parcels with sufficient coverage (at least 70%) from the concatenated data based on a coverage file.

    Args:
        all_data_concat (np.array or pd.DataFrame): concatenated data. If DF, the extraction will happen based on parcel names, not indices.
        coverage_file (str): coverage file (a txt file with a column containing the percent covered for each parcel of the atlas)
        verbose (bool): whether to print additional information (default: False)
        remove_all_zero (bool): whether to remove columns that are all zero, currently only possible for DataFrame inputs (default: False)

    Returns:
        np.array: concatenated data with parcels with sufficient coverage

    Example:
        extract_parcels(all_data_concat, 'coverage.txt')
        # Output: concatenated data with parcels with sufficient coverage

    """

    print("Removing ROIs without sufficient coverage.")
    if isinstance(all_data_concat, pd.DataFrame):
        print("Shape of original data", len(all_data_concat.columns))

    # read coverage txt file
    txt_file = np.genfromtxt(
    coverage_file, 
    delimiter=",", 
    skip_header=1,  # skip the header row
    dtype=None,  # automatically determine the data types
    encoding='utf-8',  # handle non-ASCII characters if needed
    comments="#"  # ignore lines starting with '#'
    )

    coverage=txt_file['f1'] # access the second column for Percent Coverage
    roi_names=txt_file['f0'] # access the first column for ROI names
    
    invalid_roi_names=[]
    valid_rois = [] # store indices

    for roi in range(360):
        percent_coverage = coverage[roi] # get percent coverage for this roi

        if percent_coverage >= 0.7:
            valid_rois.append(roi) # get index
        else:
            invalid_roi_names.append(roi_names[roi])
    
    if isinstance(all_data_concat, np.ndarray):
        # filter all_data_concat to only include valid_rois
        all_data_concat_new = all_data_concat[:, valid_rois]
    else:
        rois_to_remove = [s.replace('_ROI', '') for s in invalid_roi_names] # remove '_ROI' from the names bc absent in the dataframe roi names
        all_data_concat_new = all_data_concat.copy()

        for uncovered_roi in rois_to_remove:            
            normalized_columns = {col: clean_region_name(col) for col in all_data_concat.columns}
            matching_columns = [col for col, base_name in normalized_columns.items() if base_name == uncovered_roi]

            all_data_concat_new = all_data_concat_new.drop(columns=matching_columns)

            if verbose:
                print(f"Removing {uncovered_roi}...")
                print("Matching columns:", matching_columns)

        if remove_all_zero:
            columns_to_remove = [col for col in all_data_concat_new.columns if (all_data_concat_new[col] == 0).all()]
            all_data_concat_new.drop(columns=columns_to_remove, inplace=True)
            
            for col in columns_to_remove:
                print(f"Removed column: {col}")

            
    del all_data_concat # free up memory
    print("Shape of new data", all_data_concat_new.shape)
    return all_data_concat_new

def generate_generic_suffix(hemi_suffix:str):

    """
    Generates a generic suffix based on the hemisphere suffix. 
    Used to robustly name output files that are not hemisphere-specific but derived from hemisphere-specific files.

    Args:
        hemi_suffix (str): suffix containing hemisphere information

    Returns:
        str: generic suffix

    Example:
        generate_generic_suffix('run-01_L_layer_1.txt')
        # Output: 'run-01_layer_1.txt'
    
    """


    if '_L_' in hemi_suffix:
        parts = hemi_suffix.split('_L_')
        marker = '_L_'
    elif '_R_' in hemi_suffix:
        parts = hemi_suffix.split('_R_')
        marker = '_R_'
    elif '_L' in hemi_suffix:
        parts = hemi_suffix.split('_L')
        marker = '_L'
    elif '_R' in hemi_suffix:
        parts = hemi_suffix.split('_R')
        marker = '_R'
    else:
        raise ValueError("The suffix must contain '_L_', '_R_', '_L', '_R'.")

    if len(parts) == 2:
        surrounding_part = parts[0] + '_' + parts[1]
    else:
        if parts[0].endswith('_'):
            surrounding_part = parts[0][:-1]
        else:
            surrounding_part = parts[0]

    return(surrounding_part)

def concat_R_L(file_L:str|np.ndarray, file_R:str|np.ndarray, interleaved=False, averaged=False, output_dir:str=None, keep_indices:dict=None):
    """
    Concatenates right and left hemispheric txt timeseries files.

    Args:
        file_L (str): left hemisphere file
        file_R (str): right hemisphere file
        interleaved (bool): whether to concatenate interleaved (default: False)
        averaged (bool): whether to concatenate averaged (default: False)
        output_dir (str): output directory (default: None)
        keep_indices (dict): indices to keep, with separate keys 'L' & 'R' containing column indices as a list (optional)

    Returns:
        str: new filename for the concatenated data

    """

    if isinstance(file_L, str):
        
        # get suffix
        match = re.search(r'run-\d+_(.+)\.txt$', file_L)
        if match:
            suffix_L = match.group(1)
        else:
            raise ValueError("The filename does not contain a valid 'run-<number>_' pattern.")

        new_suffix=generate_generic_suffix(suffix_L)

        # load data
        run_L=np.loadtxt(file_L, dtype=float, encoding='utf-8')
        run_R=np.loadtxt(file_R, dtype=float, encoding='utf-8')

        # print("Shape of left and right runs:", run_L.shape, run_R.shape)
    else:
        run_L=file_L
        run_R=file_R
    
    if keep_indices is not None:
        run_L = run_L[:, keep_indices['L']]
        run_R = run_R[:, keep_indices['R']]
        print("Shape of left and right runs after keeping indices:", run_L.shape, run_R.shape)

    # concatenate horizontally
    if interleaved:
        stacked_3d=np.stack((run_L, run_R), axis=-1)
        run_RL=stacked_3d.reshape(run_L.shape[0], -1)
    elif averaged:
        run_RL = np.mean(np.stack((run_L, run_R), axis=2), axis=2)
    else:
        run_RL=np.concatenate((run_L, run_R), axis=1)

    # save concatenated runs
    if isinstance(file_L, str):
        if output_dir is not None:
            new_filename=os.path.join(output_dir, os.path.basename(file_L).replace(suffix_L, new_suffix))
        else:
            new_filename=os.path.join(os.path.dirname(file_L), os.path.basename(file_L).replace(suffix_L, new_suffix))

        np.savetxt(new_filename, run_RL)

        return new_filename
    else:
        return run_RL

def concat_runs(dir:str, filenamebase:str, suffix_L:str, suffix_R:str, interleaved=False, averaged=False, new_dir=False, keep_indices:dict=None):
    """
    Concatenates all runs within a session.
    This works for the txt files for layer ROI timeseries.
    suffixes should not contain file extensions

    Args:
        dir (str): directory containing the runs
        filenamebase (str): base filename for the runs
        suffix_L (str): suffix for the left hemisphere files (necessary to find files and construct generic suffix)
        suffix_R (str): suffix for the right hemisphere files (necessary to find files)
        interleaved (bool): whether to concatenate interleaved (default: False)
        averaged (bool): whether to concatenate hemispheres averaged (default: False)
        new_dir (bool): whether to save the concatenated files in a new directory (default: False)
        keep_indices (dict): indices to keep, with separate keys 'L' & 'R' containing column indices as a list (optional)
    
    Returns:
        str: filename of the concatenated data

    """
    print("Concatenating runs for ", filenamebase, "...")

    run_pattern = re.compile(r'run-(\d+)')

    run_list=[]

    if new_dir:
        output_dir=os.path.join(dir, 'concatenated')
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir=None


    for f in os.listdir(dir):
        if suffix_L in f: # assumes if L is present, R is also present (this prevents duplicate entries, since each run has two files)
            current_run = int(run_pattern.search(f).group(1))
            run_L_filename=os.path.join(dir, filenamebase + '_run-' + str(current_run) + '_' + suffix_L + '.txt')
            run_R_filename=os.path.join(dir, filenamebase + '_run-' + str(current_run) + '_' + suffix_R + '.txt')

            run_RL=concat_R_L(run_L_filename, run_R_filename, interleaved, averaged, output_dir=output_dir, keep_indices=keep_indices)
            # run_RL is a filename
            run_list.append(run_RL)

    # get new suffix
    generic_suffix = generate_generic_suffix(suffix_L)

    # concatenate runs
    run_list=sorted(run_list)
    all_runs=concat_files(run_list)

    # save concatenated runs
    if new_dir:
        filename=os.path.join(output_dir, filenamebase + '_concatenated_' + generic_suffix + '.txt')
    else:
        filename=os.path.join(dir, filenamebase + '_concatenated_' + generic_suffix + '.txt')
    np.savetxt(filename, all_runs)
    
    return filename

"""
Network Layer Analysis
"""

def mtx_corr(timeseries_parcels:np.ndarray|pd.DataFrame, upper_triangle=True, ignore_NaNs=False): 
    """
    Compute the correlation matrix for a given timeseries. With option to extract only the upper matrix and ignore NaNs.

    Args:
        timeseries_parcels (np.ndarray or pd.DataFrame): timeseries data
        upper_triangle (bool): whether to extract only the upper matrix (default: True)
        ignore_NaNs (bool): whether to ignore NaNs (default: False) (only works for np.arrays)
    
    Returns:
        np.array: correlation matrix or upper matrix

    """
    if isinstance(timeseries_parcels, pd.DataFrame):
        correlation_matrix = timeseries_parcels.corr()
        
        if upper_triangle:
            correlation_matrix = correlation_matrix.values[np.triu_indices(correlation_matrix.shape[0], k = 1)]            

        return correlation_matrix
    if ignore_NaNs==True: # this is useful if there are regions that have 0 std -> lead to NaN in correlation matrix
        masked_data=np.ma.masked_invalid(timeseries_parcels)
        mtx_correlation_np = np.ma.corrcoef(masked_data, rowvar=False)
        mtx_correlation_np = np.ma.filled(mtx_correlation_np, np.nan)
    else:
        mtx_correlation_np = np.corrcoef(timeseries_parcels, rowvar=False)

    print('Number of NaNs in the correlation matrix:', np.count_nonzero(np.isnan(mtx_correlation_np))) # sanity check - should only be for sub-p26 for one region (i.e. for 719 cells)
    # note: above comment is only true when using this function to create parcel x parcel correlations

    #mtx_correlation_np = np.nan_to_num(mtx_correlation_np, nan=0.0)

    if upper_triangle==True:
        corr_mtx = mtx_correlation_np[np.triu_indices(mtx_correlation_np.shape[0], k = 1)]
        # triu indices is a function made just for the extraction of the upper triangle
    else:
         corr_mtx=mtx_correlation_np
    
    return(corr_mtx)

def get_layer_corr_matrix(sup_data:np.ndarray, mid_data:np.ndarray, deep_data:np.ndarray, remove_indices=None, averaged=None, networks=None):
    """
    Computes an all-layer to all-layer correlation matrix. Averaged networks are appended to the end of the resulting array, changing the original order! 
    Therefore, new_networks is returned to reflect the new order.

    Args:
        sup_data (np.ndarray): superficial layer data
        mid_data (np.ndarray): middle layer data
        deep_data (np.ndarray): deep layer data
        remove_indices (list): indices to remove from the data (optional)
        averaged (list): networks to average over (optional) (computes averages over networks specified by a prefix)
        networks (list): network names (optional)

    Returns:
        np.ndarray: correlation matrix
        list: new network order
    
    """

    if remove_indices is not None:
        sup_data = np.delete(sup_data, remove_indices, axis=1)
        mid_data = np.delete(mid_data, remove_indices, axis=1)
        deep_data = np.delete(deep_data, remove_indices, axis=1)

        print("removed indices. The new shapes are:", sup_data.shape, mid_data.shape, deep_data.shape)
    
    if averaged is not None:
        if networks is None:
            raise ValueError("networks must be provided if averaging is set.")
        
        insert_indices=[]
        cols_to_remove=[]
        averaged_dict = {key: None for key in averaged}

        for net in averaged:
            print(f"Averaging '{net}'.")
            # figure out where the networks to be averaged are based on whether they start with the specified string
            nets_to_average_indices = [index for index, value in sorted(enumerate(networks), key=lambda x: x[1], reverse=True) if value.startswith(net)]

            cols_to_remove.extend(nets_to_average_indices)
            
            if len(nets_to_average_indices) == 0:
                raise ValueError("No networks found with the specified prefix.")
            elif len(nets_to_average_indices) == 1:
                print(f"Only one network found with the prefix '{net}'. Averaging skipped.")
            else:
                # average the specified columns
                sup_avg = np.mean(sup_data[:, nets_to_average_indices], axis=1)
                mid_avg = np.mean(mid_data[:, nets_to_average_indices], axis=1)
                deep_avg = np.mean(deep_data[:, nets_to_average_indices], axis=1)

                # assign to dict
                averaged_dict[net] = (sup_avg, mid_avg, deep_avg)

                # insert the net column at the position of the first column in nets_to_average_indices
                insert_indices.append(sorted(nets_to_average_indices)[0])

        # remove the original columns to be averaged
        sup_data = np.delete(sup_data, cols_to_remove, axis=1)
        mid_data = np.delete(mid_data, cols_to_remove, axis=1)
        deep_data = np.delete(deep_data, cols_to_remove, axis=1)

        print(sup_data.shape, mid_data.shape, deep_data.shape)
        print(averaged_dict['Default'][0].shape)
        
        # new average is added onto the end
        for net in averaged:
            print(f'Inserting averaged {net} network...')
            sup_data = np.hstack((sup_data, averaged_dict[net][0][:, np.newaxis]))
            mid_data = np.hstack((mid_data, averaged_dict[net][1][:, np.newaxis]))
            deep_data = np.hstack((deep_data, averaged_dict[net][2][:, np.newaxis]))

        print(f'Averaged {averaged} networks for all layers. New shapes are:', sup_data.shape, mid_data.shape, deep_data.shape)
            
    
        # reorder networks according to this metric
        new_networks = [item for item in networks if not any(item.startswith(prefix) for prefix in averaged)]
        new_networks.extend(averaged)
        print("New network order:", new_networks)

    # if there still are zeros left
    zero_column_dict = {
        'superficial': [],
        'mid': [],
        'deep': []
    }
    zero_column_dict['superficial']= np.where(np.all(sup_data == 0, axis=1))[0]
    zero_column_dict['mid'] = np.where(np.all(mid_data == 0, axis=1))[0]
    zero_column_dict['deep'] = np.where(np.all(deep_data == 0, axis=1))[0]
    
    # some networks may not be present in all layers
    # keep only the networks that are present in all layers
    all_zero_column_indices = set()
    for layer in ['superficial', 'mid', 'deep']: 
        # assume zero_column_indices for the current layer is available
        zero_column_indices = zero_column_dict[layer]
        # add the elements of zero_column_indices to the set (duplicates will be ignored)
        all_zero_column_indices.update(zero_column_indices)
    # convert the set back to a list
    all_zero_column_indices_list = sorted(list(all_zero_column_indices), reverse=True)
    print("Unique zero_column_indices in at least one layer:", all_zero_column_indices_list)

    sup_data = np.delete(sup_data, all_zero_column_indices_list, axis=1)
    mid_data = np.delete(mid_data, all_zero_column_indices_list, axis=1)
    deep_data = np.delete(deep_data, all_zero_column_indices_list, axis=1)

    print("new shapes:", sup_data.shape, mid_data.shape, deep_data.shape)

    assert sup_data.shape == mid_data.shape == deep_data.shape, "Data shapes do not match."

    # for the all-layer to all-layer, we want the three layers to be interleaved in the columns 
    stacked_3d=np.stack((sup_data, mid_data, deep_data), axis=-1)
    result=stacked_3d.reshape(sup_data.shape[0], -1)
    print(result.shape)

    correlation_matrix=mtx_corr(result, upper_triangle=False, ignore_NaNs=True)
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
    
    if averaged is not None:
        return correlation_matrix, new_networks
    else:
        return correlation_matrix

def plot_corr_matrix(data:np.ndarray, title_string:str,  overarching_indices, regressed=False, z_scored=False, grey_diagonal=True):
    """
    Plots a heatmap of the correlation matrix with evenly spaced tick labels.
    NOT ROBUST: optimised for individual_nets_layer_conn.py and individual_nets_layer_conn_averaged.py
    
    Args:
        data: 2D numpy array or pandas DataFrame (correlation matrix)
        title_string: Title for the plot
        overarching_indices: List of overarching indices (e.g., network names) for the plot
        z_scored: Boolean indicating whether the data is z-scored (optional)
        grey_diagonal: Boolean indicating whether to grey out the diagonal blocks (optional)

    Returns:
        plt.figure: the plot
   
    """

    if data.shape[0] != data.shape[1]:
        # non-square matrix for plotting averaged across networks
        n_networks, n_layers = data.shape  

        overarching_indices_x = ["superficial", "middle", "deep"]
        overarching_indices_y=overarching_indices


    else:
        # square matrix: networks x networks
        n_layers=3
        overarching_indices_x = overarching_indices
        overarching_indices_y = overarching_indices

    cbar_legend = "z(r)" if z_scored else "r"

    max_val=np.max(data)
    min_val=np.min(data)
    abs_max = max(abs(min_val), abs(max_val))
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)


    n_cols = data.shape[1]

    if grey_diagonal:
        # grey out the diagonal blocks (for entire networks) by setting it to NaN
        data_no_diag = np.copy(data)
        num_networks = n_cols // n_layers # 3 layers -> n networks 

        for network_idx in range(num_networks):
            start = network_idx * n_layers
            end = start + n_layers

            data_no_diag[start:end, start:end] = np.nan
        
        max_val=np.nanmax(data_no_diag)
        min_val=np.nanmin(data_no_diag)
        abs_max = max(abs(min_val), abs(max_val))
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

        heatmap=sns.heatmap(data_no_diag, 
                    mask=np.isnan(data_no_diag),
                    annot=False, cmap="RdYlBu_r",
                    xticklabels=overarching_indices, yticklabels=overarching_indices, 
                    cbar_kws={'label': cbar_legend},
                    norm=norm,  # use the custom normalization
                    vmin=-abs_max, 
                    vmax=abs_max)

    else:
        max_val=np.max(data)
        min_val=np.min(data)
        abs_max = max(abs(min_val), abs(max_val))
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

        heatmap=sns.heatmap(data, annot=False, cmap="RdYlBu_r",
                    xticklabels=overarching_indices_x, 
                    yticklabels=overarching_indices_y, 
                    cbar_kws={'label': cbar_legend},
                    norm=norm,  # use the custom normalization
                    vmin=-abs_max, 
                    vmax=abs_max)
    
    # customise colorbar
    cbar = heatmap.collections[0].colorbar
    tick_values = np.linspace(-abs_max, abs_max, num=6)  # 6 evenly spaced ticks
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{tick:.2f}" for tick in tick_values])

    ax = plt.gca()
    
    if data.shape[0] == data.shape[1]:
        # square matrix
        tick_positions = np.arange(n_layers / 2, data.shape[1], step=n_layers)
        line_positions = np.arange(0, data.shape[1], step=n_layers)

        for line in line_positions:
            ax.axvline(x=line, color='black', linewidth=0.5) # vertical grid lines
            ax.axhline(y=line, color='black', linewidth=0.5) # horizontal grid lines

        plt.xticks(tick_positions, overarching_indices_x, rotation=90, ha='center', fontsize=6)
        plt.yticks(tick_positions, overarching_indices_y, rotation=0, ha='right', fontsize=6)

    

    plt.title(title_string)
    plt.tight_layout()
    plt.subplots_adjust(left=0.2, bottom=0.3, right=0.9, top=0.9)

    return plt.gcf()

def save_plot(plot: plt.Figure | Axes, filename:str):
    """
    Saves a plot to a file.

    Args:
        plot: matplotlib plot object
        filename: output filename

    Returns:
        None
    
    """


    plot.savefig(filename, dpi=300)
    plt.close(plot)
    print("Saved plot as", filename)

def regress_out_layer(target_layer:np.ndarray|str, *predictor_layer:np.ndarray|str, df:pd.DataFrame=None, verbose=False):
    """
    Regress out the effects of one or more predictor arrays from a target array.

    Args:
        target_layer: layer that should be cleaned (np.array or string if dataframe is provided)
        predictor_layer: one or more predictor arrays (np.array or string if dataframe is provided)
        df: dataframe with timeseries (instead of np.arrays)
        verbose: whether to print additional information (default: False)

    Returns:
        adjusted_target: np.ndarray (The target array with the effects of predictors regressed out.).
                        or pd.DataFrame (if df is provided)
    
    """
    if isinstance(target_layer, np.ndarray):
        n_samples = target_layer.shape[0]
        if not all(predictor.shape[0] == n_samples for predictor in predictor_layer):
            raise ValueError("All predictors and the target must have the same number of rows (time points).")
        
        adjusted_target = np.copy(target_layer)
        n_columns = target_layer.shape[1]

        model = LinearRegression()

        for i in range(n_columns):
            # stack all predictor columns for this specific target column
            stacked_predictors = np.column_stack([pred[:, i] for pred in predictor_layer])
            
            # fit the model on the predictors and the target column
            model.fit(stacked_predictors, target_layer[:, i])
            
            # predict the part of the target explained by the predictors
            predicted_signal = model.predict(stacked_predictors)
            
            # subtract the predicted signal from the target column
            adjusted_target[:, i] = target_layer[:, i] - predicted_signal

        return adjusted_target
    elif isinstance(target_layer, str):
        if df is None:
            raise ValueError("A dataframe must be provided if the target layer is a string.")

        # initialize
        model = LinearRegression()

        # identify target columns by looking for columns ending with the target_layer
        target_columns = [col for col in df.columns if col.endswith(f'_{target_layer}')]

        # create a copy of the dataframe to store adjusted values
        adjusted_df = df.copy()

        for target_column in target_columns:
            # get prefix of target (e.g., '1_L_FEF_10')
            prefix = target_column.rsplit(f'_{target_layer}', 1)[0]

            # initialize a list for matching predictor columns
            matching_predictors = []

            for predictor in predictor_layer:
                # construct predictor column name (e.g., '1_L_FEF_10_mid')
                predictor_column = f'{prefix}_{predictor}'

                # check if the constructed predictor column exists in the dataframe
                if predictor_column in df.columns:
                    matching_predictors.append(df[predictor_column].values)
                    if verbose:
                        print(f"For {target_column} found predictor column {predictor_column}.")
                else:
                    raise ValueError(f"Predictor column {predictor_column} not found in the dataframe.")

            # stack predictors horizontally for regression
            stacked_predictors = np.column_stack(matching_predictors)
            
            # get target column values for regression (target_layer should be reshaped for use)
            target_values = df[target_column].values

            # fit model on stacked predictors and target column
            model.fit(stacked_predictors, target_values)

            # predict part of target explained by predictors
            predicted_signal = model.predict(stacked_predictors)

            # subtract predicted signal from target column to regress it out
            adjusted_df[target_column] = target_values - predicted_signal
        
        # return adjusted dataframe with only modified columns
        return adjusted_df[target_columns].rename(columns={col: col.rsplit(f'_{target_layer}', 1)[0] for col in adjusted_df[target_columns].columns})

def average_correlation_layer_with_others(corr_matrix:np.ndarray, network_index:int, layer_index:int, num_layers:int, num_networks:int):
    """
    Computes the average correlation of a layer of one network with all layers of other networks. (Helper function)

    Args:
        corr_matrix (np.ndarray): correlation matrix
        network_index (int): index of the current network
        layer_index (int): index of the current layer
        num_layers (int): number of layers
        num_networks (int): number of networks

    Returns:
        float: average correlation

    """
   
    start_row = network_index * num_layers + layer_index
    print(f"Network {network_index+1}, Layer {layer_index+1} (row {start_row})")
    
    # initialize list to store correlations
    correlations = []
    
    # loop through all networks except the current one
    for other_network in range(num_networks):
        if other_network == network_index:
            continue  # skip self-correlation
        
        # indices of layers for the other network
        other_start_index = other_network * num_layers
        other_end_index = other_start_index + num_layers

        print(f"Correlating with Network {other_network+1}, Layer indices {other_start_index+1}-{other_end_index}")
        
        # add correlations between the current layer and all layers of the other network
        correlations.extend(corr_matrix[start_row, other_start_index:other_end_index])
    
    # compute the average correlation
    avg_corr = np.mean(correlations)
    
    return avg_corr

def get_network_wise_layer_corr_matrix(sup_data:np.ndarray, mid_data:np.ndarray, deep_data:np.ndarray, networks:list, remove_indices=None, averaged=None):

    """
    Computes a network-wise layer to layer correlation matrix. This means that the correlation matrix represents the correlation of one network with an average of all other networks. 
    Averaged networks are appended to the end of the resulting array, changing the original order!
    
    Args:
        sup_data (np.ndarray): superficial layer data
        mid_data (np.ndarray): middle layer data
        deep_data (np.ndarray): deep layer data
        networks (list): network names
        remove_indices (list): indices to remove from the data (optional)
        averaged (list): networks to average over (optional) (computes averages over networks specified by a prefix)

    Returns:
        np.ndarray: correlation matrix
        list: new network order

    """

    correlation_matrix, new_networks=get_layer_corr_matrix(sup_data, mid_data, deep_data, remove_indices, averaged, networks)
    np.fill_diagonal(correlation_matrix, 1.0)  # self-correlation is 1.0
    print("Correlation matrix shape:", correlation_matrix.shape)

    correlation_matrix = r_to_z(correlation_matrix)

    num_networks=len(new_networks)
    num_layers=3 #TODO: make this more robust later

    # prepare results array
    results = np.zeros((num_networks, num_layers))

    # loop through networks and layers
    for network in range(num_networks):
        for layer in range(num_layers):

            # compute average correlation of that layer with all other networks' layers
            avg_corr = average_correlation_layer_with_others(
            correlation_matrix, network, layer, num_layers, num_networks
            )

            print(f"Network {network+1}, Layer {layer+1} has an average correlation of {avg_corr:.2f}")
            
            # store in array
            results[network, layer] = avg_corr

    print("Results shape:", results.shape)
    print(results)

    results=z_to_r(results)

    return results, new_networks

"""
Functions for k-means
"""

def concat_WB(files_to_concat:list, output_filename:str, return_data=True):

    """
    Uses wb_command to concatenate files along time dimension.

    Args:
        files_to_concat (list): list of files to concatenate
        output_filename (str): output filename
        return_data (bool): whether to return the concatenated data (default: True)

    Returns:
        np.array: concatenated data (if return_data is True)
    
    """

    files_with_prefix = []
    for f in files_to_concat:
        files_with_prefix.extend(["-cifti", f])
        
    subprocess.run(["wb_command", "-cifti-merge", output_filename] + files_with_prefix, check=True)

    if return_data:
        img=nib.load(output_filename)
        all_data_concat = img.get_fdata()

        return all_data_concat
    else:
        return None

def which_runs(session_dir:str, pattern:re.Pattern):

    """
    Get a list of runs based on a pattern.

    Args:
        session_dir (str): session directory
        pattern (re.Pattern): regular expression pattern

    Returns:
        list: list of runs (sorted ascending)
    
    """

    run_list=[]
    for f in os.listdir(session_dir):
        match=pattern.match(f)
        if match:
            run_number=match.group(1)
            run_list.append(int(run_number))

    if run_list == []:
        raise ValueError("No runs found for the specified pattern.")
    
    print("Runs found:", sorted(run_list))

    return sorted(run_list)

def z_score_np(array:np.ndarray):

    """
    Z-scores a numpy array.

    Args:
        array (np.array): input array

    Returns:
        np.array: z-scored array

    """

    # compute the mean and standard deviation along rows for each column
    mean = np.mean(array, axis=0)  # mean for each column
    std = np.std(array, axis=0)    # std for each column

    # avoid division by zero by replacing zero std with a small value
    zero_stds_count = np.sum(std == 0)
    print(f"Number of zero standard deviations: {zero_stds_count}")

    std[std == 0] = 1e-8

    # Z-score the array
    zscored_data = (array - mean) / std
    return zscored_data

def get_network(data:np.ndarray | str, mask:str | np.ndarray, remove_rest=False):
    """
    inspired by Denis Chaimow's layer_analysis.mask_image in fmri-analysis
    changed: handling different input dataypes + doesn't return a nifti image object but the data array

    Args:
        data (np.ndarray): data to be masked (np array or path to file)
        mask (str or np.ndarray): ROI / network mask (currently as str leading to a dscalar.nii or directly as binary array)

    Returns:
        np.array: masked data array

    """
    print("Getting network for mask", mask)

    if type(data) is str:
        img = nib.load(data)
        data = img.get_fdata()
    elif type(data) is np.ndarray:
        pass
    else:
        raise ValueError(f"Data must be a path to a file or a numpy array but is {type(data)}.")

    if type(mask) is str:
        mask = nib.load(mask)
        mask_data = mask.get_fdata()
    else:
        mask_data = mask

    img_masked_data = data * (mask_data != 0)

    # delete non-network columns
    if remove_rest:
        non_zero_columns = (mask_data != 0).flatten()
        print(non_zero_columns)
        img_masked_data = img_masked_data[:, non_zero_columns]


    print("Masked data shape:", img_masked_data.shape)
    return img_masked_data

def nib_save(filename:str, data:np.ndarray, template:str):
    """
    For saving into a dtseries.nii, ptseries.nii, or dscalar.nii file. For dlabel.nii use write_dlabel.

    Args:
        filename: output filename to save to
        data: np array to save
        template: a cifti template (same cifti type as output file)

    Returns:
        None
    """

    template_img=nib.load(template)
    header=template_img.header
    axes=[header.get_axis(i) for i in range(template_img.ndim)]

    if template.endswith("dtseries.nii"):
        time_axis, brain_model_axis=axes
        new_time_axis=time_axis[0:data.shape[0]]
        # header.get_axis(0)=time_axis

        new_dtseries=nib.Cifti2Image(data, header=(new_time_axis, brain_model_axis), nifti_header=template_img.nifti_header)
        new_dtseries.to_filename(filename)

    elif template.endswith("dscalar.nii"):
        scalaraxis=axes[0]
        brain_model_axis=axes[1]
        if data.shape[0] != scalaraxis.name.shape[0]:
            newscalaraxis=nib.cifti2.ScalarAxis(['Map'+str(i) for i in range(data.shape[0])])
        else:
            newscalaraxis=scalaraxis
        new_dscalar=nib.Cifti2Image(data, header=(newscalaraxis, brain_model_axis), nifti_header=template_img.nifti_header)
        new_dscalar.to_filename(filename)

    elif template.endswith("ptseries.nii"):
        time_axis, parcelsaxis=axes
        new_time_axis=time_axis[0:data.shape[0]]
        # header.get_axis(0)=time_axis

        new_ptseries=nib.Cifti2Image(data, header=(new_time_axis, parcelsaxis), nifti_header=template_img.nifti_header)
        new_ptseries.to_filename(filename)
    else:
        raise ValueError(f"File extension is not recognised by the function. Should be dtseries, dscalar or ptseries, but is {template}.")

    print("Saved as", filename)

    return None

def unmask(data_to_unmask:np.ndarray, output_filename:str, og_mask:str | np.ndarray, dtseries_template:str, remap_to_verts=False, ignore_values=None, ids=None):
    """
    Unmasks data row by row into the original mask shape. (Used when masking FPN to save back into a cifti.)

    Args:
        data_to_unmask: data to be unmasked - array or column vector (np array) 
        output_filename: path to save the unmasked data
        og_mask: original mask used to create the data_to_unmask shape. Can be a path to a file or np.ndarray
        dtseries_template: any dtseries_file to use the header from
        remap_to_verts: remap to vertices - used when data_to_unmask is no longer on vertex level, but e.g., on subnetworks level (default: False)
        ignore_values: values to ignore (use only with remap_to_verts) (default: None) TODO: remove this, unnecessary
        ids: indices that were kept (that correspond to networks or subnetworks) - use only with remap_to_verts (default: None)

    Returns:
        None

    """

    if type(og_mask)==str:
        og_mask=nib.load(og_mask).get_fdata()

    if remap_to_verts:
        n_rows=data_to_unmask.shape[0]
        # create zeros matrix to store remapped results
        result=np.zeros((n_rows,og_mask.shape[1]))
        # loop through entries in the data
        counter=0
        for i in ids:
            print((og_mask == i).shape)
            
            if i==ignore_values:
                print(f"Ignoring value {i}.")
                continue

            # insert the values of all rows in data_to_unmask into mapped_matrix where the og_mask had the value i (e.g., SN)
            result[:, og_mask.flatten() == i] = data_to_unmask[:, counter][:, np.newaxis]
            counter+=1

    else:# make sure 1d
        og_mask = np.squeeze(og_mask)
        if og_mask.ndim != 1:
            raise ValueError(f"Expected 1D mask after squeezing, but got shape {og_mask.shape}")
        
        n_rows = data_to_unmask.shape[0]
        result = np.zeros((n_rows, og_mask.size), dtype=data_to_unmask.dtype)

        # where is the mask not zero
        non_zero_indices = np.flatnonzero(og_mask)

        # fill non-zero locations with data
        for i, row in enumerate(data_to_unmask):
            result[i].ravel()[non_zero_indices] = row

    print("Unmasked shape:", result.shape)    

    nib_save(output_filename, result, dtseries_template)

    return None

def load_data(filename:str, return_img=False):

    """
    Loads data from a file using nibabel. Converts memmap to np.array.

    Args:
        filename (str): filename to load
        return_img (bool): return nibabel image object (default: False)

    Returns:
        np.array: data array
        img: nibabel image object (if return_img is True)

    
    """

    if filename.endswith(".nii" or ".nii.gz"):
        img = nib.load(filename)
        data = img.get_fdata()
        
        if isinstance(data, np.memmap):
            data = np.array(data)
    elif filename.endswith(".gii"):
        img=nib.load(filename)
        data=img.agg_data()
    else:
        raise ValueError("File extension not recognised. Must be .nii or .gii.")


    if return_img:
        return img, data
    else:
        return data

"""
Layering
"""

def format_number(n:int):
    """
    Formats a number to have two digits.

    Args:
        n (int): number

    Returns:
        str: formatted number

    Example:
        format_number(3)
        # Output: '03'

    """

    return f"{n:02d}"

def get_thres(num: int):
    """
    Calc threshold based on number.

    Args:
        num (int): number

    Returns:
        list: thresholds list

    Example:
        get_thres(3)
        # Output: [0, 0.33, 0.66, 1]
    """
    total = 1
    thres = total / num

    # get cumulative values
    result = [round(i * thres, 4) for i in range(num + 1)]
    return result

def layer_data(n_layers, output_dir, depth_file, return_filenames=False):
    """
    Creates an arbitrary number of layers based on a depth file.
    
    Args:
        n_layers (int): number of layers
        output_dir (str): output directory
        depth_file (str): depth file
        return_filenames (bool): return list of filenames

    Returns:
        list: list of filenames

    Example:
        layer_data(3, "dir", "depth.nii")
        # Output: ["dir/layer_01.nii", "dir/layer_02.nii", "dir/layer_03.nii"]
    """

    print(f"Creating {n_layers} layers...")
    
    output_dir = os.path.join(output_dir, f"{n_layers}_layers")
    os.makedirs(output_dir, exist_ok=True)


    # get thresholds
    thresholds = get_thres(n_layers)

    for i in range(n_layers):
        print(f"Creating layer {i+1}...")
        layer_name=format_number(i+1)
        print(f"Lower threshold: {thresholds[i]}, Upper threshold: {thresholds[i+1]}")

        command=["fslmaths",
                        depth_file,
                        "-nan",
                        "-thr", str(thresholds[i]),
                        "-uthr", str(thresholds[i+1]),
                        "-bin",
                        os.path.join(output_dir, f"layer_{layer_name}")]
        
        subprocess.run(command,
                        check=True)
    print("Finished.")
    
    if return_filenames:
        return [os.path.join(output_dir, f"layer_{format_number(i)}") for i in range(1,n_layers+1)]
    else:
        return None

def layer_tseries(epi, atlas_L, atlas_R, output_dir, layer_masks):

    """
    Parcellates a slab volume into layers based on layer masks and an atlas. Requires parcellate_slab_volume_space_generic.sh script from preprocessing folder.

    Args:
        epi (str): epi file to be parcellated
        atlas_L (str): left hemisphere atlas
        atlas_R (str): right hemisphere atlas
        output_dir (str): output directory
        layer_masks (list): list of layer masks (from layer_data)
    
    Returns:
        None

    Example:
        layer_tseries("epi.nii", "atlas_L.nii", "atlas_R.nii", "output_dir", ["layer_01.nii", "layer
        _02.nii", "layer_03.nii"])
        # creates parcellated files in output_dir
    
    """

    print("Running parcellation...")

    subprocess.run(
        [
            'bash', os.path.join(script_dir, 'preprocessing/HR_slab', 'parcellate_slab_volume_space_generic.sh'), 
            epi, 
            atlas_L, atlas_R,
            output_dir,
            *layer_masks
            ])
    
"""
Functions for spider plots and subnetworks.
"""

def get_labels(filename:str, n_map:int=0, return_data=False):
    """
    Get labels from a cifti dlabel.nii file.

    Args:
        filename (str): cifti file
        n_map (int): map number to extract (default: 0)
        return_data (bool): return data object (default: False)

    Returns:
        dict: labels
        data object: data object (if return_data is True)

    Example:
        get_labels("file.dlabel.nii")
        # Output: e.g.,
        array([{0: ('???', (1.0, 1.0, 1.0, 0.0)), 1: ('B', (0.960784, 0.933333, 0.152941, 1.0)), 2: ('A', (0.960784, 0.156863, 0.568627, 1.0)), 3: ('LABEL_3', (0.188235, 0.490196, 0.176471, 1.0)), ...
    
    """

    cifti_img, cifti_data = load_data(filename, return_img=True)

    cifti_header=cifti_img.header # <nibabel.cifti2.cifti2.Cifti2Header object at 0x7fed0e677650>

    cifti_axes = [cifti_header.get_axis(i) for i in range(cifti_img.ndim)] # [<nibabel.cifti2.cifti2_axes.LabelAxis object at 0x7fed0c78bf90>, <nibabel.cifti2.cifti2_axes.BrainModelAxis object at 0x7fed0c74d290>]

    labelaxis=cifti_axes[0] # <nibabel.cifti2.cifti2_axes.LabelAxis object at 0x7fed0c78bf90>

    labels=labelaxis.label

    # each row is one map, containing a dict
    # dict: key is the key in the label table, the first string is the name, followed by RGBA values


    if return_data:
        return labels[n_map], cifti_data
    else:
        return labels[n_map]
    
def get_network_from_dlabel(dlabel_file:str, network:str|int, return_dlabel_img=True, keep_number=None, map_to_use=None):
    """
    Get a single network from a dlabel file.

    Args:
        dlabel_file (str): dlabel file
        network (str or int): network name or index
        return_dlabel_img (bool): return nibabel image object of the dlabel file (default: True)
        keep_number (bool): number to store the output mask with (default is None, which keeps the original index number)

    Returns:
        nibabel image object: dlabel image object (if return_dlabel_img is True)
        np.array: dlabel data of the requested network


    Example:
        get_network_from_dlabel("file.dlabel.nii", "Frontoparietal")
        get_network_from_dlabel("file.dlabel.nii", 9, keep_number=True) # will also return the FPN

    
    """
    labels=get_labels(dlabel_file)

    if isinstance(network, str):
        network_index = [key for key, value in labels.items() if value[0] == network][0]
    else:
        network_index = network

    if return_dlabel_img:
        img, data = load_data(dlabel_file, return_img=return_dlabel_img)
    else:
        data = load_data(dlabel_file, return_img=return_dlabel_img)

    if map_to_use is not None:
        if data.shape[0]==1:
            print("Only one map found. Using that.")
        else:
            data=data[map_to_use,:]
    else:
        if data.shape[0]>1:
            raise ValueError("Multiple maps found. Please specify which one to use.")
        else:
            pass
    

    if keep_number:
        network = np.where(data == network_index, keep_number, 0)
    else:
        network = np.where(data == network_index, network_index, 0)

    print(f"Network {network_index} extracted.")
    print(f'Found {np.sum(data == network_index)} vertices for this network.')

    if return_dlabel_img:
        return img, network
    else:
        return network

def create_mask(data:np.ndarray, value_to_mask_on:int):
    """
    Creates a mask based on a specific value.

    Args:
        data (np.ndarray): data to create mask from
        value_to_mask_on (int): value to mask on

    Returns:
        np.ndarray: mask

    Example:
        create_mask(np.array([[1, 2], [1, 2]]), 1)
    
    """
    masked_array = np.where(data == value_to_mask_on, 1, 0)

    return masked_array

# def mask_data(data:np.ndarray, mask:np.ndarray):
#     """
#     Masks data based on a mask - not to be used for timeseries.

#     Args:
#         data (np.ndarray): data to be masked
#         mask (np.ndarray): mask

# TODO: remove this function, not used  

#     Returns:
#         np.ndarray: masked data

#     Example:
#         mask_data(np.array([[1, 2], [1, 2]]), np.array([[1, 0], [0, 1]]))
    
#     """

#     masked_data = np.where(mask == 1, data, 0)

#     return masked_data

def pickle_save(filename, var):
    with open(filename, 'wb') as f:
        pickle.dump(var, f)
    print('**** File saved ****')

    return None

def write_dlabel(input_cifti:str|np.ndarray, template_file=None|str, label_table=str, out_dlabel=str, discard_others=False, drop_unused_labels=False, map_to_save:int=0):
    """
    Writes a dlabel file based on a cifti file and a label table.

    Args:
        input_cifti (str or np.ndarray): input cifti file (also accepts data as np array)
        template_file (str): template file (default: None) - only use if input_cifti is np.ndarray. Important: this should be a dtseries.nii file
        label_table (str): label table file
        out_dlabel (str): output dlabel file
        discard_others (bool): whether to discard values not found in label table (default: False)
        drop_unused_labels (bool): drop unused labels (default: False)
        map_to_save (int): map to save (default: 0)
    
    Returns:
        None
    
    """
    if map_to_save !=0:
        print(f"Using map {map_to_save} to save.")
        if isinstance(input_cifti, np.ndarray):
            input_cifti=input_cifti[map_to_save,:]
        else:
            data=load_data(input_cifti)
            data=data[map_to_save,:] # now its a 1d array
            data=data[None,:] # make it a 2d array
            
            new_filename=input_cifti.replace(".dscalar.nii", f"_map{map_to_save}.dscalar.nii")
            nib_save(new_filename, data, input_cifti)
            print(f"Saved new dscalar with map {map_to_save} as", new_filename)

            input_cifti=new_filename


    if isinstance(input_cifti, np.ndarray):
        output_dir=os.path.dirname(out_dlabel)
        nib_save(os.path.join(output_dir, 'temp.dtseries.nii'), input_cifti, template_file)
        input_cifti = os.path.join(output_dir, 'temp.dtseries.nii')

    if discard_others:
        if drop_unused_labels:
            subprocess.run(['wb_command', '-cifti-label-import', input_cifti, label_table, out_dlabel, '-discard-others', '-drop-unused-labels'])
        else:
            subprocess.run(['wb_command', '-cifti-label-import', input_cifti, label_table, out_dlabel, '-discard-others'])
    else:
        if drop_unused_labels:
            subprocess.run(['wb_command', '-cifti-label-import', input_cifti, label_table, out_dlabel, '-drop-unused-labels'])
        else:
            subprocess.run(['wb_command', '-cifti-label-import', input_cifti, label_table, out_dlabel])

    print("Saved new dlabel as", out_dlabel)

    try:
        os.remove(os.path.join(output_dir, 'temp.dtseries.nii'))
        print("Removed temporary file.")
    except:
        pass

def add_jitter(angles:list, jitter_strength=0.1):
    """
    Adds jitter to angles. (used for spider plots)

    Args:
        angles (list): list of angles
        jitter_strength (float): jitter strength (default: 0.1)
    
    Returns:
        np.ndarray: jittered angles
    """
    jitter = np.random.uniform(-jitter_strength, jitter_strength, len(angles))
    return np.clip(np.array(angles) + jitter, 0, 2 * np.pi)

"""
Subnetworks functions
"""

def kmeans_standard(n_clusters:list, data:np.ndarray, save_to_file=True, remap_to_verts=False, filename=None, mask_file=None, dtseries_template=None, ignore_values=None, ids=None, switch_dict=None):

    """
    K-means clustering with standard settings.

    Args:
        n_clusters (list): list of number of clusters
        data (np.ndarray): data to cluster (already z_scored)
        save_to_file (bool): save to file (default: True)
        remap_to_verts (bool): remap to vertices (default: False)
        filename (str): output filename if save_to_file(default: None)
        mask_file (str): mask file if save_to_file (default: None)
        dtseries_template (str): dtseries template if save_to_file (default: None)
        ignore_values (list): values to ignore (default: None)
        ids (list): indices that were kept (only if save_to_file) (default: None)
        switch_dict (dict): dictionary to switch values (used to remap numbers and make SNs consistent across subjects) (default: None)
    
    Returns:
        np.ndarray: cluster results
        list: inertia
        list: silhouette scores
        list: kmeans list

    """


    inertia = [] # for elbow method
    silhouette_scores = [] # for silhouette score
    kmeans_list=[]

    cluster_results = np.zeros((len(n_clusters), data.shape[0]))

    print("Running k-means...")
    for k in n_clusters:
        print("Number of clusters:", k)
        # settings similar to MATLAB defaults
        kmeans = KMeans(
            n_clusters=k,         # number of clusters
            init='k-means++',     # MATLAB's default initialization
            n_init=1,             # MATLAB's default number of replicates
            max_iter=100,         # MATLAB's default maximum number of iterations
            tol=1e-4,             # tolerance for convergence
            random_state=0        # for reproducibility (optional)
        )

        # fit the k-means algorithm
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
        kmeans_list.append(kmeans)

        # results
        # print("Cluster Centers:", kmeans.cluster_centers_)
        print("Labels:", kmeans.labels_)

        if k >= 2 and k < data.shape[0]:
            score = silhouette_score(data, kmeans.labels_)
            silhouette_scores.append(score)

        # map labels back to a np array
        cluster_results[k-1,:] = kmeans.labels_
        print(kmeans.n_iter_)

    # adding one to everything so that in the visualisation all of the FPN has a color
    cluster_results_plus_one = cluster_results + 1

    if switch_dict is not None:
        k_to_use=max(max(switch_dict.keys()), max(switch_dict.values()))
        print(f'Remapping k={k_to_use} clusters...')

        result = cluster_results_plus_one.copy()
        row = result[k_to_use-1]
        row_mapped = np.vectorize(lambda x: switch_dict.get(x, x))(row)     
        result[k_to_use-1] = row_mapped

        cluster_results_plus_one = result


    # save cluster results
    # map to cifti
    if save_to_file:
        unmask(cluster_results_plus_one, filename, mask_file, dtseries_template, remap_to_verts=remap_to_verts, ignore_values=ignore_values, ids=ids)

    return cluster_results, inertia, silhouette_scores, kmeans_list

def compute_entropy(labels):
    """
    Compute entropy based on cluster label distribution.

    Args:
        labels (np.ndarray): cluster labels

    Returns:
        float: entropy

    """
    labels = labels.astype(int) 
    n_clusters = len(np.unique(labels))
    total_points = len(labels)
    cluster_counts = np.bincount(labels)  # counts for each cluster
    
    entropy = -np.sum(
        (count / total_points) * np.log(count / total_points)
        for count in cluster_counts if count > 0
    )
    return entropy

def compute_bic(kmeans, data):
    """
    Compute BIC for the k-means clustering result.

    Args:
        kmeans: k-means clustering object
        data: data array
    
    Returns:
        float: BIC
    """
    n = data.shape[0]  # number of data points
    k = kmeans.n_clusters  # number of clusters
    inertia = kmeans.inertia_  # within-cluster sum of squared distances
    
    bic = n * np.log(inertia / n) + k * np.log(n)
    return bic

def smallest_cluster_size(labels):
    """
    Compute the size of the smallest cluster.

    Args:
        labels (np.ndarray): cluster labels

    Returns:
        int: smallest cluster size
    """
    labels = labels.astype(int) 
    cluster_counts = np.bincount(labels)  # counts for each cluster
    return np.min(cluster_counts)

def elbow_plot(n_clusters:list, inertia:list, output_file:str):
    """
    Plot results: Elbow method

    Args:
        n_clusters (list): number of clusters
        inertia (list): inertia values
        output_file (str): output filename

    Returns:
        None
    """

    # plot the Elbow Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_clusters, inertia, marker='o')
    ax.set_title('Elbow Method for Optimal Clusters')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia (Sum of Squared Distances)')
    ax.grid()

    save_plot(fig,output_file)

def silhouette_plot(silhouette_scores, output_file):
    """
    Plot results: Silhouette score

    Args:
        silhouette_scores (list): silhouette scores
        output_file (str): output filename
    
    Returns:
        None

    """

    # plot the Silhouette Scores
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(2, len(silhouette_scores)+2), silhouette_scores, marker='o')
    ax.set_title('Silhouette Analysis for Optimal Clusters')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.grid()

    save_plot(fig,output_file)

"""
Density maps
"""

def density_map(input_dir:str, subject_list:list, filename:str, network:str, label_table:str, template_dtseries:str, map_to_use:int=0, keep_number=None, output_prefix=None, thresholded=False):

    """
    Creates a density map across subjects.

    Args:
        input_dir (str): input directory
        subject_list (list): list of subjects
        filename (str): filename
        network (str): network name to perform density map on
        label_table (str): label table file (to write a cross-subject dlabel for subsequent density maps) 
        template_dtseries (str): any dtseries file (for an intermediate step)
        keep_number (int): number to keep in case of remapping the desired network (default: None)
        output_prefix (str): output prefix (default: None)
        thresholded (bool): threshold the density map so that includes at least 2 subjects (default: False)

    Returns:
        None
    
    """

    # store for later
    network_across_subjects=np.zeros((len(subject_list), 59412)) # TODO: make n vertices more robust

    # loop across subjects
    for subject in subject_list:
        print(f'Processing {subject}...')

        # filenames are templates and can't start with the variable that is only accessed here, so passed as ??? in the beginning.
        if filename.startswith('???'):
            filename_loop=filename.replace('???', subject)
        else:
            filename_loop=filename

        # construct full filename
        network_file=os.path.join(input_dir, subject, filename_loop)
        print('extracting network...')
        
        # for kmeans the dlabel files contain multiple maps, so need to extract which map is relevant for the k iteration
        net = get_network_from_dlabel(network_file, network, return_dlabel_img=False, keep_number=keep_number, map_to_use=map_to_use)

        # assign to array
        network_across_subjects[subject_list.index(subject), :]=net


    # now get back to dlabel
    print('writing dlabel...')
    if output_prefix is not None:
        output_dlabel=os.path.join(input_dir, f'{network.replace(" ", "_").replace("/", "_")}_{output_prefix}_across_subjects_whole_dataset.dlabel.nii')
    else:
        output_dlabel=os.path.join(input_dir, f'{network.replace(" ", "_").replace("/", "_")}_across_subjects_whole_dataset.dlabel.nii')

    write_dlabel(network_across_subjects, template_dtseries, label_table, output_dlabel)


    # now create density map
    print('creating density map...')
    if output_prefix is not None:
        output_density_map=os.path.join(input_dir, f'{network.replace(" ", "_").replace("/", "_")}_{output_prefix}_across_subjects_density_map.dscalar.nii')
        output_thresholded=os.path.join(input_dir, f'{network.replace(" ", "_").replace("/", "_")}_{output_prefix}_across_subjects_density_map_thresholded.dscalar.nii')
    else:
        output_density_map=os.path.join(input_dir, f'{network.replace(" ", "_").replace("/", "_")}_across_subjects_density_map.dscalar.nii')   
        output_thresholded=os.path.join(input_dir, f'{network.replace(" ", "_").replace("/", "_")}_across_subjects_density_map_thresholded.dscalar.nii')

    subprocess.run(['wb_command',
                    '-cifti-label-probability',
                    output_dlabel,
                    output_density_map,
                    '-exclude-unlabeled'])
    
    if thresholded:
        threshold=(1/len(subject_list))*2
        expression = f"x>={threshold}"
        subprocess.run(['wb_command', '-cifti-math',
                        expression,
                        output_thresholded,
                        '-var', 'x', output_density_map])

"""
Functions for individual vs group effects (Gratton et al. 2018 replication)
"""


def create_big_data_mat(all_matrices:list,subject_inds:list, session_inds:list, n_combinations):
    """
    Create a big data matrix for comparison of individual vs group effects.

    Args:
        all_matrices (list): list of FC matrices
        subject_inds (list): list of subjects
        session_inds (list): list of sessions
        n_combinations (int): number of combinations - size of third dimension for comparison of size of big_data_mat

    Returns:
        np.ndarray: big data matrix
        np.ndarray: subject indices
        np.ndarray: session indices

    """

    # print("subject_inds:",subject_inds)
    print("subject indices has this length:", len(subject_inds))
    # print("session_inds:", session_inds)
    print("session indices has this length:", len(session_inds))
    print("the list of matrices has this length:", len(all_matrices))
    # print(matrices)

    if len(subject_inds)!=len(session_inds)!=len(all_matrices):
        raise ValueError("Length of subject_inds, session_inds and all_matrices must be the same.")

    # stack all matrices into a 3D numpy array
    big_data_mat = np.stack(all_matrices, axis=2)  # shape: (num_parcels, num_parcels, num_subject_sessions) -> 360 x 360 x 10 in my case

    # convert subject and session indices to numpy arrays (1d)
    big_data_sub_ind = np.array(subject_inds)
    big_data_sess_ind = np.array(session_inds)

    if big_data_mat.shape[2]==n_combinations:
        print("big_data_mat shape is as expected:", big_data_mat.shape)
        print("big_data_sub_ind shape:", big_data_sub_ind.shape)
        print("big_data_sess_ind shape:", big_data_sess_ind.shape)
    else:
        print("big_data_mat shape is not as expected:", big_data_mat.shape)
        print("big_data_sub_ind shape:", big_data_sub_ind.shape)
        print("big_data_sess_ind shape:", big_data_sess_ind.shape)

    return big_data_mat, big_data_sub_ind, big_data_sess_ind

def flatten_big_data_mat(big_data_mat:np.ndarray, n_combinations:int, n_regions:int):
    """
    Flatten the big data matrix into ROI*(ROI-1)/2 x n_combinations.

    Args:
        big_data_mat (np.ndarray): big data matrix
        n_combinations (int): number of combinations
        n_regions (int): number of regions

    Returns:
        np.ndarray: flattened data
    
    """


    upper_tri_indices = np.triu_indices(big_data_mat.shape[0], k=1)
    num_combinations = len(upper_tri_indices[0]) # number of cells in the upper triangle = number of parcel x parcel combinations
    flattened_data = np.zeros((num_combinations, big_data_mat.shape[2])) 

    for i in range(big_data_mat.shape[2]):
        flattened_data[:, i] = big_data_mat[:, :, i][upper_tri_indices]


    desired_shape_0=n_regions*(n_regions-1)/2
    # https://blogs.sas.com/content/iml/2015/10/19/corr-upper-tri.html
    # should be n(n–1)/2 => (360*359)/2 = 64620 for shape[0]
    # shape[1] should then be the amount of sub ses combinations which is 24

    if flattened_data.shape[0]==desired_shape_0 and flattened_data.shape[1]==n_combinations:
        print("flattened_data shape is as expected :", flattened_data.shape)
    else:
        print("at least one of the dimensions of flattened_data is not as expected:", flattened_data.shape)

    return flattened_data

def extract_cols_for_networks(network_indices:list, columns_to_parcels:pd.DataFrame):
    """
    Extract columns for network indices.

    Args:
        network_indices (list): list of network indices
        columns_to_parcels (pd.DataFrame): columns to parcels dataframe

    Returns:
        np.ndarray: region columns
    
    """

    # extract regions in the network through the reordered columns to parcels 
    ROIs_df=columns_to_parcels[columns_to_parcels["Network"].isin(network_indices)] # in this case columns means columns in the np array, i.e., regions
    # ROIs should be a df that is a subset of the rows of columns_parcels_sorted df 

    # extract just the column indices we need from the sorted matrices
    rois_columns=ROIs_df["Column"].values
    print("Number of regions in network group:",rois_columns.shape)

    return rois_columns

def similarity_workflow(matrices:list, subject_inds:list, session_inds:list, n_combinations:int, n_regions:int, z_scored=True):

    """
    Calculate the similarity for plotting.

    In the original code (https://github.com/MidnightScanClub/MSC_Gratton2018_Codebase/blob/master/similarity_analyses/similarity_figs_SPLIThalf.m)
    Gratton et al implement different similarity measures.

    In their paper, they say that they computed correlations between the upper triangles: https://www.sciencedirect.com/science/article/pii/S0896627318302411?via%3Dihub#sectitle0260
    Here, I thus implement correlation but we can consider other metrics.

    Calculate average similarity (across all functional networks??) for each effect:
        - group (different individuals, sessions)
        - individual (same sub, diff session)
        - individual & session (same sub, same session)

    Their process step by step:
    1. parcel x parcel x condition matrix (where condition represents every possible combination of subject, session and task)
    2. (extract subnetworks - only relevant for the control vs processing plots)
    3. correlate "conditions" across parcel pairs -> 2d matrix of conditions x conditions (where the values inside each cell are collapsed across all parcel, parcel pairs)
    4. z-transform the matrix
    5. mask for different types of effects (e.g. same subject, same session, diff task etc)

    Note: Theres is a discrepancy: the description of Fig 3A claims to look at pairs of networks, but based on their code, they do not do that.

    Args:
        matrices (list): list of matrices
        subject_inds (list): list of subject indices
        session_inds (list): list of session indices
        n_combinations (int): number of combinations
        n_regions (int): number of regions
        z_scored (bool): z-score the correlations (default: True)
    
    Returns:
        np.ndarray: correlation across combinations

    """

    
    big_data_mat, big_data_sub_ind, big_data_sess_ind = create_big_data_mat(matrices, subject_inds, session_inds, n_combinations)


    # Step 3: correlate conditions (third dimension);
    # for this they:

    # a. linearise the upper triangle into a vector
    flattened_data=flatten_big_data_mat(big_data_mat, n_combinations, n_regions)

    # b. transpose -> rows = parcel, parcel combinations, columns = conditions
    # not necessary -> already incorporated into the flattening

    # c. correlate across columns -> conditions x conditions matrix 
    correlation_across_combinations=mtx_corr(flattened_data, upper_triangle=False, ignore_NaNs=True)
    print(correlation_across_combinations.shape) # should be 10 x 10 


    # d. z-transform
    if z_scored==True:
        print("z-scored correlations:")
        correlation_across_combinations_z=r_to_z(correlation_across_combinations)
        print(correlation_across_combinations_z.shape)
        return correlation_across_combinations_z

    elif z_scored==False:
        return correlation_across_combinations

def file_processing(derivative_dir:str, sequenceName:str, subject_list:list, run_list=None, split_half=True, consider_runs=False):
    """
    Process files for the similarity workflow.

    Args:
        derivative_dir (str): derivative directory
        sequenceName (str): sequence name
        subject_list (list): list of subjects
        run_list (list): list of runs (default: None)
        split_half (bool): split half (default: True)
        consider_runs (bool): consider runs (default: False)

    Returns:
        dict: complete data or first and second half data

    """

    ses_pattern = re.compile(r'ses-(\d+)')

    if split_half==True: # in Gratton et al. they use split half with first and second half of the sessions to aggregate
        first_half_data={}
        second_half_data={}
    elif split_half==False: # for session and run specific
        complete_data={}

    # need way to store individual
    for subject in subject_list:
        print(subject)

        sessions_list=[]
        filenames=[]
        if split_half==False:
            sub_data={}

        timeseries_dir=os.path.join(derivative_dir, "extracted_tseries_parc", subject)
        session_dirs = sorted([d for d in os.listdir(timeseries_dir) if d.startswith('ses-') and os.path.isdir(os.path.join(timeseries_dir, d))])

        for dir in session_dirs:
            session = int(ses_pattern.search(dir).group(1)) # number of session

            # we have internal values of sessions as 3,4,5,6,7 -> should start at 1
            new_session=session-2

            sessions_list.append(new_session)
            current_dir = os.path.join(timeseries_dir, dir)

            # not the cleanest, but this will be overwritten if consider_runs
            file=os.path.join(current_dir, f'{subject}'+ f'_ses-{session}' + f'_{sequenceName}' + '_concat.ptseries.nii')
            filenames.append(file)

            if split_half==False and consider_runs==False: # all sessions
                    cur_data=load_data(file)
                    sub_data[new_session]=cur_data

            elif consider_runs==True: # run specific
                session_data={}
                for run in run_list:
                    file=os.path.join(current_dir, f'{subject}'+ f'_ses-{session}' + f'_{sequenceName}' + f'_run-{run}LR.Glasser.ptseries.nii')
                    filenames.append(file)
                    cur_data=load_data(file)
                    session_data[run]=cur_data

                sub_data[new_session]=session_data

        if split_half==True:
            if len(sessions_list) % 2 == 0: # one sub has four sessions
                    first_half_data[subject] = concat_files(filenames[0:2])
            else:
                    first_half_data[subject] = concat_files(filenames[0:3])
            second_half_data[subject] = concat_files(filenames[-2:])
        
        if split_half==False:
            complete_data[subject]=sub_data

        print('Number of sessions analyzed for this subject:', len(sessions_list))
    
    if split_half==True:
        return first_half_data, second_half_data
    elif split_half==False:
        return complete_data
    
def calculate_FD_P(in_file:str, rot_type='degrees'):
    """
    Method to calculate Framewise Displacement (FD) calculations
    (Power et al., 2012)
    Function written by Romy Lorenz

    Args:
        in_file (str): movement parameters vector file path
        rot_type (str): rotation type (default: 'degrees')

    Returns:
        out_file : string: Frame-wise displacement mat file path
    
    """
    
    #out_file = os.path.join(os.getcwd(), 'FD.1D') 

    lines = open(in_file, 'r').readlines()
    rows = [[float(x) for x in line.split()] for line in lines]
    cols = np.array([list(col) for col in zip(*rows)])
    
    # this is the same as in FSL
    translations = np.transpose(np.abs(np.diff(cols[3:6, :])))

    #from Power et al., 2012: radius 50 mm = approximately the mean distance from the cerebral cortex to the center of the head
    radius_mm = 50

    if rot_type=='radians':
        rotations_radians = np.transpose(np.abs(np.diff(cols[0:3, :], axis=1)))
        FD_power = np.sum(translations, axis=1) + radius_mm * np.sum(rotations_radians, axis=1)

    elif rot_type=='degrees':
        rotations_degrees = np.transpose(np.abs(np.diff(cols[0:3, :], axis=1)))
        degrees2mm = radius_mm * np.pi / 180 #convert degrees to millimeters using a 50mm radius
        FD_power = np.sum(translations, axis=1) + degrees2mm * np.sum(rotations_degrees, axis=1)

    
    #FD is zero for the first time point
    FD_power = np.insert(FD_power, 0, 0)
    
    #np.savetxt(out_file, FD_power)
    
    #return out_file
    return FD_power


"""
Functions for subnetworks: Analyses
"""

def wb_find_clusters(output_dir:str, network_to_get:str, ciftify_dir:str):
    """
    Find clusters in a roi file. Uses 20 mm^2 as a surface threshold as in D'Andrea et al., 2023.

    Args:
        output_dir (str): output directory
        network_to_get (str): network name
        ciftify_dir (str): ciftify directory
    
    Returns:
        str: output file

    """

    if "/" in network_to_get:
        output_file=os.path.join(output_dir, f'{network_to_get.replace("/", "_")}_clusters.dscalar.nii')
        network_to_get_underscore=network_to_get.replace("/", "_")
    else:
        output_file=os.path.join(output_dir, f'{network_to_get}_clusters.dscalar.nii')
        network_to_get_underscore=network_to_get
    
    # find clusters
    subprocess.run(['wb_command', '-cifti-find-clusters',
                    os.path.join(output_dir, f'{network_to_get_underscore}_roi.dscalar.nii'),
                    '0', '20', '0', '0',
                    'COLUMN',
                    output_file,
                    '-left-surface', os.path.join(ciftify_dir, 'freesurfer/MNINonLinear/fsaverage_LR32k/freesurfer.L.midthickness.32k_fs_LR.surf.gii'),
                    '-right-surface', os.path.join(ciftify_dir, 'freesurfer/MNINonLinear/fsaverage_LR32k/freesurfer.R.midthickness.32k_fs_LR.surf.gii')
                    ])

    return output_file

def wb_label_to_roi(dlabel_in:str, output_dir:str, network_to_get:str):
    """
    Converts a network from a dlabel to a dscalar for masking.

    Args:
        dlabel_in (str): dlabel input file
        output_dir (str): output directory
        network_to_get (str): network name of the network to get

    Returns:
        str: output filename

    """

    if "/" in network_to_get:
        output_file=os.path.join(output_dir, f'{network_to_get.replace("/", "_")}_roi.dscalar.nii')
    else:
        output_file=os.path.join(output_dir, f'{network_to_get}_roi.dscalar.nii')
        

    result=subprocess.run(['wb_command', '-cifti-label-to-roi',
                dlabel_in,
                output_file,
                '-name', network_to_get], capture_output=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr.decode()}")
        return output_file, False
    else:
        print(f"ROI saved to {output_file}")
        return output_file, True

def reconstruct_row_vector_from_labels(segmentation_matrix):

    """
    Reconstruct a row vector from a labels matrix (from the watershed algorithm)

    Args:
        segmentation_matrix (np.ndarray): segmentation matrix

    Returns:
        np.ndarray: row vector
    
    """

    # create an empty vector to store the label for each vertex
    row_vector = np.zeros(segmentation_matrix.shape[0], dtype=int)

    # for each vertex (corresponding to each row), find the most frequent label
    for i in range(segmentation_matrix.shape[0]):
        # cet the i-th row or column, and find the most frequent label
        mode_result = mode(segmentation_matrix[i, :])  # mode returns an array of most frequent

        if isinstance(mode_result[0], np.ndarray):
            # if it's an array (multiple modes), take the first mode
            most_frequent_label = mode_result[0][0]
        else:
            # if it's a scalar (single mode), just use the scalar
            most_frequent_label = mode_result[0]

        row_vector[i] = most_frequent_label  # assign this most frequent label to the row vector

    return row_vector

def run_watershed(dscalar_in:str, map_to_use:int, func_data:np.ndarray):

    """
    Run watershed algorithm on a dscalar file.

    Args:
        dscalar_in (str): dscalar input file
        map_to_use (int): map to use
        func_data (np.ndarray): functional data

    Returns:
        np.ndarray: watershed result

    """    
    

    img, data=load_data(dscalar_in, return_img=True)
    data=data[map_to_use-1,:]

    # get connectivity matrix of subnetwork
    mask = (data > 0).astype(int)
    print(f'found {np.sum(mask)} vertices for this network')
    func_data_sn = get_network(func_data, mask, remove_rest=True)
    FC_matrix=corr_matrix(func_data_sn)

    # extract only positive correlations
    FC_matrix = np.where(FC_matrix > 0, FC_matrix, 0)

    # distance matrix
    distance_matrix = 1 - FC_matrix
    print(f"Distance matrix shape: {distance_matrix.shape}")

    # local minima
    local_minima = (distance_matrix == ndi.minimum_filter(distance_matrix, size=2))
    markers, _ = ndi.label(local_minima)

    # apply watershed
    labels = ndi.watershed_ift((distance_matrix * 255).astype(np.uint8), markers.astype(np.int32))
    print('Watershed done.')
    print(f"Labels shape: {labels.shape}")

    # go back to row vector form
    vector=reconstruct_row_vector_from_labels(labels)

    print(f'Converted labels matrix to a vector: {vector.shape}')
    print(f'Found {len(np.unique(vector))} unique labels.')

    # remap to start at 1 and increase consecutively
    unique_values = np.unique(vector)
    value_mapping = {val: i+1 for i, val in enumerate(unique_values)}
    renumbered_matrix = np.vectorize(value_mapping.get)(vector)

    print(f'After renumbering vector, found {len(np.unique(renumbered_matrix))} unique labels.')

    return renumbered_matrix

def wb_cifti_rois_from_extrema(cifti_file, surf_limit, output_file, left_surface, right_surface):
    """
    Not tested.
    """

    subprocess.run(['wb_command', '-cifti-rois-from-extrema',
                    cifti_file,
                    surf_limit, '0', 'COLUMN',
                    output_file,
                    '-left-surface', left_surface,
                    '-right-surface', right_surface,
                    '-overlap-logic', 'CLOSEST'])
    
def nib_get_axes(img:object):
    """
    Gets the axes of a nibabel image object.

    Args:
        img (nibabel image object): image object

    Returns:
        list: axes

    Example:
        nib_get_axes(ptseries_img)
        [<nibabel.cifti2.cifti2_axes.SeriesAxis object at 0x7f10f28e3fd0>, <nibabel.cifti2.cifti2_axes.ParcelsAxis object at 0x7f10f28e3e20>]

    """
    header=img.header
    axes=[header.get_axis(i) for i in range(img.ndim)]

    return axes

def remove_medial_wall(dtseries:str, ciftify_dir:str, resolution='32k'):
    """
    Removes the medial wall from a dtseries.nii or dlabel.nii or dscalar.nii using the medial wall ROI from the ciftify directory

    Args:
        dtseries (str): dtseries file
        ciftify_dir (str): ciftify directory
        resolution (str): resolution (default: '32k')

    Returns:
        str: output file name

    Example: 
        remove_medial_wall('sub-01.dtseries.nii', '/path/to/ciftify', resolution='32k')
        file='sub-01_no_medial_wall.dtseries.nii'

    """

    mw_dir=os.path.join(ciftify_dir, f'freesurfer/MNINonLinear/fsaverage_LR{resolution}')
    left_mw=os.path.join(mw_dir, f'freesurfer.L.atlasroi.{resolution}_fs_LR.shape.gii')
    right_mw=os.path.join(mw_dir, f'freesurfer.R.atlasroi.{resolution}_fs_LR.shape.gii')

    if dtseries.endswith('.dtseries.nii'):
        output_file=dtseries.replace('.dtseries.nii', '_no_medial_wall.dtseries.nii')
    elif dtseries.endswith('.dlabel.nii'):
        output_file=dtseries.replace('.dlabel.nii', '_no_medial_wall.dlabel.nii')
    elif dtseries.endswith('.dscalar.nii'):
        output_file=dtseries.replace('.dscalar.nii', '_no_medial_wall.dscalar.nii')
        print('Code is not optimized for this file type. Attempting medial wall removal.')
    else:
        raise ValueError("Input file must be .dtseries.nii or .dlabel.nii")

    subprocess.run(['wb_command', '-cifti-restrict-dense-map',
                    dtseries,
                    'COLUMN',
                    output_file,
                    '-left-roi', left_mw,
                    '-right-roi', right_mw])
    
    return output_file

def parc_covered_vertices(vertices_per_roi:dict, roi_indices:list, data:np.ndarray, to_file=False, template_file:str=None, subnetwork_name:str=None, total_rois:int=360):
    """
    Parcellates vertex-based timeseries using only vertices covered by an ROI (not the whole ROI)

    Args:
        vertices_per_roi (dict): vertices per ROI
        roi_indices (list): ROI indices
        data (np.ndarray): data
        to_file (bool): save to file (default: False)
        template_file (str): template file (default: None)
        subnetwork_name (str): subnetwork name (default: None)
        total_rois (int): total number of ROIs (default: 360)
    
    Returns:
        np.ndarray: output data
        str: new filename (if to_file=True)
    
    """
    output_data=np.zeros((data.shape[0], total_rois))
    for roi in roi_indices:
        vertices_for_this_roi=vertices_per_roi[roi]

        if type(vertices_for_this_roi)==set:
            vertices_for_this_roi=list(vertices_for_this_roi)

        output_data[:, roi]=np.mean(data[:, vertices_for_this_roi], axis=1)
    
    if to_file:
        if "/" in subnetwork_name:
            subnetwork_name=subnetwork_name.replace("/", "_")
        new_filename=template_file.replace('.ptseries.nii', f'_subnetwork_{subnetwork_name}.ptseries.nii')
        nib_save(new_filename, output_data, template_file)

        # remove zero columns
        output_data=output_data[:, roi_indices]
        return output_data, new_filename
    else:
        output_data=output_data[:, roi_indices]
        return output_data

def subnetwork_rois_from_atlas(atlas:str, subnetwork_mask:str|np.ndarray, map_to_use:int|None, subnetwork_name:str, vertex_data:np.ndarray, threshold:int=20, to_dlabel=False, dtseries_template:str=None, dlabel_filename:str=None):

    """
    Get the ROIs for a subnetwork using an atlas, and a threshold of at least 20 vertices to be covered.

    Args:
        atlas (str): can be passed either as ptseries parcellated using the desired atlas.
        subnetwork_mask (str|np.ndarray): subnetwork mask (dscalar.nii file or np.ndarray)
        map_to_use (int|None): map to use in the subnetwork_mask file (default: None) - 1 based indexing
        subnetwork_name (str): subnetwork name
        vertex_data (np.ndarray): vertex data
        threshold (int): threshold (default: 20)
        to_dlabel (bool): save to dlabel (default: False)
        dtseries_template (str): dtseries template file - needs to be provided if to_dlabel (default: None) 
        dlabel_filename (str): dlabel filename (default: None)

    Returns:
        list: all_roi_vertices: stores the number of vertices in each ROI
        list: covered_vertices: stores the number of vertices in each ROI that are covered by the subnetwork
    """

    if isinstance(subnetwork_mask, str):
        subnetwork_mask = load_data(subnetwork_mask)
        try:
            subnetwork_mask=subnetwork_mask[map_to_use-1,:]
        except:
            raise ValueError("Specify which map to use or provide mask as np.ndarray.")

    # get vertices of the subnetwork 
    subnetwork_vertices=list(np.nonzero(subnetwork_mask)[0])

    img, atlas_data = load_data(atlas, return_img=True)

    # ptseries data is on region level, not vertex level, so need mapping
    axes=nib_get_axes(img)
    parcelsaxis=axes[1]
    all_vertices=parcelsaxis.vertices 
    # vertices is an np.ndarray (regions,) big. Each region is stored in one of the entries. within it is a dict with an array containing the vertices of that ROI
    regions = len(all_vertices)  # total number of regions

    # shift the vertices of the right half
    for i in range(regions // 2, regions):
        for key in all_vertices[i]:
            all_vertices[i][key] += 32492

    all_roi_vertices = [] # n vertices per ROI
    covered_vertices=[] # n vertices per ROI covered by SN
    roi_names=[] # names of covered ROIs
    roi_indices=[] # indices of covered ROIs (zero-based)

    vertices_covered_dict={}

    if to_dlabel:
        new_data=np.zeros((1, subnetwork_mask.shape[0]))
        print(new_data.shape)
        network_and_indices={}

    for roi in range(all_vertices.shape[0]):
        these_vertices=list(next(iter(all_vertices[roi].values()))) # extracts np.ndarray of that roi as list

        vertices_in_SN = len(set(these_vertices) & set(subnetwork_vertices)) # only stores the amount of vertices covered
        actual_vertices=set(these_vertices) & set(subnetwork_vertices)

        if vertices_in_SN>=threshold:
            roi_name=parcelsaxis.name[roi]

            all_roi_vertices.append(len(these_vertices))
            covered_vertices.append(vertices_in_SN)

            roi_names.append(roi_name)
            roi_indices.append(roi)

            vertices_covered_dict[roi]=actual_vertices

            if to_dlabel:
                new_roi_name=subnetwork_name + '_' + roi_name
                new_data[:, list(actual_vertices)]=roi+1
                network_and_indices[new_roi_name]=roi+1

    print(f'Found {len(covered_vertices)} regions for this subnetwork')
    print(roi_names)

    # mask ptseries to only include the rois in the subnetwork
    new_tseries, new_ptseries_name = parc_covered_vertices(vertices_covered_dict, roi_indices, vertex_data, to_file=True, template_file=atlas, subnetwork_name=subnetwork_name)
    
    if to_dlabel:
        label_table_file=os.path.join(os.path.dirname(atlas), 'label_table.txt')
        generic_lable_table(label_table_file, network_and_indices=network_and_indices)
        write_dlabel(new_data, dtseries_template, label_table=label_table_file, out_dlabel=dlabel_filename, drop_unused_labels=True)
        os.remove(label_table_file)
    
    return all_roi_vertices, covered_vertices, new_tseries, new_ptseries_name
    
def cifti_resample(input_cifti:str, output_file:str, template:str):
    """
    Resample a cifti file to a template. (only tested for dlabel files)

    Args:
        input_cifti (str): input cifti file
        output_file (str): output file
        template (str): template file
    
    Returns:
        None
    
    """

    if input_cifti.endswith('.dlabel.nii'):
        subprocess.run(['wb_command', '-cifti-resample',
                        input_cifti, 'COLUMN',
                        template, 'COLUMN',
                        'ADAP_BARY_AREA', 'ENCLOSING_VOXEL',
                        output_file]) 
    elif input_cifti.endswith('dtseries.nii') or input_cifti.endswith('dscalar.nii'):
        subprocess.run(['wb_command', '-cifti-resample',
                        input_cifti, 'COLUMN',
                        template, 'COLUMN',
                        'ADAP_BARY_AREA', 'CUBIC',
                         output_file])

def get_colors(filename:str, n_map:int=0):
    """
    Get colours from a cifti dlabel.nii file.

    Args:
        filename (str): cifti file
        n_map (int): map number to extract (default: 0)

    Returns:
        dict: list of touples with colours

    Example:
        get_colors("file.dlabel.nii")
        # Output: e.g.,
        {'B': (0.960784, 0.933333, 0.152941, 1.0), 'A': (0.960784, 0.156863, 0.568627, 1.0)...}
    """

    labels=get_labels(filename,n_map)

    colors = {value[0]: value[1] for value in labels.values()}
    return colors

def compute_geodesic_distance(surface, vertex_index, output_file):
    """
    Compute geodesic distance from a vertex on a surface.

    Args:
        surface (str): surface file
        vertex_index (int): vertex index
        output_file (str): output file
    
    Returns:
        np.ndarray: geodesic distance
    
    """

    command = [
        'wb_command', "-surface-geodesic-distance", surface, str(vertex_index), output_file
    ]
    subprocess.run(command, shell=False)
    data = load_data(output_file)
    os.remove(output_file)
    return data

def process_vertex(args):
    """
    Process a vertex for geodesic distance computation.

    Args:
        args (tuple): arguments

    Returns:
        np.ndarray: geodesic distance
    
    """
    vertex, surface, out_dir = args
    temp_file = f"{out_dir}/tmp/temp_{vertex}.shape.gii"
    os.makedirs(os.path.dirname(temp_file), exist_ok=True)
    return compute_geodesic_distance(surface, vertex, temp_file)

def surf_distance_matrix(ref_cifti, midthick_surfs, out_dir, n_workers):
    """
    Compute surface distance matrix.

    Args:
        ref_cifti (str): reference cifti file
        midthick_surfs (list): midthickness surfaces
        out_dir (str): output directory
        n_workers (int): number of workers
    
    Returns:
        np.ndarray: distance matrix

    """
    os.makedirs(f"{out_dir}/tmp", exist_ok=True)
    
    # Load reference CIFTI
    if isinstance(ref_cifti, str):
        ref_cifti = nib.load(ref_cifti)
    
    ref_cifti_data = ref_cifti.get_fdata()
    ref_cifti_data[:] = 0  # Remove data

    # Load midthickness surfaces
    LH = nib.load(midthick_surfs[0]).darrays[0].data
    RH = nib.load(midthick_surfs[1]).darrays[0].data
    
    LH_verts = np.arange(len(LH))
    RH_verts = np.arange(len(RH))
    
    # compute geodesic distances
    with mp.Pool(n_workers) as pool:
        lh_results = pool.map(process_vertex, [(v, midthick_surfs[0], "lh") for v in LH_verts])
        rh_results = pool.map(process_vertex, [(v, midthick_surfs[1], "rh") for v in RH_verts])
    
    lh_matrix = np.array(lh_results, dtype=np.uint8).T
    rh_matrix = np.array(rh_results, dtype=np.uint8).T
    
    # Merge left and right hemisphere matrices
    top = np.hstack([lh_matrix, np.full((lh_matrix.shape[0], rh_matrix.shape[1]), 999, dtype=np.uint8)])
    bottom = np.hstack([np.full((rh_matrix.shape[0], lh_matrix.shape[1]), 999, dtype=np.uint8), rh_matrix])
    D = np.vstack([top, bottom])
    
    np.save(f"{out_dir}/DistanceMatrixCortexOnly.npy", D)
    
    # # compute Euclidean distances
    # coords_surf = np.vstack([LH, RH])
    # D2 = squareform(pdist(coords_surf))
    # D2 = np.uint8(D2)
    
    # # Merge Euclidean and geodesic distances
    # D = np.vstack([D, D2[D.shape[0]:, :D.shape[1]]])
    # D = np.hstack([D, D2[:D.shape[0], D.shape[1]:]])
    
    # np.save(f"{out_dir}/DistanceMatrix.npy", D)
    
    print("Distance matrices saved successfully.")

    return D

def between_ROI_distance_matrix(distance_matrix, vertex_groups):
    """
    Modify the distance matrix to keep only between-ROI distances.

    Args:
        distance_matrix (ndarray): The full distance matrix.
        vertex_groups (dict): Dictionary where keys are ROI names, values are lists of vertex indices.

    Returns:
        modified_matrix (ndarray): Distance matrix with only between-group distances.
    """
    pass

def surf_data_from_cifti(data, axis, surf_name):
    """
    Function from https://github.com/effigies/nibabel-presentations/blob/ef3addf947004ca8f5610f34e767a578c4934c09/NiBabel.py#L768-L814

    Extracts surface data from a CIFTI file.

    Args:
        data (ndarray): The data from the CIFTI file.
        axis (Cifti2BrainModelAxis): The BrainModelAxis from the CIFTI file.
        surf_name (str): The name of the surface to extract.

    Returns:
        surf_data (ndarray): The data from the named surface
    
    """

    assert isinstance(axis, nib.cifti2.BrainModelAxis)
    for name, data_indices, model in axis.iter_structures():  # Iterates over volumetric and surface structures
        if name == surf_name:                                 # Just looking for a surface
            data = data.T[data_indices]                       # Assume brainmodels axis is last, move it to front
            vtx_indices = model.vertex                        # Generally 1-N, except medial wall vertices
            surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
            surf_data[vtx_indices] = data
            return surf_data
    raise ValueError(f"No structure named {surf_name}")

def cifti_separate(img, map_to_use:int=1):
    """
    Separates cifti file into left and right hemisphere. Like wb_command -cifti-parcellate but into arrays not gifti files.
    From https://github.com/effigies/nibabel-presentations/blob/ef3addf947004ca8f5610f34e767a578c4934c09/NiBabel.py#L768-L814
    But added map_to_use argument

    Args:
        img (nibabel image object): cifti image object
        map_to_use (int): map to use (default: 0)   

    Returns:
        left hemisphere data, right hemisphere data

    """

    data = img.get_fdata(dtype=np.float32)
    data=data[map_to_use-1,:]
    brain_models = img.header.get_axis(1)  # Assume we know this
    return surf_data_from_cifti(data, brain_models, "CIFTI_STRUCTURE_CORTEX_LEFT"), surf_data_from_cifti(data, brain_models, "CIFTI_STRUCTURE_CORTEX_RIGHT")

def cifti_parcellate(data_to_parc:str|np.ndarray, atlas:str, output_file=None, interleaved=True, map_to_use:int=1):

    """
    Parcellate a cifti file: either as dtseries using -cifti-parcellate or when passing np.ndarray as Left and Right hemispheres separately.

    Args:
        data_to_parc (str|np.ndarray): data to parcellate
        atlas (str): atlas file
        output_file (str): output file - only use when data_to_parc is a str (default: None) 

    Returns:
        np.ndarray: parcellated data or output file name
    """

    if isinstance(data_to_parc, str):
        layer_analysis.dtseries_parcellate(data_to_parc, atlas, output_file)
        return output_file
    
    img, atlas_data=load_data(atlas, return_img=True)
    left, right=cifti_separate(img, map_to_use=map_to_use)
    left = left.flatten()
    right = right.flatten()

    # get networks
    unique_values_L = np.unique(left)
    unique_values_L = unique_values_L[unique_values_L != 0] 

    unique_values_R = np.unique(right)
    unique_values_R = unique_values_R[unique_values_R != 0]

    assert data_to_parc.shape[1]==left.shape[0]+right.shape[0], f"Data to parcellate must have the same number of vertices as the atlas, but have {data_to_parc.shape[1]} vertices and the atlas has {left.shape[0]+right.shape[0]} vertices."

    left_data=data_to_parc[:, :left.shape[0]]
    right_data=data_to_parc[:, left.shape[0]:]

    parc_data_L = np.zeros((data_to_parc.shape[0], len(unique_values_L)))
    parc_data_R = np.zeros((data_to_parc.shape[0], len(unique_values_R)))

    for i, val in enumerate(unique_values_L):
        L_indices = np.where(left == val)[0]  # extract column indices
        parc_data_L[:, i] = np.mean(left_data[:, L_indices], axis=1)

    for i, val in enumerate(unique_values_R):
        R_indices = np.where(right == val)[0]  # extract column indices
        parc_data_R[:, i] = np.mean(right_data[:, R_indices], axis=1)

    print(f"Left parcellated data shape: {parc_data_L.shape}")
    print(f"Right parcellated data shape: {parc_data_R.shape}")

    all_data=concat_R_L(parc_data_L, parc_data_R, interleaved=interleaved)
    print(f'Left and Right combined data shape: {all_data.shape}')

    return all_data

def spider_plot_non_interactive(output_dir:str, subject:str, num_columns:int, corr_matrices:dict, labels:dict, network_names:list, filename_base:str='kmeans', network_of_interest:str='FPN', minimal=False):
    """
    Creates a spider plot for the subnetworks, not using plotly.

    Args:
        output_dir (str): output directory
        subject (str): subject ID
        num_columns (int): number of columns (networks)
        corr_matrices (dict): correlation matrices for this k and subnetwork formatted like this: k=2: {'Subnetwork 1': None, 'Subnetwork 2': None} ...
        labels (dict): labels of the subnetwork file
        network_names (list): network names
        filename_base (str): filename base (default: 'kmeans') (should contain the k, will otherwise overwrite for each iteration)
        network_of_interest (str): network of interest (default: 'FPN')
        minimal (bool): if true, suppresses labels and x-axis labels (default: False)

    Returns:
        None
    
    """


    # create angles for the spider plot (one for each sub-key)
    angles = np.linspace(0, 2 * np.pi, num_columns, endpoint=False).tolist()
    
    sub_dict=corr_matrices

    # prep colours to plot in
    sub_key_to_color = {}
    for label, (label_name, rgba) in labels.items():
        sub_key_to_color[label_name] = rgba
                
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.9)
    fig2, ax2 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True)) # negative correlations
    
    # superimpose all the sub-keys in the current dictionary (for example, 'A', 'B', etc.)
    for sub_key, correlations in sub_dict.items():
        # ensure the correlations are in a circular form by repeating the first value at the end
        positive_correlations = [value if value > 0 else 0 for value in correlations]
        negative_correlations = [value if value < 0 else 0 for value in correlations]
        print(f"Found {len(positive_correlations)} positive correlations for {sub_key}.")

        if not positive_correlations:
            print(f"No positive correlations found for {sub_key}. Skipping...")
            continue

        correlations = np.concatenate((positive_correlations, [positive_correlations[0]]))
        correlations_neg = np.concatenate((negative_correlations, [negative_correlations[0]]))
        jittered_angles = add_jitter(angles)
        angles_with_repeated_end = np.append(jittered_angles, jittered_angles[0])

        color = sub_key_to_color.get(sub_key, (0.0, 0.0, 0.0, 1.0))  # Default to black if not found
        
        # plot each sub-key (e.g., 'A', 'B', 'P', etc.)
        ax.plot(angles_with_repeated_end, correlations, label=sub_key, color=color[:3] + (0.3,), alpha=0.75, linewidth=2)
        ax.fill(angles_with_repeated_end, correlations, color=color[:3] + (0.3,), alpha=0.25)

        ax2.plot(angles_with_repeated_end, correlations_neg, label=sub_key, color=color[:3] + (0.3,), alpha=0.75, linewidth=2)
        ax2.fill(angles_with_repeated_end, correlations_neg, color=color[:3] + (0.3,), alpha=0.25)
    
    # set the labels (can be based on actual data or just the key)
    ax.set_xticks(angles)

    if minimal == False:
        ax.set_xticklabels(network_names, fontsize=7, ha='center', va='center', rotation=45)  # rotate and set alignment   
        ax.tick_params(pad=10)
    if minimal==True:
        ax.set_xticklabels([])
    max_correlation = max(max(correlations) for correlations in sub_dict.values())
    ax.set_ylim(0, max_correlation * 1.1) # add buffer
    ax.set_yticks(np.linspace(0, max_correlation, 5))
    ax.tick_params(axis='y', labelsize=6, colors='grey')
    ax.set_yticklabels([f"{tick:.2f}" for tick in np.linspace(0, max_correlation, 5)])

    ax2.set_xticks(angles)

    if minimal==False:
        ax2.set_xticklabels(network_names, fontsize=8)  # rotate and set alignment
    if minimal==True:
        ax2.set_xticklabels([])

    ax2.set_ylim(0, -1)  # invert radial axis so neg correlations extend outwards
    ax2.set_yticks(np.linspace(-1, 0, 10))
    ax2.set_yticklabels([f"{tick:.2f}" for tick in np.linspace(-1, 0, 10)])

    
    # add legend to distinguish between different sub-keys ('A', 'B', etc.)
    # if minimal==False:
    #     ax.legend(title="Subnetworks",loc="upper left", bbox_to_anchor=(-0.17, 1.13))
    #     ax2.legend(title="Subnetworks")
    
    # Show the plot
    save_plot(fig, os.path.join(output_dir, f'{subject}_{network_of_interest}_{filename_base}_spider_plot.png'))
    save_plot(fig2, os.path.join(output_dir, f'{subject}_{network_of_interest}_{filename_base}_spider_plot_neg.png'))

def generic_lable_table(filename:str, network_name:str=None, indices:list=None, network_and_indices:dict=None):
    """
    Creates a generic lable table that is needed to create a dlabel.nii file. Note: all ROIs will appear black.

    Args:
        network_name (str): network name
        filename (str): filename for lable table
        indices (list): indices to use for the lable table (needs to be int!)
        network_and_indices (dict): dictionary with network names as keys and indices as values as an alternative to providing network_name and indices 

    Returns:
        None    
    """

    if network_name is not None and indices is not None:
        with open(filename, 'w') as file:
            for i in indices:
                if type(i)!=int:
                    i=int(i)
                file.write(f"{network_name}_{i}\n")
                file.write(f"{i} 255 255 255 255\n")
    elif network_and_indices is not None:
        with open(filename, 'w') as file:
            for network, i in network_and_indices.items():
                if type(i)!=int:
                    i=int(i)
                file.write(f"{network}\n")
                file.write(f"{i} 255 255 255 255\n")
    else:
        raise ValueError("Provide either network_name and indices or network_and_indices.")

def wb_cifti_merge(output_file:str, files_to_merge:list, add_argument=None):
    """
    Merges cifti files.

    Args:
        output_file (str): output file
        files_to_merge (list): files to merge
        add_argument (str): additional argument
    
    Returns:
        None
    """

    cmd = ['wb_command', '-cifti-merge', output_file] + [item for file in files_to_merge for item in ('-cifti', file)] + add_argument if add_argument else []

    subprocess.run(cmd)

def wb_cifti_separate(input_file:str, output_file_L:str, output_file_R:str):
    """
    Separates a cifti file into left and right hemisphere.

    Args:
        input_file (str): input file
        output_file_L (str): output file for left hemisphere
        output_file_R (str): output file for right hemisphere
    
    Returns:
        None
    """

    subprocess.run(['wb_command', '-cifti-separate', input_file, 'COLUMN', '-label', 'CORTEX_LEFT', output_file_L, '-label', 'CORTEX_RIGHT', output_file_R])  

def label_resample(input_file:str, current_sphere:str, new_sphere:str, output_file:str, current_midthick:str, new_midthick:str, method:str='ADAP_BARY_AREA'):
    """
    Resamples a label.gii file.

    Args:
        input_file (str): input file
        current_sphere (str): current sphere file
        new_sphere (str): new sphere file
        method (str): method
        output_file (str): output file
        current_midthick (str): current midthickness file
        new_midthick (str): new midthickness file

    Returns:
        None
    """
    subprocess.run(['wb_command', '-label-resample',
                    input_file,
                    current_sphere, new_sphere,
                    method,
                    output_file,
                    '-area-surfs', current_midthick, new_midthick])
    
def cifti_merge_single_map(files_to_merge:str, output_file:str, dtseries_template:str):
    """
    Merges cifti files into a single map.

    Args:
        files_to_merge (str): files to merge
        output_file (str): output file
        dtseries_template (str): dtseries template file to save into dlabel (any dtseries will work that is in the required space.)
    
    Returns:
        None
    """
    data_to_merge = []
    indices=[]
    new_labels=[]
    rois_indices_dict={}
    

    for file in files_to_merge:
        labels, data = get_labels(file, return_data=True)

        indices_network=np.unique(data)
        indices_network=indices_network[indices_network!=0]

        indices_network=indices_network.tolist() # convert to list
        indices_network=list(map(int, indices_network)) # convert to int
        indices_network_reverse=sorted(indices_network, reverse=True) # reverse the list to be descending

        for index in indices_network_reverse:
            
            new_index=index
            while new_index in indices:
                new_index+=1
            
            current_label=labels[index][0]
            base_name, suffix = current_label.rsplit("_", 1)

            new_label=f"{base_name}_{new_index}"

            print(f'Found label {current_label} for index {index}. Adding as {new_index}, {new_label}.')

            # store
            new_labels.append(new_label)
            data[data==index]=new_index
            indices.append(new_index)
            rois_indices_dict[new_label]=new_index

        data_to_merge.append(data)
        print(f'New data values are {np.unique(data)}')
    
    print(rois_indices_dict)

    # merge the data
    arrays=np.stack(data_to_merge)

    nonzero_counts = np.count_nonzero(arrays, axis=0)

    if np.any(nonzero_counts > 1):
        raise ValueError("Conflict detected: The same vertex is assigned multiple times.")

    merged_arrays=np.where(arrays.any(axis=0), arrays.max(axis=0), 0)
    print(f'Merged data values are {np.unique(merged_arrays)}')
    print(f'Shape of merged data is {merged_arrays.shape}') # (1, 64984)

    # save the merged data
    label_table_file=os.path.join(os.path.dirname(output_file), 'labeltable.txt')
    generic_lable_table(label_table_file, network_and_indices=rois_indices_dict)

    write_dlabel(merged_arrays, template_file=dtseries_template, label_table=label_table_file, out_dlabel=output_file, discard_others=True,drop_unused_labels=True)
    print(f"Saved merged file to {output_file}")

def dict_from_label_table(label_table: str):
    """
    Extracts region names from a label table txt file and maps them to their corresponding numbers.

    Args:
        label_table (str): Path to the label table file.

    Returns:
        dict: A dictionary mapping region numbers to their names.
    """
    region_dict = {}
    
    with open(label_table, "r") as file:
        lines = file.readlines()
    
    for i in range(0, len(lines), 2):  # Process every two lines
        name = lines[i].strip()
        if i + 1 < len(lines):  # Ensure there is a corresponding number line
            parts = lines[i + 1].split()
            if parts and parts[0].isdigit():
                region_dict[int(parts[0])] = name
    
    return region_dict

def find_duplicates(lst:list):
    """
    Finds duplicates in a list.

    Args:
        lst (list): list

    Returns:
        list: duplicates
    """
    seen = set()
    duplicates = []
    for item in lst:
        if item in seen:
            duplicates.append(item)
        else:
            seen.add(item)
    return duplicates

def find_atlas_rois_in_vol_atlas(vol_atlases_L:list, vol_atlases_R:list):
    """
    Finds the ROIs in a volumetric atlas.

    Args:
        vol_atlases (list): volumetric atlases

    Returns:
        all_labels (dict): all labels (with indices as keys, and roi names as values)
        keep_indices (dict): indices present for each hemispheres in at least one atlas (with hemispheres as keys and indices lists as values)
    """

    all_labels={}
    keep_indices_L=[]
    keep_indices_R=[]

    for atlas_L_file, atlas_R_file in zip(vol_atlases_L, vol_atlases_R):
        atlas_L=load_data(atlas_L_file)
        atlas_R=load_data(atlas_R_file)
        
        # get unique values
        unique_L=np.unique(atlas_L[atlas_L!=0])
        unique_R=np.unique(atlas_R[atlas_R!=0])

        print(f'Found {len(unique_L)} unique ROIs in the left hemisphere.')
        print(f'Found {len(unique_R)} unique ROIs in the right hemisphere.')

        left_label_table=os.path.join(os.path.dirname(atlas_L_file), 'label_table_L.txt')
        right_label_table=os.path.join(os.path.dirname(atlas_R_file), 'label_table_R.txt')

        subprocess.run(['wb_command', '-volume-label-export-table', atlas_L_file, '1', left_label_table])
        subprocess.run(['wb_command', '-volume-label-export-table', atlas_R_file, '1', right_label_table])

        labels_L=dict_from_label_table(left_label_table)
        labels_R=dict_from_label_table(right_label_table)

        # add labels only if they are present in unique_L or unique_R
        for key, value in labels_L.items():
            if int(key) in unique_L and not key in all_labels:
                all_labels[key] = value

                if key-1 not in keep_indices_L:
                    keep_indices_L.append(key-1)

        for key, value in labels_R.items():
            if int(key) in unique_R and not key in all_labels:
                all_labels[key] = value

                if key-1 not in keep_indices_R:
                    keep_indices_R.append(key-1)

        os.remove(left_label_table)
        os.remove(right_label_table)
    
    keep_indices={}
    keep_indices['L']=keep_indices_L
    keep_indices['R']=keep_indices_R

    return all_labels, keep_indices

"""
Layer connectivity functions with general applications.
"""

def interleave_layers(data:pd.DataFrame):
    """
    Interleaves columns of type {xyz}_superficial, {xyz}_mid, {xyz}_deep into a single DataFrame.

    Args:
        data (DataFrame): Dictionary containing DataFrames for 'superficial', 'mid', and 'deep' layers.

    Returns:
        pd.DataFrame: Interleaved DataFrame with columns named as {xyz}_{layer}.
    """

    interleaved_columns = []
    interleaved_data = []
    column_order = data['superficial'].columns  
    for col in column_order:
        for key in ['superficial', 'mid', 'deep']:  # maintain correct order
            interleaved_columns.append(f'{col}_{key}')
            interleaved_data.append(data[key][col])  # collect columns in correct order

    # concatenate along columns
    interleaved_df = pd.concat(interleaved_data, axis=1)
    interleaved_df.columns = interleaved_columns

    return interleaved_df

def get_n_rois(columns:list, string:str):
    """
    Count the number of ROIs for a network / subnetwork of a DataFrame based on prefixes.

    Args:
        columns (list): List of column names.
        string (str): Prefix string to search for.
    
    Returns:
        int: Number of ROIs with the specified prefix.
    """
    n_rois = sum(1 for col in columns if isinstance(col, str) and col.startswith(string))

    return n_rois

def get_columns(df:pd.DataFrame, prefix_string:str):
    """
    Get columns from a DataFrame that start with a specific prefix string.

    Args:
        df (pd.DataFrame): The DataFrame to search.
        prefix_string (str): The prefix string to match.

    Returns:
        list: A list of column names that start with the prefix string.
    """

    return [col for col in df.columns if col.startswith(prefix_string)]

def calculate_average_layer_conn_upper_triangle(list_of_columns:list, df:pd.DataFrame, aggregation=['superficial', 'mid', 'deep'], verbose=False, groups=['1', '2']):
    """
    Calculate the average correlation within each layer (across all ROIs) in the upper triangle of a correlation matrix.

    Args:
        columns (list): List of column names.
        df (pd.DataFrame): DataFrame containing correlation values.
        aggregation (list): List of layer types to consider (default: ['superficial', 'mid', 'deep']).

    Returns:
        dict: A dictionary with layer types as keys and average correlations as values.
    
    """
    if len(list_of_columns)==1:
        columns=list_of_columns[0]

        averages = {}
        for agg in aggregation:
            # find column names
            layer_columns = [col for col in columns if col.endswith(f'_{agg}')]
            
            # extract relevant columns and rows (submatrix)
            sub_corr_df = df[layer_columns].loc[layer_columns]
            
            # extract upper triangle
            upper_triangle = sub_corr_df.where(np.triu(np.ones(sub_corr_df.shape, dtype=bool), k=1)).stack()
            if verbose:
               print("Upper triangle of the correlation matrix:\n", upper_triangle)

            # compute mean
            layer_correlations = upper_triangle.mean()
            averages[agg] = layer_correlations
            if verbose:
                print(f"For {agg}, the average correlation with {agg} for this subnetwork is {layer_correlations}.")

    else:
        averages = {}

        for (i, group1), (j, group2) in combinations(enumerate(groups), 2):
            pair_key = f"{group1}_{group2}"
            averages[pair_key] = {}

            if verbose:
                print(f"Processing pair: {pair_key}")

            columns1 = list_of_columns[i]
            columns2 = list_of_columns[j]

            # for each layer
            for agg in aggregation:
                layer_group1_cols = [col for col in columns1 if col.endswith(f'_{agg}')]
                layer_group2_cols = [col for col in columns2 if col.endswith(f'_{agg}')]

                if verbose:
                    print(f"Layer {agg}: {len(layer_group1_cols)} cols in {group1}, {len(layer_group2_cols)} cols in {group2}")

                # select submatrix
                sub_corr_df = df.loc[layer_group1_cols, layer_group2_cols]

                # flatten and compute mean (ignoring nan)
                mean_correlation = np.nanmean(sub_corr_df.values.flatten())
                averages[pair_key][agg] = mean_correlation

                if verbose:
                    print(f"avg correlation for {agg}: {mean_correlation:.4f}")
    
    return averages

def read_region_file(filename):
    """
    Reads a region file in markdown format containing regions classified by categories (e.g., medial vs lateral lobe).

    Args:
        filename (str): Path to the region file.
    
    Returns:
        dict: A dictionary where keys are categories and values are lists of regions.
    
    Example:
        Region file should be formatted like this:
        Lateral:
            - L_p9-46v_ROI
            ...
    """
    regions = defaultdict(list)
    category = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.endswith(':'):
                category = line[:-1]
            elif line.startswith('-'):
                regions[category].append(line[2:])
    return regions

"""
Permutation tests
"""

def get_parcel_names(parcels_axis):
    """
    Get parcel names from a parcels axis.

    Args:
        parcels_axis (nibabel object): parcels axis

    Returns:
        list: parcel names
    """
    return [parcels_axis.name[i] for i in range(len(parcels_axis.name))]

def get_mean_corr(rows:list, cols:list, corr_matrix_z:pd.DataFrame, verbose_f=True):
    """
    Calculate the mean correlation for a submatrix defined by the specified rows and columns.

    Args:
        rows (list): List of row indices.
        cols (list): List of column indices.
        corr_matrix_z (pd.DataFrame): Correlation matrix in z-score format.
        verbose_f (bool): If True, print additional information.

    Returns:
        float: Mean correlation value for the specified submatrix.
    
    """
    submatrix = corr_matrix_z.loc[rows, cols]
    mean_corr = submatrix.values.mean()
    if verbose_f:
        print(f'Submatrix shape: {submatrix.shape}')
    return mean_corr

def dice_coefficient(mask_a, mask_b):
    """
    Computes the Dice coefficient between two binary masks.
    
    Args:
        mask_a (np.ndarray): First binary mask.
        mask_b (np.ndarray): Second binary mask.

    Returns:
        float: Dice coefficient between the two masks.    

    """

    intersection = np.sum((mask_a != 0) & (mask_b != 0))
    size_a = np.sum(mask_a != 0)
    size_b = np.sum(mask_b != 0)

    dice = 2 * intersection / (size_a + size_b)
    return dice

def get_label_table_from_cifti(cifti, filename, map=1):
    """
    Extracts the label table from a cifti file and saves it to a file.

    Args:
        cifti (str): Path to the cifti file.
        map (int): Map number to extract (default: 0).
        filename (str): Output filename for the label table.
    
    Returns:
        None
    """

    subprocess.run(['wb_command', '-cifti-label-export-table',
                        cifti, str(map), filename])