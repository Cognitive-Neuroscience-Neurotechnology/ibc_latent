import os

# Define the directories based on the hostname
hostname = os.uname().nodename
if hostname == 'nyx-login0.hpc.kyb.local':
    base_dir = '/home/hmueller2/Downloads/ibc_contrast_maps/resulting_smooth_maps_surface/' # find the contrast maps here
    output_dir = '/home/hmueller2/ibc_code/ibc_output_RSA_spearman'
    code_dir = '/home/hmueller2/ibc_code/ibc_latent'
else:
    base_dir = '/Users/hannahmuller/nyx_mount/Downloads/ibc_contrast_maps/resulting_smooth_maps_surface/'
    output_dir = '/Users/hannahmuller/nyx_mount/ibc_code/ibc_output_RSA_cosine'
    code_dir = '/Users/hannahmuller/nyx_mount/ibc_code/ibc_latent'
print(f"Base directory set to: {base_dir}")
print(f"Output directory set to: {output_dir}")

# Define RSA method
method = 'spearman'  # 'spearman' or 'cosine'