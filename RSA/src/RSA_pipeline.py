import os
import nibabel as nib
import numpy as np

# Import from config and task_contrasts
from RSA.config_RSA import base_dir, output_dir
from RSA.task_contrasts import task_contrasts

# Define the parameters to iterate over
subjects = [d.split('-')[1] for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('sub-')] # Extract the subject ID after 'sub-'

