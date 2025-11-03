#!/bin/bash -l
#SBATCH --job-name=kmeans_ver
#SBATCH --output=/ptmp/hmueller2/kmeans_vertices_logs/output/%A_%x_%a_%u.out
#SBATCH --error=/ptmp/hmueller2/kmeans_vertices_logs/errors/%A_%x_%a_%u.err
#SBATCH --partition=compute
#SBATCH --exclusive=user
#SBATCH --array=0-7
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

container=/home/rglz/containers/gfae.sif
working_dir=/ptmp/hmueller2/Downloads
wheel_dir=/home/hmueller2/wheelhouse

# Clean bind vars and bind the needed paths
unset SINGULARITY_BINDPATH SINGULARITY_BIND APPTAINER_BINDPATH APPTAINER_BIND
BIND="--bind /ptmp,/home"

export PIP_NO_INDEX=1

# Full offline install of compatible versions into ~/.local (inside container)
apptainer exec ${BIND} ${container} python3 -m pip install --user --no-index --find-links="${wheel_dir}" --upgrade --force-reinstall \
  numpy==1.24.4 scipy==1.10.1 joblib==1.2.0 threadpoolctl==3.1.0 scikit-learn==1.2.2 spherecluster==0.1.7

# Subject
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${working_dir}/subjects_resting.txt")
subject=$(echo "$line" | awk '{print $1}')
echo "Processing subject: sub-${subject}"

# Sanity check
apptainer exec ${BIND} ${container} python3 - <<'PY'
import sys
print("py", sys.version.split()[0])
try:
    import numpy, scipy, sklearn, spherecluster
    from spherecluster import SphericalKMeans
    print("numpy", numpy.__version__, "scipy", scipy.__version__, "sklearn", sklearn.__version__, "spherecluster", getattr(spherecluster,"__version__","?"))
    print("[info] SphericalKMeans import OK")
except Exception as e:
    print("[warn] spherecluster not usable:", e)
PY

# Run
srun apptainer exec ${BIND} ${container} python3 /home/hmueller2/ibc_code/ibc_latent/Subnetworks/kmeans_on_vertices.py --subject ${subject}