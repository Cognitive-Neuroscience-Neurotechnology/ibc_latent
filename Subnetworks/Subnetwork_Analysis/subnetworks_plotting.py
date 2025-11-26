"""
Batch plot k=2 subnetworks for all subjects:
 - Infomap relabeled: {subject}_FPN_infomap_communities_kmeans_relabeled.dscalar.nii (row 1 = k=2)
 - KMeans relabeled:  sub-{subject}_kmeans_on_vertices_relabeled.dtseries.nii (row 0 = k=2)
Uses colors from label_table_infomap_kmeans.txt (teal, dark blue).
Saves PNGs on inflated fsaverage surface.
"""

import os
import sys
import numpy as np
import nibabel as nb
from nilearn import datasets, plotting
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import argparse
from nibabel.cifti2 import Cifti2Image

# Root working dir (adjust if needed)
WORKING_DIR = '/ptmp/hmueller2/Downloads'
SUBNETWORK_DIR = os.path.join(WORKING_DIR, 'subnetworks')
INFOMAP_DIR = os.path.join(SUBNETWORK_DIR, 'infomap')
KMEANS_DIR = os.path.join(SUBNETWORK_DIR, 'kmeans')
LABEL_TABLE = os.path.join(SUBNETWORK_DIR, 'label_table_infomap_kmeans.txt')
OUT_DIR = os.path.join(SUBNETWORK_DIR, 'brain_plots_k2')
os.makedirs(OUT_DIR, exist_ok=True)

# Load label table (expect blocks: id \n id R G B A)
def load_label_colors(label_table_path):
    colors = {}
    with open(label_table_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    for ln in lines:
        parts = ln.split()
        if len(parts) == 5 and parts[0].isdigit():
            idx = int(parts[0])
            r, g, b, a = map(float, parts[1:])
            colors[idx] = (r/255.0, g/255.0, b/255.0, a)
    return colors

label_colors = load_label_colors(LABEL_TABLE)
# Ensure we have colors for labels 1 and 2 (fallback if missing)
teal = label_colors.get(1, (0/255,128/255,128/255,1))
dark_blue = label_colors.get(2, (0/255,0/255,128/255,1))

# Colormap: 0 (background) -> transparent/black, 1 -> teal, 2 -> dark blue
cmap = ListedColormap([
    (0,0,0,0),            # background
    teal,                 # label 1
    dark_blue             # label 2
])

# Use fsLR32k Workbench surfaces
FSLR_SURF_DIR = os.path.join(WORKING_DIR, 'fsLR_masks')
FSLR = {
    'infl_left':  os.path.join(FSLR_SURF_DIR, 'fs_LR.32k.L.inflated.surf.gii'),
    'infl_right': os.path.join(FSLR_SURF_DIR, 'fs_LR.32k.R.inflated.surf.gii'),
    # sulc_left and sulc_right will be added after extraction
}
for k, p in FSLR.items():
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing fsLR file: {k} -> {p}")

# Compute fsLR32k cortex vertex counts (should be 32492 per hemi, 64984 total)
fslr_left = nb.load(FSLR['infl_left'])
fslr_right = nb.load(FSLR['infl_right'])
FSLR_LEFT_VERTS = fslr_left.darrays[0].data.shape[0]
FSLR_RIGHT_VERTS = fslr_right.darrays[0].data.shape[0]
FSLR_CORTEX_VERTICES = FSLR_LEFT_VERTS + FSLR_RIGHT_VERTS
print(f"[info] fsLR32k vertices: L={FSLR_LEFT_VERTS}, R={FSLR_RIGHT_VERTS}, total={FSLR_CORTEX_VERTICES}")

# Load sulc dscalar and extract left/right arrays
SULC_DSCALAR = os.path.join(FSLR_SURF_DIR, 'fs_LR.32k.LR.sulc.dscalar.nii')
sulc_img = nb.load(SULC_DSCALAR)
sulc_data = sulc_img.get_fdata()[0]  # shape: (64984,)
sulc_left = sulc_data[:FSLR_LEFT_VERTS]
sulc_right = sulc_data[FSLR_LEFT_VERTS:FSLR_LEFT_VERTS + FSLR_RIGHT_VERTS]

# Now add sulc arrays to FSLR dictionary
FSLR['sulc_left'] = sulc_left
FSLR['sulc_right'] = sulc_right

ATLASROI = {
    'left': os.path.join(FSLR_SURF_DIR, 'L.atlasroi.32k_fs_LR.shape.gii'),
    'right': os.path.join(FSLR_SURF_DIR, 'R.atlasroi.32k_fs_LR.shape.gii'),
}
for k, p in ATLASROI.items():
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing atlasroi mask: {k} -> {p}")
atlas_left = nb.load(ATLASROI['left']).darrays[0].data.astype(float)
atlas_right = nb.load(ATLASROI['right']).darrays[0].data.astype(float)
if atlas_left.shape[0] != FSLR_LEFT_VERTS or atlas_right.shape[0] != FSLR_RIGHT_VERTS:
    raise ValueError("AtlasROI mask vertex count mismatch.")

def split_hemis(arr):
    """Split cortex-only array using fsLR vertex counts."""
    return arr[:FSLR_LEFT_VERTS], arr[FSLR_LEFT_VERTS:FSLR_LEFT_VERTS + FSLR_RIGHT_VERTS]

def extract_cortex_from_cifti(cifti_img: Cifti2Image, row_idx: int = 0):
    """
    Extract fsLR32k cortex data (L and R) from a CIFTI image row using brain models.
    Returns concatenated L||R arrays of length FSLR_CORTEX_VERTICES.
    """
    data = cifti_img.get_fdata()[row_idx, :]
    bm = cifti_img.header.get_index_map(1)  # brain models
    left = np.zeros(FSLR_LEFT_VERTS, dtype=float)
    right = np.zeros(FSLR_RIGHT_VERTS, dtype=float)

    offset = 0
    for m in bm.brain_models:
        count = m.index_count
        sl = slice(offset, offset + count)
        if m.model_type == 'CIFTI_MODEL_TYPE_SURFACE':
            verts = np.array(m.vertex_indices, dtype=int)
            if m.brain_structure == 'CIFTI_STRUCTURE_CORTEX_LEFT':
                if verts.size > 0:
                    left[verts] = data[sl]
            elif m.brain_structure == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
                if verts.size > 0:
                    right[verts] = data[sl]
        offset += count

    return np.concatenate([left, right])

def apply_atlasroi_mask(label_array):
    left, right = split_hemis(label_array)
    left[atlas_left == 0] = 0
    right[atlas_right == 0] = 0
    return np.concatenate([left, right])

def save_surface_plot(label_array, out_png, title):
    left, right = split_hemis(label_array)
    fig = plt.figure(figsize=(14, 6))

    ax1 = plt.subplot(1, 2, 1, projection='3d')
    plotting.plot_surf_stat_map(
        FSLR['infl_left'],
        left,
        hemi='left',
        view='lateral',
        colorbar=False,
        bg_map=sulc_left,  # now an array
        axes=ax1,
        cmap=cmap,
        darkness=0.4,
        title=title + ' (L)'
    )

    ax2 = plt.subplot(1, 2, 2, projection='3d')
    plotting.plot_surf_stat_map(
        FSLR['infl_right'],
        right,
        hemi='right',
        view='lateral',
        colorbar=False,
        bg_map=sulc_right,  # now an array
        axes=ax2,
        cmap=cmap,
        darkness=0.4,
        title=title + ' (R)'
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

def process_subject(sub_str):
    subject = sub_str.replace('sub-','')
    infomap_file = os.path.join(INFOMAP_DIR, sub_str, f'{subject}_FPN_infomap_communities_kmeans_relabeled.dscalar.nii')
    kmeans_file = os.path.join(KMEANS_DIR, sub_str, f'{sub_str}_kmeans_on_vertices_relabeled.dtseries.nii')

    if not os.path.exists(infomap_file):
        print(f'[skip] missing infomap file: {infomap_file}')
    else:
        try:
            img_infomap = nb.load(infomap_file)
            # FIX: Use row_idx=1 for Infomap k=2
            infomap_labels_k2 = extract_cortex_from_cifti(img_infomap, row_idx=1)
            print(f'[debug] infomap cortex shape: {infomap_labels_k2.shape}')
            infomap_labels_k2 = np.nan_to_num(infomap_labels_k2).astype(int)
            infomap_labels_k2 = apply_atlasroi_mask(infomap_labels_k2)
            infomap_out = os.path.join(OUT_DIR, f'{sub_str}_infomap_k2.png')
            save_surface_plot(infomap_labels_k2, infomap_out, f'{sub_str} Infomap k=2')
            print(f'[ok] infomap k=2 plotted: {infomap_out}')
        except Exception as e:
            print(f'[error] infomap {sub_str}: {e}')
            import traceback; traceback.print_exc()

    if not os.path.exists(kmeans_file):
        print(f'[skip] missing kmeans file: {kmeans_file}')
    else:
        try:
            img_kmeans = nb.load(kmeans_file)
            # KMeans k=2 is still row_idx=0
            kmeans_labels_k2 = extract_cortex_from_cifti(img_kmeans, row_idx=0)
            print(f'[debug] kmeans cortex shape: {kmeans_labels_k2.shape}')
            kmeans_labels_k2 = np.nan_to_num(kmeans_labels_k2).astype(int)
            kmeans_labels_k2 = apply_atlasroi_mask(kmeans_labels_k2)
            kmeans_out = os.path.join(OUT_DIR, f'{sub_str}_kmeans_k2.png')
            save_surface_plot(kmeans_labels_k2, kmeans_out, f'{sub_str} KMeans k=2')
            print(f'[ok] kmeans k=2 plotted: {kmeans_out}')
        except Exception as e:
            print(f'[error] kmeans {sub_str}: {e}')
            import traceback; traceback.print_exc()

def discover_subjects():
    subs = []
    if os.path.isdir(INFOMAP_DIR):
        for d in os.listdir(INFOMAP_DIR):
            if d.startswith('sub-') and os.path.isdir(os.path.join(INFOMAP_DIR, d)):
                subs.append(d)
    return sorted(subs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot k=2 subnetworks for a single subject.')
    parser.add_argument('--subject', required=True, help='Subject number, e.g. 04')
    args = parser.parse_args()
    subject = args.subject.zfill(2)
    sub_str = f'sub-{subject}'
    process_subject(sub_str)