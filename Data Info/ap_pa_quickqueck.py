'''
Quick check of AP vs PA runs by comparing global-mean time series and per-vertex standard deviation maps.
Why? To make sure that ap and pa are actually separate runs and comparibly long.
'''

import os
import numpy as np
import nibabel as nib
from math import ceil

AP_PATH = "/ptmp/hmueller2/2025_ibc_latent/outputs/preprocessing/fmriprep_out/sub-04/ses-01/postfmriprep/GLM/sub-04_ses-01_task-HcpEmotion_dir-ap_cleaned.dtseries.nii"
PA_PATH = "/ptmp/hmueller2/2025_ibc_latent/outputs/preprocessing/fmriprep_out/sub-04/ses-01/postfmriprep/GLM/sub-04_ses-01_task-HcpEmotion_dir-pa_cleaned.dtseries.nii"
SECONDS = 20  # first N seconds to inspect

def tr_and_len(img):
    ax0, ax1 = img.header.get_axis(0), img.header.get_axis(1)
    ts_axis = ax0 if isinstance(ax0, nib.cifti2.SeriesAxis) else ax1 if isinstance(ax1, nib.cifti2.SeriesAxis) else None
    tr = float(getattr(ts_axis, "step", 2.0)) if ts_axis is not None else 2.0
    n = ts_axis.size if ts_axis is not None else img.shape[0]
    time_dim_first = isinstance(ax0, nib.cifti2.SeriesAxis)
    return tr, n, time_dim_first

def first_seconds_global_mean(img, seconds):
    tr, n, time_first = tr_and_len(img)
    k = min(n, int(ceil(seconds / tr)))
    if time_first:
        # dataobj is (T, V); slice only first k timepoints to keep memory small
        chunk = np.asarray(img.dataobj[:k, :], dtype=np.float32)
        ts = chunk.mean(axis=1)
    else:
        # dataobj is (V, T)
        chunk = np.asarray(img.dataobj[:, :k], dtype=np.float32)
        ts = chunk.mean(axis=0)
    return tr, n, k, ts

def full_run_global_mean(img):
    tr, n, time_first = tr_and_len(img)
    if time_first:
        ts = np.asarray(img.dataobj[:, :], dtype=np.float32).mean(axis=1)
    else:
        ts = np.asarray(img.dataobj[:, :], dtype=np.float32).mean(axis=0)
    return ts

def std_map(img):
    ax0, ax1 = img.header.get_axis(0), img.header.get_axis(1)
    time_first = isinstance(ax0, nib.cifti2.SeriesAxis)
    if time_first:
        return np.asarray(img.dataobj[:, :], dtype=np.float32).std(axis=0, ddof=1)
    else:
        return np.asarray(img.dataobj[:, :], dtype=np.float32).std(axis=1, ddof=1)

def summarize(path):
    if not os.path.exists(path):
        print(f"[missing] {path}")
        return None
    img = nib.load(path)
    tr, n, k, ts = first_seconds_global_mean(img, SECONDS)
    print(f"- {os.path.basename(path)}")
    print(f"  TR={tr:.6f}s, timepoints={n}, inspected_first={k} TR")
    print(f"  global-mean ts (first {min(10, k)} TR): {np.array2string(ts[:min(10, k)], precision=4)}")
    return (tr, n, k, ts)

def main():
    print("AP vs PA quick check\n")
    ap = summarize(AP_PATH)
    pa = summarize(PA_PATH)
    if ap is None or pa is None:
        return
    k = min(ap[2], pa[2])
    if k >= 2:
        r = float(np.corrcoef(ap[3][:k], pa[3][:k])[0, 1])
        print(f"\nOverlap (first {k} TR) global-mean correlation (AP vs PA): r={r:.4f}")
        print(f"AP mean±SD: {ap[3][:k].mean():.6e} ± {ap[3][:k].std(ddof=1):.6e}")
        print(f"PA mean±SD: {pa[3][:k].mean():.6e} ± {pa[3][:k].std(ddof=1):.6e}")

    # Full-run checks
    img_ap, img_pa = nib.load(AP_PATH), nib.load(PA_PATH)
    ts_ap_full = full_run_global_mean(img_ap)
    ts_pa_full = full_run_global_mean(img_pa)
    m = min(ts_ap_full.size, ts_pa_full.size)
    r_full = float(np.corrcoef(ts_ap_full[:m], ts_pa_full[:m])[0, 1])
    print(f"\nFull-run global-mean correlation (AP vs PA): r={r_full:.4f}")

    std_ap = std_map(img_ap); std_pa = std_map(img_pa)
    mstd = min(std_ap.size, std_pa.size)
    r_std = float(np.corrcoef(std_ap[:mstd], std_pa[:mstd])[0, 1])
    print(f"Per-vertex STD mean±SD (AP): {std_ap.mean():.6e} ± {std_ap.std(ddof=1):.6e}")
    print(f"Per-vertex STD mean±SD (PA): {std_pa.mean():.6e} ± {std_pa.std(ddof=1):.6e}")
    print(f"Correlation of per-vertex STD maps: r={r_std:.4f}")

if __name__ == "__main__":
    main()