"""
Relabel k=2 subclusters so that 1=DMN-like and 2=DAN-like, based on correlation profiles.
"""

import os
import sys
import csv
import argparse
import numpy as np
import nibabel as nib

sys.path.insert(1, '/home/hmueller2/ibc_code/ibc_latent/Preprocessing/Aradia')
import RR_utils as RR

approach = "kmeans" # choose "kmeans" or "infomap"

def main(subject: str, write_dscalar: bool = True):
    subject = subject.zfill(2)
    working_dir = '/ptmp/hmueller2/Downloads'
    sub_str = f"sub-{subject}"

    # Helper: ensure exactly one _relabeled suffix
    def make_relabeled(base_name: str, kind: str) -> str:
        for ext in ('.dtseries.nii', '.dscalar.nii', '.dlabel.nii'):
            if base_name.endswith(ext):
                stem = base_name[:-len(ext)]
                break
        else:
            stem = os.path.splitext(base_name)[0]
        if stem.endswith('_relabeled'):
            stem = stem[:-10]  # remove existing suffix
        return f"{stem}_relabeled.{kind}.nii"

    # Inputs - determine which approach to use
    kmeans_dir = os.path.join(working_dir, "subnetworks", "kmeans", sub_str)
    infomap_dir = os.path.join(working_dir, "subnetworks", "infomap", sub_str)
    
    if approach == "kmeans":
        input_file = os.path.join(kmeans_dir, f"{sub_str}_kmeans_on_vertices.dtseries.nii")
        output_dir = kmeans_dir
    elif approach == "infomap":
        # Try both .dscalar and .dlabel variants
        dscalar_file = os.path.join(infomap_dir, f"{subject}_FPN_infomap_communities_kmeans.dscalar.nii")
        dlabel_file = os.path.join(infomap_dir, f"{subject}_FPN_infomap_communities_kmeans.dlabel.nii")
        if os.path.exists(dscalar_file):
            input_file = dscalar_file
        elif os.path.exists(dlabel_file):
            input_file = dlabel_file
        else:
            print(f"[error] No infomap file found at {dscalar_file} or {dlabel_file}")
            sys.exit(1)
        output_dir = infomap_dir
    else:
        print(f"[error] Unknown approach: {approach}")
        sys.exit(1)
    
    parc_filename = os.path.join(working_dir, 'individual_networks', sub_str, 'resting_state', f'{sub_str}_individual_nets_concat.ptseries.nii')
    dtseries_path = os.path.join(working_dir, 'individual_networks', sub_str, 'resting_state', f'{sub_str}_all-tasks_concatenated_cleaned_fsLR_cortexOnly.dtseries.nii')

    if not os.path.exists(input_file):
        print(f"[error] Missing input file: {input_file}")
        sys.exit(1)
    if not os.path.exists(parc_filename):
        print(f"[error] Missing ptseries: {parc_filename}")
        sys.exit(1)
    if not os.path.exists(dtseries_path):
        print(f"[error] Missing dtseries: {dtseries_path}")
        sys.exit(1)

    # Load labels (all k) and extract k=2 map (index 1)
    labels_all = RR.load_data(input_file)
    if labels_all.ndim == 1:
        print("[error] Input file does not contain multiple k maps (expected 2D).")
        sys.exit(1)
    
    # For kmeans dtseries, k=2 is at index 0; for infomap, it's at index 1
    k2_index = 0 if approach == "kmeans" else 1
    labels_k2 = labels_all[k2_index, :].astype(int)

    # Align to cortex-only dtseries if needed
    if labels_k2.shape[0] == 91282:
        labels_k2_cortex = labels_k2[:64984]
    else:
        labels_k2_cortex = labels_k2
    if labels_k2_cortex.shape[0] != 64984:
        print(f"[error] Unexpected labels length for cortex: {labels_k2_cortex.shape}")
        sys.exit(1)

    # Load ptseries and drop FPN + Noise to match indices used elsewhere
    all_data_concat = RR.load_data(parc_filename)
    all_data_concat = np.delete(all_data_concat, [8, -1], axis=1)  # remove FPN and Noise
    dmn_idx = [0, 1, 2, 3]
    dan_idx = [8, 9]

    # Load cortex-only dtseries for averaging subnetwork time series
    dtseries_concat = RR.load_data(dtseries_path)  # shape: (T, 91282)
    
    # Extract actual cortex-only data (first 64984 vertices)
    dtseries_concat_cortex = dtseries_concat[:, :64984]  # shape: (T, 64984)

    # Compute DMN/DAN means per cluster (1 and 2)
    dmn_dan_means = []
    for cid in (1, 2):
        mask = (labels_k2_cortex == cid)
        if not np.any(mask):
            print(f"[warn] No vertices for cluster {cid}")
            dmn_dan_means.append((cid, np.nan, np.nan))
            continue
        comm_ts = RR.get_network(dtseries_concat_cortex, mask, remove_rest=True)  # (T, N)
        avg_ts = np.mean(comm_ts, axis=1)
        corr_matrix = np.corrcoef(all_data_concat.T, avg_ts)
        corr_vec = corr_matrix[-1, :-1]
        dmn_mean = float(np.nanmean(np.asarray(corr_vec)[dmn_idx]))
        dan_mean = float(np.nanmean(np.asarray(corr_vec)[dan_idx]))
        dmn_dan_means.append((cid, dmn_mean, dan_mean))

    # Build assignment: choose the labeling that maximizes DMN-DAN separation
    if any(np.isnan(m[1]) or np.isnan(m[2]) for m in dmn_dan_means):
        print("[error] Could not compute DMN/DAN means for both clusters.")
        sys.exit(1)
    
    # Extract connectivity values for both clusters
    cluster_1_dmn, cluster_1_dan = dmn_dan_means[0][1], dmn_dan_means[0][2]
    cluster_2_dmn, cluster_2_dan = dmn_dan_means[1][1], dmn_dan_means[1][2]
    
    # Calculate two possible assignment scenarios:
    # Scenario A: Cluster 1 → Label 1 (DMN-like), Cluster 2 → Label 2 (DAN-like)
    scenario_a_label1_dmn_advantage = cluster_1_dmn - cluster_1_dan  # should be positive (higher DMN)
    scenario_a_label2_dan_advantage = cluster_2_dan - cluster_2_dmn  # should be positive (higher DAN)
    scenario_a_total_separation = scenario_a_label1_dmn_advantage + scenario_a_label2_dan_advantage
    
    # Scenario B: Cluster 1 → Label 2 (DAN-like), Cluster 2 → Label 1 (DMN-like)
    scenario_b_label1_dmn_advantage = cluster_2_dmn - cluster_2_dan  # should be positive (higher DMN)
    scenario_b_label2_dan_advantage = cluster_1_dan - cluster_1_dmn  # should be positive (higher DAN)
    scenario_b_total_separation = scenario_b_label1_dmn_advantage + scenario_b_label2_dan_advantage
    
    # Choose the scenario with maximum total separation
    if scenario_a_total_separation >= scenario_b_total_separation:
        assign_map = {1: 1, 2: 2}  # Cluster 1 → DMN-like, Cluster 2 → DAN-like
        label1_dmn, label1_dan = cluster_1_dmn, cluster_1_dan
        label2_dmn, label2_dan = cluster_2_dmn, cluster_2_dan
        chosen_separation = scenario_a_total_separation
        scenario = "A"
    else:
        assign_map = {1: 2, 2: 1}  # Cluster 1 → DAN-like, Cluster 2 → DMN-like
        label1_dmn, label1_dan = cluster_2_dmn, cluster_2_dan
        label2_dmn, label2_dan = cluster_1_dmn, cluster_1_dan
        chosen_separation = scenario_b_total_separation
        scenario = "B"
    
    print(f"[info] k=2 assignment based on maximum DMN-DAN separation (Scenario {scenario}):")
    print(f"       Label 1 (DMN-like): DMN={label1_dmn:.3f}, DAN={label1_dan:.3f}, Diff={label1_dmn-label1_dan:.3f}")
    print(f"       Label 2 (DAN-like): DMN={label2_dmn:.3f}, DAN={label2_dan:.3f}, Diff={label2_dan-label2_dmn:.3f}")
    print(f"       Total separation score: {chosen_separation:.3f}")

    # Relabel cortex labels
    relabeled_cortex = np.zeros_like(labels_k2_cortex)
    for old, new in assign_map.items():
        relabeled_cortex[labels_k2_cortex == old] = new

    # Save assignment CSV
    assign_csv = os.path.join(output_dir, f"{sub_str}_k2_assignment.csv")
    with open(assign_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['original_label', 'assigned_label', 'mean_DMN', 'mean_DAN'])
        # Map means to rows
        mm = {cid: (dmn, dan) for cid, dmn, dan in dmn_dan_means}
        for old in (1, 2):
            dmn_mean, dan_mean = mm.get(old, (np.nan, np.nan))
            w.writerow([old, assign_map.get(old, ''), dmn_mean, dan_mean])
    print(f"[info] Saved assignment CSV: {assign_csv}")

    # Optionally write a relabeled file (k=2 row replaced)
    if write_dscalar:
        try:
            img = nib.load(input_file)
            data = img.get_fdata().astype(np.float32)
            if data.shape[0] < 2:
                raise RuntimeError(f"Unexpected shape: {data.shape}")
            # If original was 91k, we need to write back full-length
            if labels_k2.shape[0] == 91282:
                full = labels_all[k2_index, :].astype(int).copy()
                full[:64984] = relabeled_cortex
                data[k2_index, :] = full
            else:
                data[k2_index, :labels_k2_cortex.shape[0]] = relabeled_cortex
            
            # Determine output filename based on input type (avoid double _relabeled)
            base_name = os.path.basename(input_file)
            if base_name.endswith('.dtseries.nii'):
                kind = 'dtseries'
            elif base_name.endswith('.dscalar.nii'):
                kind = 'dscalar'
            elif base_name.endswith('.dlabel.nii'):
                kind = 'dlabel'
            else:
                kind = 'dscalar'
            out_name = make_relabeled(base_name, kind)
            out_path = os.path.join(output_dir, out_name)
            
            out_img = nib.Cifti2Image(data, header=img.header, nifti_header=img.nifti_header)
            nib.save(out_img, out_path)
            print(f"[info] Wrote relabeled file: {out_path}")

            # Also write a dlabel version
            try:
                from nibabel.cifti2.cifti2_axes import LabelAxis
                brain_axis = img.header.get_axis(1)  # reuse brain models
                # Build full-length relabeled vector (matching original vertex length)
                if labels_k2.shape[0] == 91282:
                    relabeled_full = full  # already assembled above
                else:
                    relabeled_full = data[k2_index, :].astype(int)
                label_data = relabeled_full.reshape(1, -1).astype(np.int32)

                # Define label table (RGBA floats 0-1)
                label_dict = {
                    0: ("Background", (0.0, 0.0, 0.0, 0.0)),
                    1: ("DMN_like", (0.0, 0.5, 0.5, 1.0)),  # teal
                    2: ("DAN_like", (0.0, 0.0, 0.5, 1.0)),  # dark blue
                }
                # Correct LabelAxis construction: needs a list of map names
                from nibabel.cifti2.cifti2_axes import LabelAxis
                label_axis = LabelAxis(['Subnetworks'], label_dict)
                dlabel_header = nib.cifti2.Cifti2Header.from_axes((label_axis, brain_axis))
                dlabel_img = nib.Cifti2Image(label_data, header=dlabel_header, nifti_header=img.nifti_header)

                dlabel_name = make_relabeled(os.path.basename(input_file), 'dlabel')
                dlabel_path = os.path.join(output_dir, dlabel_name)
                nib.save(dlabel_img, dlabel_path)
                print(f"[info] Wrote relabeled dlabel file: {dlabel_path}")
            except Exception as e:
                print(f"[warn] Failed to write dlabel file: {e}")
        except Exception as e:
            print(f"[warn] Failed to write relabeled file: {e}")
            # Always save the relabeled cortex vector as numpy for downstream overrides
            npy_path = os.path.join(output_dir, f"{sub_str}_k2_labels_relabel.npy")
            np.save(npy_path, relabeled_cortex.astype(np.int16))
            print(f"[info] Saved relabeled cortex labels as NPY: {npy_path}")
    else:
        npy_path = os.path.join(output_dir, f"{sub_str}_k2_labels_relabel.npy")
        np.save(npy_path, relabeled_cortex.astype(np.int16))
        print(f"[info] Saved relabeled cortex labels as NPY: {npy_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Relabel k=2 subclusters so 1=DMN-like and 2=DAN-like.")
    ap.add_argument("--subject", required=True, help="Subject without 'sub-' prefix, e.g., 04")
    ap.add_argument("--no-dscalar", action="store_true", help="Do not write relabeled dscalar; save NPY only.")
    args = ap.parse_args()
    main(args.subject, write_dscalar=not args.no_dscalar)

