#!/usr/bin/env bash

# run with: bash /home/hmueller2/ibc_code/ibc_latent/Preprocessing/prep_topup.sh
'''
This script prepares BIDS-formatted fieldmap JSON files for topup correction
by associating them with the appropriate functional scans via the IntendedFor field.
For each subject it 1) scans for fieldmap nifti files, 2) determines their session and phase-encoding
direction, 3) finds candidate functional files in the same session, and 4) writes a JSON next to each
fieldmap file containing metadata from a generic JSON plus the IntendedFor entries.
'''

set -euo pipefail

# Test: BIDS_ROOT=/ptmp/hmueller2/Downloads/ibc_raw_test
# Test: sub=sub-04

BIDS_ROOT=/ptmp/hmueller2/Downloads/ibc_raw
SUBJECTS_FILE=/ptmp/hmueller2/Downloads/subjects_resting.txt

# Read subjects from file
while IFS= read -r sub_id || [ -n "$sub_id" ]; do
  # Skip empty lines or comments
  [[ -z "$sub_id" || "$sub_id" =~ ^# ]] && continue
  
  sub="sub-${sub_id}"
  echo "========================================="
  echo "Processing subject: ${sub}"
  echo "========================================="

  # Check if subject directory exists
  if [ ! -d "${BIDS_ROOT}/${sub}" ]; then
    echo "  WARNING: Subject directory not found: ${BIDS_ROOT}/${sub}. Skipping."
    continue
  fi

  # Path to the generic JSONs (where your EffectiveEchoSpacing / PED etc live)
  GENERIC_JSON_DIR="${BIDS_ROOT}"

  # preview before writing
  echo "Scanning for fmap files for ${sub} under ${BIDS_ROOT}..."

  find "${BIDS_ROOT}/${sub}" -type f -name "*dir-*_epi.nii*" | while read -r fmap; do
    fmap_dir=$(dirname "${fmap}")
    fmap_base=$(basename "${fmap}" .nii.gz)
    echo "---"
    echo "Found fmap: ${fmap}"

    # determine session by walking up until you hit ses-XX
    ses=$(echo "${fmap}" | sed -n 's#.*\(ses-[0-9][0-9]*\).*#\1#p' || true)
    if [ -z "${ses}" ]; then
      echo "  WARNING: could not determine session for ${fmap}. Skipping."
      continue
    fi
      echo "  session: ${ses}"

    # find functional files in same session to intended-for (we assume *bold.nii* naming)
    # Build proper BIDS-relative paths: ses-XX/func/filename
    mapfile -t func_files < <(find "${BIDS_ROOT}/${sub}/${ses}/func" -maxdepth 1 -type f -name "*_bold.nii*" 2>/dev/null)

    if [ ${#func_files[@]} -eq 0 ]; then
      echo "  No func files found in ${BIDS_ROOT}/${sub}/${ses}/func — skipping."
      continue
    fi

    # turn into proper BIDS relative paths for IntendedFor: ses-XX/func/<filename>
    intended=()
    for f in "${func_files[@]}"; do
      # Extract relative path from BIDS_ROOT/sub-XX/
      rel_path=$(realpath --relative-to="${BIDS_ROOT}/${sub}" "${f}")
      intended+=("${rel_path}")
    done

    # take matching generic JSON if it exists (e.g. dir-pa_epi.json or task-...json)
    # try to locate a generic JSON by suffix: dir-xx_epi.json
    suffix=$(echo "${fmap_base}" | sed -n 's/.*\(dir-[a-zA-Z]*_epi\).*/\1/p' || true)
    generic_json=""
    if [ -n "${suffix}" ]; then
      # e.g. /ptmp/.../dir-pa_epi.json
      if [ -f "${GENERIC_JSON_DIR}/${suffix}.json" ]; then
        generic_json="${GENERIC_JSON_DIR}/${suffix}.json"
      fi
    fi

    # fallback: try any top-level JSON (as you described)
    if [ -z "${generic_json}" ]; then
      # look for any JSON that mentions "EffectiveEchoSpacing" etc (best effort)
      generic_json=$(find "${GENERIC_JSON_DIR}" -maxdepth 1 -type f -name "*_epi.json" -print | head -n 1 || true)
    fi

    echo "  Using generic JSON: ${generic_json:-<none found>}"

    # create output JSON next to fmap (do not overwrite if exists; show message)
    out_json="${fmap_dir}/${fmap_base}.json"
    if [ -f "${out_json}" ]; then
      echo "  NOTE: JSON already exists next to fmap: ${out_json} (skipping creation)."
      continue
    fi

    # Build JSON: inherit content from generic JSON if available, add IntendedFor
    if [ -n "${generic_json}" ]; then
      python3 - "${generic_json}" "${out_json}" "${intended[@]}" <<'PY'
import json
import sys

gpath = sys.argv[1]
out = sys.argv[2]
intended = sys.argv[3:]

with open(gpath, 'r') as f:
    data = json.load(f)

# Add IntendedFor list
data['IntendedFor'] = intended

# Write pretty
with open(out, 'w') as f:
    json.dump(data, f, indent=2, sort_keys=True)

print(f"  WROTE {out}")
PY
    else
      # create minimal JSON with PE and IntendedFor. Adjust PhaseEncodingDirection/EES if you know them
      python3 - "${out_json}" "${intended[@]}" <<'PY'
import json
import sys

out = sys.argv[1]
intended = sys.argv[2:]

data = {
  "PhaseEncodingDirection": "j-",
  "EffectiveEchoSpacing": 4.2e-05,
  "IntendedFor": intended
}

with open(out, 'w') as f:
    json.dump(data, f, indent=2, sort_keys=True)

print(f"  WROTE minimal {out}")
PY
    fi

  done

done < "${SUBJECTS_FILE}"

echo "========================================="
echo "All subjects processed."
echo "========================================="