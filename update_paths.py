#!/usr/bin/env python3
"""
Script to update hardcoded paths across all Python files in the project.
Replaces old /ptmp/hmueller2/2025_ibc_latent/outputs/ paths with new organized structure.
"""

import os
import re
from pathlib import Path

# Define path mappings (old -> new)
PATH_MAPPINGS = [
    # Order matters! More specific paths first
    ('/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks/derived_networks', '/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks/derived_networks'),
    ('/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_activation', '/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_activation'),
    ('/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/ppi_results_dmn_dan', '/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/ppi_results_dmn_dan'),
    ('/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/old_versions/ppi_results_fpn', '/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/old_versions/ppi_results_fpn'),
    ('/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/old_versions/ppi_results', '/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/old_versions/ppi_results'),
    ('/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_derivation', '/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_derivation'),
    ('/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_images', '/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/subnetwork_images'),
    ('/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks/old_versions/individual_networks_october', '/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks/old_versions/individual_networks_october'),
    ('/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks/old_versions', '/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks/old_versions'),
    ('/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks/network_images_wb', '/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks/network_images_wb'),
    ('/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks', '/ptmp/hmueller2/2025_ibc_latent/outputs/individual_networks'),
    ('/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/joint_mapping', '/ptmp/hmueller2/2025_ibc_latent/outputs/subnetworks/joint_mapping'),
    ('/ptmp/hmueller2/2025_ibc_latent/outputs/glm/contrast_maps_fsLR', '/ptmp/hmueller2/2025_ibc_latent/outputs/glm/contrast_maps_fsLR'),
    ('/ptmp/hmueller2/2025_ibc_latent/outputs/glm/ibc_contrast_maps', '/ptmp/hmueller2/2025_ibc_latent/outputs/glm/ibc_contrast_maps'),
    ('/ptmp/hmueller2/2025_ibc_latent/outputs/preprocessing/fmriprep_out', '/ptmp/hmueller2/2025_ibc_latent/outputs/preprocessing/fmriprep_out'),
    ('/ptmp/hmueller2/2025_ibc_latent/data/ibc_raw', '/ptmp/hmueller2/2025_ibc_latent/data/ibc_raw'),
    ('/ptmp/hmueller2/2025_ibc_latent/outputs/preprocessing/ibc_preprocessed_MNI', '/ptmp/hmueller2/2025_ibc_latent/outputs/preprocessing/ibc_preprocessed_MNI'),
    ('/ptmp/hmueller2/2025_ibc_latent/misc/all_contrasts.tsv', '/ptmp/hmueller2/2025_ibc_latent/misc/all_contrasts.tsv'),
    ('/ptmp/hmueller2/2025_ibc_latent/misc/cognitive_atlas_task_concept_mapping.csv', '/ptmp/hmueller2/2025_ibc_latent/misc/cognitive_atlas_task_concept_mapping.csv'),
    ('/ptmp/hmueller2/2025_ibc_latent/misc', '/ptmp/hmueller2/2025_ibc_latent/misc'),
    # Generic outputs mapping (last resort)
    ('/ptmp/hmueller2/2025_ibc_latent/outputs', '/ptmp/hmueller2/2025_ibc_latent/outputs'),
]

def update_file(filepath):
    """Update paths in a single Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes = []
        
        # Apply all path mappings
        for old_path, new_path in PATH_MAPPINGS:
            if old_path in content:
                count = content.count(old_path)
                content = content.replace(old_path, new_path)
                changes.append(f"  {old_path} -> {new_path} ({count} occurrences)")
        
        # Write back if changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return changes
        return None
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def main():
    """Find and update all Python files in the project."""
    project_root = Path('/home/hmueller2/ibc_code/ibc_latent')
    python_files = list(project_root.rglob('*.py'))
    
    print(f"Found {len(python_files)} Python files")
    print("=" * 80)
    
    updated_count = 0
    for filepath in python_files:
        changes = update_file(filepath)
        if changes:
            updated_count += 1
            print(f"\n✓ Updated: {filepath.relative_to(project_root)}")
            for change in changes:
                print(change)
    
    print("\n" + "=" * 80)
    print(f"Updated {updated_count} files")

if __name__ == '__main__':
    main()
