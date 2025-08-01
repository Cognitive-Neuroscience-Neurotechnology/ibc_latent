# MSCcodebase Utilities

This directory contains the Utilities folder from the MSCcodebase repository, which provides various tools and utilities for brain data analysis.

## Source Repository

The utilities were imported from the [MSCcodebase repository](https://github.com/MidnightScanClub/MSCcodebase) maintained by the Midnight Scan Club.

## Import Process

The MSCcodebase/Utilities folder was imported as a subtree on [DATE] to preserve the commit history from the original repository. The import was performed using the following process:

```bash
# Add the MSCcodebase repository as a remote
git remote add msc-subtree https://github.com/MidnightScanClub/MSCcodebase.git

# Fetch the remote repository
git fetch msc-subtree

# Add the entire repository as a temporary subtree 
git subtree add --prefix=MSCcodebase-temp msc-subtree master --squash

# Move only the Utilities folder to the target location
mv MSCcodebase-temp/Utilities MSCcodebase/Utilities

# Clean up temporary files and remote
rm -rf MSCcodebase-temp
git remote remove msc-subtree

# Add and commit the changes
git add MSCcodebase/
git commit -m "Add MSCcodebase/Utilities as subtree"
```

## Contents

The Utilities folder contains:

- **Conte69_atlas-v2.LR.32k_fs_LR.wb/**: Conte69 atlas files for 32k fs_LR resolution
- **Conte69_atlas.LR.164k_fs_LR/**: Conte69 atlas files for 164k fs_LR resolution  
- **Infomap_wrapper/**: MATLAB functions for community detection using Infomap
- **Parcellation/**: Tools for surface parcellation and watershed algorithms
- **read_write_cifti/**: Utilities for reading and writing CIFTI files
- **Various MATLAB scripts**: Including functions for:
  - `cifti_neighbors.m`: Finding neighbors in CIFTI data
  - `export_fig.m`: Enhanced figure export functionality
  - `paircorr_mod.m`: Pairwise correlation calculations
  - `set_cifti_powercolors.m`: Color mapping for CIFTI visualizations
  - And many more analysis utilities

## Usage

These utilities are primarily MATLAB-based tools designed for:
- Brain surface analysis and visualization
- CIFTI file manipulation
- Network analysis and community detection
- Surface parcellation
- Atlas-based analyses using the Conte69 surface

## License

Please refer to the original MSCcodebase repository for licensing information.

## Updates

To update the utilities from the source repository, you would need to use git subtree pull:

```bash
git subtree pull --prefix=MSCcodebase/Utilities https://github.com/MidnightScanClub/MSCcodebase.git master --squash
```

Note: Due to the selective import process used, future updates may require manual intervention to maintain the correct directory structure.