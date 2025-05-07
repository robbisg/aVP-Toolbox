#!/usr/bin/env python3
# 
# This script is part of aVP-Toolbox v0.11 - 2023. 
# Python implementation of aVP_basics.sh
#
# aVP-Toolbox ("The software") is licensed under the Creative Commons Attribution 4.0 International License, 
# permitting use, sharing, adaptation, distribution and reproduction in any medium or format, 
# as long as you give appropriate credit to the original author(s) and the source, provide 
# a link to the Creative Commons licence, and indicate if changes were made. 
# The licensor offers the Licensed Material as-is and as-available, and makes no 
# representations or warranties of any kind concerning the Licensed Material, 
# whether express, implied, statutory, or other. This includes, without limitation, 
# warranties of title, merchantability, fitness for a particular purpose, non-infringement, 
# absence of latent or other defects, accuracy, or the presence or absence of errors, 
# whether or not known or discoverable. Where disclaimers of warranties are not allowed 
# in full or in part, this disclaimer may not apply to You. 
# Please go to http://creativecommons.org/licenses/by/4.0/ to view a complete copy of this licence.

import os
import numpy as np
import nibabel as nib
import csv
from pathlib import Path

NAME = "basics"

def main(path='./'):
    # Read study path from ONcontrol.txt
    with open(os.path.join(path,'ONcontrol.txt'), 'r') as f:
        study_path = f.read().strip()

    # Set up paths
    in_path = f"{study_path}/data/proc"
    out_path = f"{study_path}/results"
    out_file = "volume_py_version.csv"

    # Create output directory if it doesn't exist
    os.makedirs(out_path, exist_ok=True)

    # Initialize output CSV file
    with open(os.path.join(out_path, out_file), 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["Subject", "NerveSegment", "Side", "NumberVoxels", "Volume"])

    # Read subject list
    with open(f"{study_path}/data/sbj.list", 'r') as f:
        subjects = [line.strip() for line in f]

    # Process each subject
    for sbj in subjects:
        ii = os.path.join(in_path, sbj)
        
        # For each side
        for ss in ['r', 'l']:
            # For each nerve segment
            for nn in ['ot', 'oc', 'onincr', 'oninca', 'oninor']:
                mask_file = os.path.join(ii, f"{nn}_{ss}.nii.gz")
                
                # Check if file exists
                if os.path.exists(mask_file):
                    # Load the mask
                    img = nib.load(mask_file)
                    data = img.get_fdata()
                    
                    # Calculate voxel volume in mm³
                    voxel_size = np.prod(img.header.get_zooms())
                    
                    # Calculate number of voxels (non-zero values in the mask)
                    num_voxels = np.count_nonzero(data)
                    
                    # Calculate volume
                    volume = num_voxels * voxel_size
                    
                    print(mask_file)
                    print(f"{sbj};{nn};{ss};{num_voxels};{volume}")
                    
                    # Write to CSV file
                    with open(os.path.join(out_path, out_file), 'a', newline='') as f:
                        writer = csv.writer(f, delimiter=';')
                        writer.writerow([sbj, nn, ss, num_voxels, volume])
                else:
                    print(f"Warning: File {mask_file} does not exist.")
                
if __name__ == "__main__":
    main()
