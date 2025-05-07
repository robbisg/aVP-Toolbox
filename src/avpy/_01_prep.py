#!/usr/bin/env python3
# 
# This script is part of the aVP-Toolbox v0.11 - 2023 software. 
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
#
# Use thresholds to break-down the manual segmentations into a number of 
# optic nerve component segmentations, as well as a new unified segmentation.

import os
import nibabel as nib
import numpy as np
from nilearn import image
from pathlib import Path

NAME = "prep"

def apply_threshold(img_path, threshold_min, threshold_max, binary=True, multiplier=1):
    """Apply threshold to image and optionally binarize and multiply."""
    img = nib.load(img_path)
    data = img.get_fdata()
    
    # Apply threshold
    thresholded = np.logical_and(data >= threshold_min, data <= threshold_max).astype(float)
    
    # Optionally binarize (already done by logical_and)
    if not binary:
        thresholded = thresholded * data
    
    # Apply multiplier
    if multiplier != 1:
        thresholded = thresholded * multiplier
    
    # Create new image
    new_img = nib.Nifti1Image(thresholded, img.affine, img.header)
    return new_img

def main(path="./"):
    # Read study path
    with open(os.path.join(path,'ONcontrol.txt'), 'r') as f:
        study_path = f.read().strip()
    
    in_path = os.path.join(study_path, "data", "orig")
    out_path = os.path.join(study_path, "data", "proc")
    
    # Create subject list
    sbj_list_path = os.path.join(study_path, "data", "sbj.list")
    if os.path.exists(sbj_list_path):
        os.remove(sbj_list_path)
    
    subjects = []
    for item in os.listdir(in_path):
        if os.path.isdir(os.path.join(in_path, item)):
            subjects.append(item)
    
    # Write subject list
    with open(sbj_list_path, 'w') as f:
        for sbj in subjects:
            f.write(f"{sbj}\n")
    
    # Create output directories
    os.makedirs(os.path.join(study_path, "results"), exist_ok=True)
    os.makedirs(out_path, exist_ok=True)
    
    # Process each subject
    for sbj in subjects:
        ii = os.path.join(in_path, sbj)
        oo = os.path.join(out_path, sbj)
        os.makedirs(oo, exist_ok=True)
        print(f"{sbj} {ii}")
        
        # Process ot files
        for xx in ['r', 'l']:
            ot_img = apply_threshold(os.path.join(ii, f"ot{xx}.nii.gz"), 10, 10, binary=True, multiplier=16)
            nib.save(ot_img, os.path.join(oo, f"ot_{xx}.nii.gz"))
            print(f"{sbj} ont")
        
        # Process onc files
        onc_r = apply_threshold(os.path.join(ii, "onc.nii.gz"), 8, 8, binary=True, multiplier=8)
        nib.save(onc_r, os.path.join(oo, "oc_r.nii.gz"))
        
        onc_l = apply_threshold(os.path.join(ii, "onc.nii.gz"), 9, 9, binary=True, multiplier=8)
        nib.save(onc_l, os.path.join(oo, "oc_l.nii.gz"))
        print(f"{sbj} onc")
        
        # Process on files
        for xx in ['r', 'l']:
            # oninor
            oninor = apply_threshold(os.path.join(ii, f"on{xx}.nii.gz"), 2, 2, binary=True, multiplier=1)
            nib.save(oninor, os.path.join(oo, f"oninor_{xx}.nii.gz"))
            
            # oninca
            oninca = apply_threshold(os.path.join(ii, f"on{xx}.nii.gz"), 4, 4, binary=True, multiplier=2)
            nib.save(oninca, os.path.join(oo, f"oninca_{xx}.nii.gz"))
            
            # onincr
            onincr = apply_threshold(os.path.join(ii, f"on{xx}.nii.gz"), 6, 6, binary=True, multiplier=4)
            nib.save(onincr, os.path.join(oo, f"onincr_{xx}.nii.gz"))
            print(f"{sbj} oni {xx}")
        
        # List output files
        print(os.listdir(oo))
        
        # Combine files
        for xx in ['r', 'l']:
            # Load all component images
            ot = nib.load(os.path.join(oo, f"ot_{xx}.nii.gz"))
            onincr = nib.load(os.path.join(oo, f"onincr_{xx}.nii.gz"))
            oninca = nib.load(os.path.join(oo, f"oninca_{xx}.nii.gz"))
            oninor = nib.load(os.path.join(oo, f"oninor_{xx}.nii.gz"))
            oc = nib.load(os.path.join(oo, f"oc_{xx}.nii.gz"))
            
            # Add them together
            combined_data = (ot.get_fdata() + 
                            onincr.get_fdata() + 
                            oninca.get_fdata() + 
                            oninor.get_fdata() + 
                            oc.get_fdata())
            
            # Create combined image
            combined_img = nib.Nifti1Image(combined_data, ot.affine, ot.header)
            combined_path = os.path.join(oo, f"on_{xx}.nii.gz")
            nib.save(combined_img, combined_path)
            print(combined_path)

if __name__ == "__main__":
    main()
