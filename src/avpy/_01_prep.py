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
import sentry_sdk
import logging
logger = logging.getLogger(__name__)

NAME = "prep"


    


def apply_threshold(img_path, threshold_min, threshold_max, binary=True, multiplier=1):
    """Apply threshold to image and optionally binarize and multiply."""
    img = nib.load(img_path)
    data = img.get_fdata()
    
    data = np.round(data).astype(int)  # Ensure data is integer type
    #logger.warning(f"Number of unique values in {img_path}: {np.unique(data, return_counts=True)}")
        
    # Apply threshold
    thresholded = np.logical_and(data >= threshold_min, data <= threshold_max).astype(int)
    
    # Optionally binarize (already done by logical_and)
    if not binary:
        thresholded = thresholded * data
    
    # Apply multiplier
    if multiplier != 1:
        thresholded = thresholded * multiplier
    
    # Create new image
    new_img = nib.Nifti1Image(thresholded, img.affine, img.header)
    return new_img


def main(path="./", debug=False):
    # Read study path

    study_path = path
    
    if debug:
        logging.basicConfig(level=logging.DEBUG) 
    
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
            
    subjects.sort()
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
        logger.debug(f"{sbj} {ii}")
        
        # Process ot files
        for xx in ['r', 'l']:
            ot_img = apply_threshold(os.path.join(ii, f"ot{xx}.nii.gz"), 10, 10, binary=True, multiplier=16)
            nib.save(ot_img, os.path.join(oo, f"ot_{xx}.nii.gz"))
            logger.debug(f"{sbj} ont")
        
        # Process onc files
        onc_r = apply_threshold(os.path.join(ii, "onc.nii.gz"), 8, 8, binary=True, multiplier=8)
        nib.save(onc_r, os.path.join(oo, "oc_r.nii.gz"))
        
        onc_l = apply_threshold(os.path.join(ii, "onc.nii.gz"), 9, 9, binary=True, multiplier=8)
        nib.save(onc_l, os.path.join(oo, "oc_l.nii.gz"))
        logger.debug(f"{sbj} onc")
        
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
            
            logger.debug(f"{sbj} oni {xx}")
                
        # Combine files
        for xx in ['r', 'l']:
            # Load all component images
            ot = nib.load(os.path.join(oo, f"ot_{xx}.nii.gz"))
            onincr = nib.load(os.path.join(oo, f"onincr_{xx}.nii.gz"))
            oninca = nib.load(os.path.join(oo, f"oninca_{xx}.nii.gz"))
            oninor = nib.load(os.path.join(oo, f"oninor_{xx}.nii.gz"))
            oc = nib.load(os.path.join(oo, f"oc_{xx}.nii.gz"))
            
            # Check overlap
            mask = ot.get_fdata() != 0
            combined_data = ot.get_fdata().copy()
            logger.debug(f"Max value in ot data: {combined_data.max()} in {ot.get_filename()}")
            
            images = [onincr, oninca, oninor, oc]
            for img in images:
                img_data = img.get_fdata()
                logger.debug(f"Max value in {img.get_filename()} data: {img_data.max()}")
                overlap = np.logical_and(mask, img_data != 0)
               
                if np.sum(overlap) > 0:
                    logger.warning(f"Overlap detected in {img.get_filename()} "+
                                   "assuming highest value.")

                combined_data += img_data
                combined_data[overlap] -= img_data[overlap]
                
                mask = np.logical_or(mask, img_data != 0)
            
            # Create combined image
            
            # Check if the affine is identity matrix
            if np.all(np.isclose(ot.affine, np.eye(4))):
                logger.warning(f"Affine for {ot.get_filename()} is identity matrix, using custom affine.")
                affine = [[-0.6, 0, 0, 74.4],
                          [0, 0.6, 0, -60.6],
                          [0, 0, 0.6, -21.],
                          [0, 0, 0, 1]]
                
            # Check if there is no translation
            elif np.all(np.isclose(ot.affine[:3, 3], 0)):
                logger.warning(f"Affine for {ot.get_filename()} has no translation, using custom affine.")
                affine = ot.affine.copy()
                zooms = np.diag(affine)[:-1]
                
                affine[0, 3] = -74.4 * np.sign(zooms[0])
                affine[1, 3] = -60.6 * np.sign(zooms[1])
                affine[2, 3] = -21.0 * np.sign(zooms[2]) 

            else:
                affine = ot.affine
            
            
            affine = np.array(affine, dtype=np.float32)
            combined_img = nib.Nifti1Image(combined_data, affine, ot.header)
            combined_path = os.path.join(oo, f"on_{xx}.nii.gz")
                                    
            target_shape = (256, 256, 72)
            
            # Check if the affine is not identity
            if not np.all(np.isclose(combined_img.affine, np.eye(4))):
                logger.warning(f"Affine is not identity for {combined_path}, resampling required.")
                # Force a transformation matrix to the image
                affine = np.eye(4)
                affine[0, 3] = -74.4 * np.sign(combined_img.affine[0, 0])
                affine[1, 3] = -60.6 * np.sign(combined_img.affine[1, 1])
                affine[2, 3] = -21.0 * np.sign(combined_img.affine[2, 2])
                combined_img = nib.Nifti1Image(combined_data, affine, combined_img.header)
            
            if combined_data.shape != target_shape:
                    
                logger.warning(f"Resampling {combined_path}")
                target_affine = img.affine * np.eye(4)
                zooms = np.diag(target_affine)[:-1]
                
                target_affine[0, 3] = -74.4 * np.sign(zooms[0])
                target_affine[1, 3] = -60.6 * np.sign(zooms[1])
                target_affine[2, 3] = -21.0 * np.sign(zooms[2])                      
                
                combined_img = image.resample_img(
                    combined_img, 
                    target_affine=target_affine, 
                    target_shape=target_shape,
                    interpolation='nearest'
                )
                            
            logger.debug(f"Max value in combined data: {combined_data.max()} in {combined_path}")
            assert combined_data.max() <= 16
            
            nib.save(combined_img, combined_path)
            logger.info(f"Created {combined_path}")
            
            
            

if __name__ == "__main__":
    main()
