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
# Segmentation preparation module: handles opening original images and building 
# prep images from manual segmentations using thresholds.

import os
import nibabel as nib
import numpy as np
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

NAME = "segmentation_prep"


def apply_threshold(img_path, threshold_min, threshold_max, binary=True, multiplier=1):
    """Apply threshold to image and optionally binarize and multiply."""
    img = nib.load(img_path)
    data = img.get_fdata()
    
    data = np.round(data).astype(int)  # Ensure data is integer type
        
    # Apply threshold
    thresholded = np.logical_and(data >= threshold_min, data <= threshold_max).astype(int)
    
    # Optionally binarize (already done by logical_and)
    if not binary:
        thresholded = thresholded * data
    
    # Apply multiplier
    if multiplier != 1:
        thresholded = thresholded * multiplier
    
    # Create new image with original affine and header
    new_img = nib.Nifti1Image(thresholded, img.affine, img.header)
    return new_img


def process_subject_segmentations(input_dir, output_dir, subject):
    """Process all segmentation files for a single subject."""
    sbj_input = os.path.join(input_dir, subject)
    sbj_output = os.path.join(output_dir, subject)
    os.makedirs(sbj_output, exist_ok=True)
    
    logger.debug(f"Processing {subject} from {sbj_input}")
    
    # Process ot files (optic tract)
    for side in ['r', 'l']:
        ot_img = apply_threshold(os.path.join(sbj_input, f"ot{side}.nii.gz"), 10, 10, binary=True, multiplier=16)
        nib.save(ot_img, os.path.join(sbj_output, f"ot_{side}.nii.gz"))
        logger.debug(f"{subject} ot{side}")
    
    # Process onc files (optic nerve chiasm)
    onc_r = apply_threshold(os.path.join(sbj_input, "onc.nii.gz"), 8, 8, binary=True, multiplier=8)
    nib.save(onc_r, os.path.join(sbj_output, "oc_r.nii.gz"))
    
    onc_l = apply_threshold(os.path.join(sbj_input, "onc.nii.gz"), 9, 9, binary=True, multiplier=8)
    nib.save(onc_l, os.path.join(sbj_output, "oc_l.nii.gz"))
    logger.debug(f"{subject} onc")
    
    # Process on files (optic nerve components)
    for side in ['r', 'l']:
        # oninor (orbital part)
        oninor = apply_threshold(os.path.join(sbj_input, f"on{side}.nii.gz"), 2, 2, binary=True, multiplier=1)
        nib.save(oninor, os.path.join(sbj_output, f"oninor_{side}.nii.gz"))
        
        # oninca (canalicular part)
        oninca = apply_threshold(os.path.join(sbj_input, f"on{side}.nii.gz"), 4, 4, binary=True, multiplier=2)
        nib.save(oninca, os.path.join(sbj_output, f"oninca_{side}.nii.gz"))
        
        # onincr (cranial part)
        onincr = apply_threshold(os.path.join(sbj_input, f"on{side}.nii.gz"), 6, 6, binary=True, multiplier=4)
        nib.save(onincr, os.path.join(sbj_output, f"onincr_{side}.nii.gz"))
        
        logger.debug(f"{subject} on components {side}")


def combine_segmentation_components(output_dir, subject):
    """Combine individual segmentation components into unified segmentations."""
    sbj_output = os.path.join(output_dir, subject)
    
    for side in ['r', 'l']:
        # Load all component images
        ot = nib.load(os.path.join(sbj_output, f"ot_{side}.nii.gz"))
        onincr = nib.load(os.path.join(sbj_output, f"onincr_{side}.nii.gz"))
        oninca = nib.load(os.path.join(sbj_output, f"oninca_{side}.nii.gz"))
        oninor = nib.load(os.path.join(sbj_output, f"oninor_{side}.nii.gz"))
        oc = nib.load(os.path.join(sbj_output, f"oc_{side}.nii.gz"))
        
        # Check overlap and combine
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
        combined_img = nib.Nifti1Image(combined_data, ot.affine, ot.header)
        combined_path = os.path.join(sbj_output, f"on_{side}.nii.gz")
        
        logger.debug(f"Max value in combined data: {combined_data.max()} in {combined_path}")
        assert combined_data.max() <= 16
        
        # Save the basic combined image (affine processing will be done separately)
        nib.save(combined_img, combined_path)
        logger.info(f"Created {combined_path}")
        
        yield combined_img, combined_path


def main(path="./", debug=False):
    """Main function for segmentation preprocessing."""
    if debug:
        logging.basicConfig(level=logging.DEBUG) 
    
    study_path = path
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
    for subject in subjects:
        # Process individual segmentation components
        process_subject_segmentations(in_path, out_path, subject)
        
        # Combine components (affine processing will be done separately)
        for combined_img, combined_path in combine_segmentation_components(out_path, subject):
            # Basic segmentation combination is complete
            pass


if __name__ == "__main__":
    main()