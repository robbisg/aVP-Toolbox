#!/usr/bin/env python3
# 
# This script is part of the aVP-Toolbox v0.11 - 2023 software.
# Python version of the original bash script aVP_resample.sh
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
from pathlib import Path
from nilearn.image import resample_img
import logging
logger = logging.getLogger(__name__)

NAME = "resample"

def process_images(study_path, base_image, debug=True):
    """Process images for all subjects and hemispheres."""
    im_path = os.path.join(study_path, "data", "proc")
    anat = "on"
    
    # Read subject list
    with open(os.path.join(study_path, "data", "sbj.list"), 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]
        
    if debug:
        normalized_l = []
        normalized_r = []
    
    for sbj in subjects:
        for ss in ['r', 'l']:
            # Define file paths
            input_file = os.path.join(im_path, sbj, f"{anat}{ss}{base_image}")
            input_basename = os.path.basename(input_file).replace(".nii.gz", "")
            output_dir = os.path.dirname(input_file)
            output_file = os.path.join(output_dir, f"{input_basename}_iso06.nii.gz")
            
            logger.info(f"Processing: {input_file}")
            logger.debug(f"Output basename: {input_basename}")
            logger.debug(f"Output directory: {output_dir}")
            
            # Load the input image
            img = nib.load(input_file)
            
            # Set qform and sform codes to 1 (NIFTI_XFORM_SCANNER_ANAT)
            header = img.header.copy()
            header['qform_code'] = 1
            header['sform_code'] = 1
                        
            # Create a new image with modified header
            modified_img = nib.Nifti1Image(img.get_fdata(), img.affine, header)
            
            # Resample to isotropic 0.6mm resolution
            # This replaces flirt -applyisoxfm 0.6
            resampling = np.array([0.6, 0.6, 0.6, 1]) * np.sign(np.diag(img.affine))
            
            target_affine = np.diag(resampling)
            target_affine[0, 3] = img.affine[0, 3]
            target_affine[1, 3] = img.affine[1, 3]
            target_affine[2, 3] = img.affine[2, 3]
            
            # Use nearest neighbor interpolation for discrete data
            resampled_img = resample_img(
                modified_img,
                target_shape=(250, 102, 72),
                target_affine=target_affine,
                interpolation='nearest',
                force_resample=False
            )
            
            if debug:
                if ss == 'r':
                    normalized_r.append(resampled_img.get_fdata())
                else:
                    normalized_l.append(resampled_img.get_fdata())
            
            # Set the sform and qform properly
            final_img = nib.Nifti1Image(resampled_img.get_fdata(), target_affine)
            final_img.header['qform_code'] = 1
            final_img.header['sform_code'] = 1
            
            # Save the resampled image
            nib.save(final_img, output_file)
            logger.info(f"Saved: {output_file}")
    
    if debug:
        
        normalized_l = np.stack(normalized_l, axis=-1)
        normalized_r = np.stack(normalized_r, axis=-1)
        
        normalized_l_img = nib.Nifti1Image(normalized_l, resampled_img.affine)
        normalized_r_img = nib.Nifti1Image(normalized_r, resampled_img.affine)
        
        nib.save(normalized_l_img, os.path.join(im_path, "total_l"+base_image))
        nib.save(normalized_r_img, os.path.join(im_path, "total_r"+base_image))
        

def main(path="./", debug=False):
    # Read study path from ONcontrol.txt
    study_path = path
    
    if debug:
        logging.basicConfig(level=logging.DEBUG)    
    
    # Process linearized images
    logger.info("Processing linearized images...")
    process_images(study_path, "_linearize_4bc.nii.gz")
    
    # Process normalized images
    logger.info("Processing normalized images...")
    process_images(study_path, "_normalized_4bc.nii.gz")

if __name__ == "__main__":
    main()
