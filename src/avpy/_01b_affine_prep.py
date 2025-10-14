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
# Affine preparation module: handles resampling and affine transformations,
# managing sform and qform issues in NIfTI images.

import os
import nibabel as nib
import numpy as np
from nilearn import image
import logging
logger = logging.getLogger(__name__)

NAME = "affine_prep"


def fix_affine_orientation(img, target_translation=(-74.4, -60.6, -21.0)):
    """Fix affine matrix orientation and translation."""
    affine = img.affine.copy()
    
    # Adjust translation based on the sign of diagonal elements
    affine[0, 3] = target_translation[0] * np.sign(affine[0, 0])  # x translation
    affine[1, 3] = target_translation[1] * np.sign(affine[1, 1])  # y translation  
    affine[2, 3] = target_translation[2] * np.sign(affine[2, 2])  # z translation
    
    return affine


def check_and_fix_sform_qform(img):
    """Check and fix sform/qform issues in NIfTI images."""
    header = img.header
    
    # Check sform and qform codes
    sform_code = header.get_sform(coded=True)[1]
    qform_code = header.get_qform(coded=True)[1]
    
    logger.debug(f"Original sform code: {sform_code}, qform code: {qform_code}")
    
    # Ensure proper sform/qform alignment
    if sform_code == 0 and qform_code > 0:
        # No sform, copy qform to sform
        qform, _ = header.get_qform(coded=True)
        header.set_sform(qform, code=qform_code)
        logger.debug("Copied qform to sform")
    elif qform_code == 0 and sform_code > 0:
        # No qform, copy sform to qform  
        sform, _ = header.get_sform(coded=True)
        header.set_qform(sform, code=sform_code)
        logger.debug("Copied sform to qform")
    elif sform_code == 0 and qform_code == 0:
        # Neither form is set, use affine matrix
        header.set_sform(img.affine, code=1)
        header.set_qform(img.affine, code=1)
        logger.debug("Set both sform and qform from affine")
    
    return nib.Nifti1Image(img.get_fdata(), img.affine, header)


def resample_to_isotropic(img, target_resolution=0.6):
    """Resample image to isotropic voxels if needed."""
    target_voxel_size = (target_resolution, target_resolution, target_resolution)
    
    if not np.allclose(img.header.get_zooms()[:3], target_voxel_size):
        logger.warning(f"Resampling from {img.header.get_zooms()} to {target_voxel_size}")
                        
        affine = img.affine.copy()
        affine[0, 0] = target_resolution * np.sign(img.affine[0, 0])
        affine[1, 1] = target_resolution * np.sign(img.affine[1, 1])
        affine[2, 2] = target_resolution * np.sign(img.affine[2, 2])
        
        resampled_img = image.resample_img(img, 
                                          target_affine=affine, 
                                          interpolation='nearest')
        return resampled_img
    
    return img


def resample_to_target_shape(img, target_shape=(256, 256, 72)):
    """Resample image to target shape if needed."""
    current_shape = img.get_fdata().shape
    
    if current_shape != target_shape:
        logger.warning(f"Reshaping from {current_shape} to {target_shape}")
        
        affine = img.affine.copy()
        zooms = np.diag(affine)[:-1]
        
        # Adjust translation for target shape
        affine[0, 3] = -76.8 * np.sign(zooms[0])
        affine[1, 3] = -60.6 * np.sign(zooms[1])
        affine[2, 3] = -21.0 * np.sign(zooms[2])
        
        resampled_img = image.resample_img(
            img, 
            target_affine=affine, 
            target_shape=target_shape,
            interpolation='nearest',
            force_resample=False,
        )
        
        return resampled_img
    
    return img


def validate_affine_consistency(img):
    """Validate that sform and qform are consistent."""
    header = img.header
    sform, sform_code = header.get_sform(coded=True)
    qform, qform_code = header.get_qform(coded=True)
    
    if sform_code > 0 and qform_code > 0:
        # Both forms are present, check consistency
        if not np.allclose(sform, qform, atol=1e-3):
            logger.warning("sform and qform are not consistent")
            # Prefer sform over qform for spatial consistency
            header.set_qform(sform, code=sform_code)
            logger.debug("Updated qform to match sform")
    
    return nib.Nifti1Image(img.get_fdata(), img.affine, header)


def process_affine_transformations(img):
    """
    Process all affine transformations for a NIfTI image.
    
    This function handles:
    1. sform/qform consistency checks
    2. Affine orientation fixes
    3. Resampling to isotropic voxels (0.6mm)
    4. Resampling to target shape (256, 256, 72)
    5. Final validation
    """
    # Step 1: Fix sform/qform issues
    img = check_and_fix_sform_qform(img)
    
    # Step 2: Fix affine orientation
    fixed_affine = fix_affine_orientation(img)
    img = nib.Nifti1Image(img.get_fdata(), fixed_affine, img.header)
    
    # Step 3: Resample to isotropic voxels if needed
    img = resample_to_isotropic(img)
    
    # Step 4: Resample to target shape if needed  
    img = resample_to_target_shape(img)
    
    # Step 5: Final validation
    img = validate_affine_consistency(img)
    
    return img


def process_nifti_file(input_path, output_path):
    """Process a single NIfTI file with affine transformations."""
    img = nib.load(input_path)
    processed_img = process_affine_transformations(img)
    nib.save(processed_img, output_path)
    logger.info(f"Processed {input_path} -> {output_path}")
    return processed_img


def main(path="./", debug=False):
    """
    Main function for affine preprocessing.
    Can be used standalone to process all NIfTI files in a directory structure.
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    
    study_path = path
    proc_path = os.path.join(study_path, "data", "proc")
    
    if not os.path.exists(proc_path):
        logger.error(f"Processed data directory not found: {proc_path}")
        return
    
    # Get subject list
    sbj_list_path = os.path.join(study_path, "data", "sbj.list")
    if not os.path.exists(sbj_list_path):
        logger.error(f"Subject list not found: {sbj_list_path}")
        return
    
    with open(sbj_list_path, 'r') as f:
        subjects = [line.strip() for line in f.readlines()]
    
    # Process each subject's files
    for subject in subjects:
        sbj_dir = os.path.join(proc_path, subject)
        if not os.path.exists(sbj_dir):
            logger.warning(f"Subject directory not found: {sbj_dir}")
            continue
        
        # Process all NIfTI files in subject directory
        for filename in os.listdir(sbj_dir):
            if filename.endswith('.nii.gz'):
                input_path = os.path.join(sbj_dir, filename)
                output_path = input_path  # Process in-place
                
                try:
                    process_nifti_file(input_path, output_path)
                except Exception as e:
                    logger.error(f"Error processing {input_path}: {e}")


if __name__ == "__main__":
    main()