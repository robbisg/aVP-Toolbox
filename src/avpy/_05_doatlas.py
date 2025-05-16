#!/usr/bin/env python3
# 
# This script is part of the aVP-Toolbox v0.11, 12.2023 software
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
# Python version of aVP_doatlas.sh:
# Takes the on?_normalized_4.nii.gz images and binarizes them and makes a joint mask
# then divides by the number of sides for each to get percentage masks
# 
# Does the same for each iOrb, iCan, iCran, OC and OT subdivisions

import os
import nibabel as nib
import numpy as np
from nilearn import image
import logging
logger = logging.getLogger(__name__)

NAME = "doatlas"


# Function to create probability maps for a given anatomical region
def create_probability_maps(path, subjects, region_name=None, region_value=None):
    """
    Create probability maps for the given region
    
    Parameters:
    -----------
    subjects : list
        List of subject IDs
    region_name : str or None
        Name of the region (e.g., 'iOrb', 'iCan'). If None, uses whole ON.
    region_value : int or None
        Value used to threshold the region (e.g., 1 for iOrb, 2 for iCan)
    """
    prefix = f"{region_name}_" if region_name else ""
    
    im_path = os.path.join(path, "data", "proc")
    templ_path = os.path.join(path, "templates")
    norm_image_suffix = "_normalized_4bc_iso06.nii.gz"
    
    suffix = f"_norm_iso06_{region_name}" if region_name else norm_image_suffix
    
    # Initialize maps
    r_map = None
    l_map = None
    
    # Process each subject
    for sbj_idx, sbj in enumerate(subjects):
        logger.info(f"Processing subject {sbj} ({sbj_idx+1}/{len(subjects)})")
        
        for side in ['r', 'l']:
            # Prepare paths based on region
            if region_name is None:
                # For whole ON
                input_path = os.path.join(im_path, sbj, f"on{side}{suffix}")
                bin_img = image.math_img("img > 0", img=input_path)
            else:
                # For specific regions, first create binary mask if needed
                input_file = f"on{side}_normalized_4bc_iso06.nii.gz"
                input_path = os.path.join(im_path, sbj, input_file)
                output_path = os.path.join(im_path, sbj, f"{side}{suffix}")
                
                # Create binary mask for the specific region if not already done
                if not os.path.exists(output_path + ".nii.gz"):
                    threshold_img = image.math_img(f"(img == {region_value}.).astype(np.int8)", img=input_path)
                    nib.save(threshold_img, output_path + ".nii.gz")
                
                # Use the binary mask
                bin_img = nib.load(output_path + ".nii.gz")
                
            logger.info(f"Binarized image created from {input_path}.nii.gz")
            logger.debug(f"image shape: {bin_img.shape}")
            # Add to running sum based on side
            if side == 'r':
                if r_map is None:
                    r_map = bin_img.get_fdata()
                else:
                    r_map += bin_img.get_fdata()
            else:
                if l_map is None:
                    l_map = bin_img.get_fdata()
                else:
                    l_map += bin_img.get_fdata()
    
    # Save the sum maps
    affine = bin_img.affine
    header = bin_img.header
    
    # Calculate probability maps (percentage)
    r_prob = (r_map * 100) / len(subjects)
    l_prob = (l_map * 100) / len(subjects)
    combined_prob = ((r_map + l_map) * 100) / (len(subjects) * 2)
    
    # Save probability maps
    output_prefix = f"{region_name}_prob" if region_name else "aVP_prob"
    nib.save(nib.Nifti1Image(r_prob, affine, header), os.path.join(templ_path, f"{output_prefix}_r.nii.gz"))
    nib.save(nib.Nifti1Image(l_prob, affine, header), os.path.join(templ_path, f"{output_prefix}_l.nii.gz"))
    nib.save(nib.Nifti1Image(combined_prob, affine, header), os.path.join(templ_path, f"{output_prefix}.nii.gz"))
    
    return len(subjects)



def main(path="./"):
    # Read study path from ONcontrol.txt
    study_path = path

    # Set paths
    im_path = os.path.join(study_path, "data", "proc")
    templ_path = os.path.join(study_path, "templates")
    sbj_list_path = os.path.join(study_path, "data", "sbj.list")

    # Create template directory if it doesn't exist
    os.makedirs(templ_path, exist_ok=True)

    # Define normalized image suffix
    norm_image_suffix = "_normalized_4bc_iso06.nii.gz"

    # Read subject list
    with open(sbj_list_path, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]

    logger.debug(f"Total subjects: {len(subjects)}")

    # Step 1: Create the main probability maps (whole ON)
    sbj_count = create_probability_maps(study_path, subjects)
    logger.info(f"Atlas created from {sbj_count} subjects. Now generating anatomical subdivision probability masks...")

    # Create binary masks for each anatomical subdivision
    for sbj in subjects:
        sbj_dir = os.path.join(im_path, sbj)
        logger.info(f"Splitting {sbj} into anatomical subdivisions")
        
        # Process left and right
        for side in ['l', 'r']:
            input_file = f"on{side}{norm_image_suffix}"
            input_path = os.path.join(sbj_dir, input_file)
            
            # Define regions and their values
            regions = {
                'iOrb': 1,
                'iCan': 2,
                'iCran': 4,
                'OC': 8,
                'OT': 16
            }
            
            # Create binary mask for each region
            for region_name, region_value in regions.items():
                output_path = os.path.join(sbj_dir, f"{side}_norm_iso06_{region_name}.nii.gz")
                if not os.path.exists(output_path):
                    # Create binary mask where voxels equal to region_value are set to 1
                    threshold_img = image.math_img(f"(img == {region_value}.).astype(np.int8)", img=input_path)
                    nib.save(threshold_img, output_path)

    # Step 2: Create probability maps for each anatomical subdivision
    regions = {
        'iOrb': 1,
        'iCan': 2,
        'iCran': 4,
        'OC': 8,
        'OT': 16
    }

    for region_name, region_value in regions.items():
        sbj_count = create_probability_maps(study_path, subjects, region_name, region_value)
        logger.info(f"{region_name} probability mask created from {sbj_count} subjects.")

    logger.info("All probability maps have been created successfully.")


if __name__ == "__main__":
    main()