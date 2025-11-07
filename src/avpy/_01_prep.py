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
# Main preparation coordinator: orchestrates segmentation and affine preprocessing.

import os
import logging
import nibabel as nib
import numpy as np
import pandas as pd
from . import _01a_segmentation_prep, _01b_affine_prep

logger = logging.getLogger(__name__)
NAME = "prep"

#loggerfile = "prep.log"
#
## define a Handler which writes INFO messages or higher to the sys.stderr
#filelogger = logging.FileHandler(filename=loggerfile, )
#filelogger.setLevel(logging.DEBUG)
#
#logger.addHandler(filelogger)
#
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')   
#filelogger.setFormatter(formatter)

def _00_sanity_check(path='/.', debug=False):
    # Check if all affine between subjects are consistent
    
    if debug:
        logger.setLevel(level=logging.DEBUG) 
    
    study_path = path
    in_path = os.path.join(study_path, "data", "orig")
    
    if not os.path.exists(os.path.join(study_path, 'data', 'sbj.list')):
        subjects = os.listdir(in_path)
    else:    
        with open(os.path.join(study_path, 'data', 'sbj.list'), 'r') as fileID:
            subjects = [line.strip() for line in fileID]
        
    affine_list = []
    # Process each subject
    for subject in subjects:
        # Process individual segmentation components
        sbj_input = os.path.join(in_path, subject)
        logger.debug(f"Sanity checking {subject} in {sbj_input}")
        
        # Loop through all files in the subject directory
        for file_name in os.listdir(sbj_input):
            if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
                file_path = os.path.join(sbj_input, file_name)
                img = nib.load(file_path)
                affine = img.affine                
                # Here you would implement checks to compare affines
                # For simplicity, we just log the affine matrix
                logger.debug(f"Affine for {file_name} in {subject}: \n{affine}")
    
                affine_list.append(
                    {
                        'subject': subject,
                        'file_name': file_name,
                        'origin_x': affine[0, 3],
                        'origin_y': affine[1, 3],
                        'origin_z': affine[2, 3],
                        'resolution_x': affine[0, 0],
                        'resolution_y': affine[1, 1],
                        'resolution_z': affine[2, 2],
                        'dim_x': img.header.get_data_shape()[0],
                        'dim_y': img.header.get_data_shape()[1],
                        'dim_z': img.header.get_data_shape()[2],
                        'affine': affine
                    }
                )
    
    # After collecting all affines, you could implement consistency checks here
    # Check if affine are all the same across subjects for the same file_name
    
    reference_affine = affine_list[0]['affine']
    for item in affine_list:
        
        aff = item['affine']
        fname = item['file_name']
        subj = item['subject']
        
        resolutions = np.diag(aff)[:3]
        if not np.allclose(resolutions, np.diag(reference_affine)[:3]):
            logger.warning(f"Resolution mismatch found in {fname} for subject {subj}")
        
        origin = aff[:3, 3]
        if not np.allclose(origin, reference_affine[:3, 3]):
            logger.warning(f"Origin mismatch found in {fname} for subject {subj}")
        
        if not np.array_equal(aff, reference_affine):
            logger.warning(f"Affine mismatch found in {fname} for subject {subj}")
    
        item.pop('affine')  # Remove affine from final report to keep it clean
    
    logger.info("Sanity check completed.")
    
    affine_report = pd.DataFrame(affine_list)
    report_path = os.path.join(study_path, 'affine_sanity_check.csv')
    affine_report.to_csv(report_path, index=False)
    logger.info(f"Affine sanity check report saved to {report_path}")
    
    return



def main(path="./", debug=False):
    """
    Main preparation function that coordinates segmentation and affine preprocessing.
    
    This function runs both segmentation preprocessing (_01a) and affine preprocessing (_01b)
    in sequence to maintain compatibility with the existing pipeline.
    """
    logger.info("Starting preparation pipeline...")
    
    logger.info("Step 0: Sanity check of affine ")
    _00_sanity_check(path=path, debug=debug)
    
    # Step 1: Run segmentation preprocessing
    logger.info("Step 1: Running segmentation preprocessing...")
    _01a_segmentation_prep.main(path=path, debug=debug)
    
    # Step 2: Run affine preprocessing  
    logger.info("Step 2: Running affine preprocessing...")
    _01b_affine_prep.main(path=path, debug=debug)
    
    logger.info("Preparation pipeline completed successfully.")

if __name__ == "__main__":
    main()
