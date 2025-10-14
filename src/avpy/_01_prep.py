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
from . import _01a_segmentation_prep, _01b_affine_prep

logger = logging.getLogger(__name__)
NAME = "prep"


def main(path="./", debug=False):
    """
    Main preparation function that coordinates segmentation and affine preprocessing.
    
    This function runs both segmentation preprocessing (_01a) and affine preprocessing (_01b)
    in sequence to maintain compatibility with the existing pipeline.
    """
    logger.info("Starting preparation pipeline...")
    
    # Step 1: Run segmentation preprocessing
    logger.info("Step 1: Running segmentation preprocessing...")
    _01a_segmentation_prep.main(path=path, debug=debug)
    
    # Step 2: Run affine preprocessing  
    logger.info("Step 2: Running affine preprocessing...")
    _01b_affine_prep.main(path=path, debug=debug)
    
    logger.info("Preparation pipeline completed successfully.")

if __name__ == "__main__":
    main()
