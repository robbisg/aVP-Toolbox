#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is part of the aVP-Toolbox v0.11 - 2023 software.

aVP-Toolbox ("The software") is licensed under the Creative Commons Attribution 4.0 International License,
permitting use, sharing, adaptation, distribution and reproduction in any medium or format,
as long as you give appropriate credit to the original author(s) and the source, provide
a link to the Creative Commons licence, and indicate if changes were made.
The licensor offers the Licensed Material as-is and as-available, and makes no
representations or warranties of any kind concerning the Licensed Material,
whether express, implied, statutory, or other. This includes, without limitation,
warranties of title, merchantability, fitness for a particular purpose, non-infringement,
absence of latent or other defects, accuracy, or the presence or absence of errors,
whether or not known or discoverable. Where disclaimers of warranties are not allowed
in full or in part, this disclaimer may not apply to You.
Please go to http://creativecommons.org/licenses/by/4.0/ to view a complete copy of this licence.

Reads in:
- nifti files of
   axial segmentation masks:   $inPATH/SubjectID/on_SIDE_nii.gz

Calls 'aVP_resample.sh'

Produces:
- nifti files 10-fold expanded in AP (y) direction
   straightened (centered) aVP segments:   _linearize_4.nii.gz
   normalized (interpolated for length conservation) aVP segments: _normalized_4.nii.gz
- Python pickle file of the normalized aVP segments: _normalized.pkl
- text file listings of
   data values in each slice
   holes
   range
   length, CSA in each aVP anatomical section
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
from skimage import measure
import sys
import sentry_sdk
import logging
logger = logging.getLogger(__name__)

NAME = "normalize"

def main(path="./"):
    # Get current working directory
    StudyPath = path
        
    #StudyPath = "/home/robbis/git/aVP-toolbox/data/test/"

    inPath = os.path.join(StudyPath, 'data', 'proc')
    outImPath = os.path.join(StudyPath, 'data', 'proc')
    outResPath = os.path.join(StudyPath, 'results')

    # Output file paths
    DataFile = os.path.join(outResPath, 'aVP_slice_data.xlsx')


    # Define image output file names
    Lin4image = '_linearize_4bc.nii.gz'
    Norm4image = '_normalized_4bc.nii.gz'

    sides = ['l', 'r']
    isbj = 0

    # 0: use combination of aVP segments. 1: use individual aVP segments (not tested)
    ismask = 0

    # Read subject list
    with open(os.path.join(StudyPath, 'data', 'sbj.list'), 'r') as fileID:
        subject_list = [line.strip() for line in fileID]

    resolution_increase = 10
    max_slices = 160 * resolution_increase

    dataframe = []

    # Main processing loop
    # TODO: Consider to use multiprocessing for parallel processing
    # TODO: Consider to use a progress bar for better user experience
    # TODO: Consider to create a subject function to encapsulate the logic


    segment_types = {
        16: "OT",
        8: "OC",
        4: "iCran",
        2: "iCan",
        1: "iOrb"
    }

    for subject in subject_list:
        isbj += 1
        
        for side_idx, side in enumerate(sides):
            fname = os.path.join(inPath, subject, f'on_{side}')
            
            logger.info(f"Analyzing {subject} - on_{side}")
                    
            if not os.path.exists(f"{fname}.nii.gz"):
                logger.error(f"could not find: {fname}.nii.gz")
                sys.exit(1)
            
            # Load the nifti file
            nifti_img = nib.load(f"{fname}.nii.gz")
            nifti_data = nifti_img.get_fdata(dtype=np.float32)
            
            # Check if the image is empty
            if np.sum(nifti_data) == 0:
                logger.warning(f"Image {fname}.nii.gz is empty.")
                continue
            
            # Get dimensions and resolutions
            x_dim, y_dim, z_dim = nifti_data.shape
            x_resolution = nifti_img.header.get_zooms()[0]
            y_resolution = nifti_img.header.get_zooms()[1]
            z_resolution = nifti_img.header.get_zooms()[2]
            
            # Calculate the center of the slice to be able to shift the centroid of the ROI there
            image_center = np.array([x_dim/2, z_dim/2]) - 0.5
            active_slice = -1
            segment_type = 0
            
            table = []
            current_max_value = 0
            cc_value = []
            
            interpolation_data = []
            
            length_optical_nerve = 0
            length_optical_nerve_gap = 0
            partial_distance = 0
            factor = 1

            # Process each slice along y axis
            for y in range(y_dim):
                
                # Take xz "slice", eliminating other ys
                selected_y_slice = nifti_data[:, y, :]
                
                # TODO: this is current_voxel_value
                max_voxel_value = int(np.round(np.max(selected_y_slice)))
                
                # Empty slice, go to the next
                if max_voxel_value == 0:
                    continue
                
                active_slice += 1
                previous_slice = active_slice - 1
                
                if active_slice == 0:
                    current_max_value = max_voxel_value
                    number_of_areas = 0
                    sum_cross_section_area = 0
                else:
                    if current_max_value != max_voxel_value:
                        current_max_value = max_voxel_value
                        
                        cc_value[previous_slice]['save_length'] = partial_distance
                        cc_value[previous_slice]['average_area'] = sum_cross_section_area / number_of_areas
                        
                        number_of_areas = 0
                        sum_cross_section_area = 0
                        partial_distance = 0
                
                # Deal with centering the mask in a new version of the image
                binarized_slice = selected_y_slice > 0  # Binarize
                
                # Find connected components and properties
                labeled_image = measure.label(binarized_slice)
                props = measure.regionprops(labeled_image)
                
                if len(props) == 0:
                    logger.info(f"WARNING: No region found in slice {y}")
                    continue
                            
                centroid = props[0].centroid
                orig_centroids = np.array(centroid)
                            
                # Shift the centroid of the region to the center of the image
                x_center_shift = int(np.round(image_center[0] - orig_centroids[0]))
                z_center_shift = int(np.round(image_center[1] - orig_centroids[1]))
                
                # Shift the nerve to image center
                cc = np.roll(np.roll(selected_y_slice, 
                                    x_center_shift, axis=0), 
                            z_center_shift, axis=1)
                            
                # Save the mask or segments
                if ismask == 0:
                    nifti_data[:, y, :] = cc
                else:
                    nifti_data[:, y, :] = np.sign(cc)
                
                # Calculate the distance between original centroids
                x_curr = orig_centroids[0]
                z_curr = orig_centroids[1]
                y_curr = y

                if active_slice == 0:
                    # First slice, no previous slice to compare
                    x_prev = x_curr
                    z_prev = z_curr
                    y_prev = y_curr + 1
                else:
                    # Calculate the distance between original centroids                                
                    x_prev = cc_value[previous_slice]['orig_centroid_x']
                    z_prev = cc_value[previous_slice]['orig_centroid_z']
                    y_prev = cc_value[previous_slice]['original_slice_yz']
                                
                x_center_displacement = np.round(x_curr - x_prev)
                z_center_displacement = np.round(z_curr - z_prev)
                y_center_displacement = np.round(y_curr - y_prev)
                                
                zz = z_resolution * z_center_displacement
                xx = x_resolution * x_center_displacement
                yy = y_resolution * y_center_displacement * factor
                
                distance = np.sqrt(xx*xx + yy*yy + zz*zz)
                residual_distance = distance - (y_resolution * factor)
                
                n_slices = round(distance / y_resolution)
                n_slices_upsampled = round(distance / y_resolution * 10)
            
                slice_gap = round(residual_distance / y_resolution)
                slice_gap_upsampled = round(residual_distance / y_resolution * 10)

                length_optical_nerve += distance
                length_optical_nerve_gap += (slice_gap_upsampled * y_resolution / 10 + y_resolution)    
                
                partial_distance += distance
                
                area = props[0].area * x_resolution * z_resolution
                number_of_areas += 1
                sum_cross_section_area += area
                
                
                # Initialize dictionary for this slice
                slice_data = dict()
                slice_data['subject'] = subject
                slice_data['side'] = side
                slice_data['current_slice_yz'] = active_slice
                slice_data['original_slice_yz'] = y
                
                try:
                    slice_data['segment_name'] = segment_types[int(max_voxel_value)]
                except Exception as e:
                    sentry_sdk.capture_exception(e)
                    logger.info(f"ERROR: {e}")
                    continue
                
                slice_data['max_voxel_value'] = max_voxel_value
                slice_data['circshift_x'] = x_center_shift
                slice_data['circshift_z'] = z_center_shift


                slice_data['orig_centroid_x'] = orig_centroids[0]
                slice_data['orig_centroid_z'] = orig_centroids[1]
                            
                slice_data['majaxis'] = props[0].major_axis_length * x_resolution
                slice_data['minaxis'] = props[0].minor_axis_length * z_resolution
                slice_data['area']    = area
                slice_data['eccent']  = props[0].eccentricity
                
                slice_data['average_area'] = 0
                slice_data['save_length'] = 0
                
                slice_data['distance'] = distance
                slice_data['slice_gap'] = slice_gap
                slice_data['slice_gap_upsampled'] = slice_gap_upsampled
                slice_data['length_on'] = length_optical_nerve
                slice_data['length_on_upsampled'] = length_optical_nerve_gap
                slice_data['int_distance'] = n_slices
                slice_data['int_upsampled_distance'] = n_slices_upsampled
                
                cc_value.append(slice_data)


                
            # Save processed info
            cc_value[active_slice]['save_length'] = partial_distance
            cc_value[active_slice]['average_area'] = sum_cross_section_area / number_of_areas
            
            sliceframe = pd.DataFrame(cc_value)
            sliceframe.to_excel(
                os.path.join(outImPath, subject, f"on{side}_slice_data.xlsx"), 
                index=False, header=True
            )
            dataframe.append(sliceframe)       
            
            # Build the linearized volume
            logger.info("INFO: Nerve Interpolation...")
            
            upsampled_slices = sliceframe['slice_gap_upsampled'].values + resolution_increase
            total_slices_upsampled = upsampled_slices.sum() + 10
            
            if total_slices_upsampled >= max_slices:
                logger.error(f"ERROR: Total upsampled slices exceed {max_slices} in subject {subject} side {side}!")
                logger.error("Please increase the number of max_slices in the code.")
                sentry_sdk.capture_message(
                    f"ERROR: Total upsampled slices exceed {max_slices} in subject {subject} side {side}!"
                )
                max_slices = total_slices_upsampled + 10
            
            # Interpolate data pythonic
            hres_linear_image = np.zeros((x_dim, max_slices, z_dim), dtype=np.float32)
            n_slice = len(sliceframe)
                    
            slice_counter = 0
            slice_counter_vector = []  
            for i, s in enumerate(range(n_slice)):
                
                # Get what slice to use
                y_curr = sliceframe['original_slice_yz'].values[i]
                if i == 0:
                    y_prev = 0
                else:
                    y_prev = sliceframe['original_slice_yz'].values[i-1]
                
                # Interpolate the slice
                
                # Get the number of slices to be inserted          
                minislices = upsampled_slices[i]
                
                tba_minislices = minislices - resolution_increase                    
                
                for k in range(tba_minislices):        
                                    
                    if k < tba_minislices:
                        if k < tba_minislices // 2:
                            hres_linear_image[:, slice_counter, :] = nifti_data[:, y_prev, :]
                        else:
                            hres_linear_image[:, slice_counter, :] = nifti_data[:, y_curr, :]

                    slice_counter += 1
                    
                interval = slice(slice_counter, slice_counter + resolution_increase)
                
                hres_linear_image[
                    :, slice_counter:slice_counter + resolution_increase, :
                ] = nifti_data[:, y_curr, :][:, np.newaxis, :]
                
                slice_counter += resolution_increase + 1
                slice_counter_vector.append(slice_counter)
                        
            logger.info("Image creation...")
            
            # Fill hole - if neighboring slices not zero, copy in slice after
            hole_list = []
            hole_counter = 0
            
            logger.info("Hole interpolation...")
            
            for slice_idx in range(1, max_slices-2):
                if (np.max(hres_linear_image[:, slice_idx+1, :]) == 0 and
                    np.max(hres_linear_image[:, slice_idx, :]) > 0 and
                    np.max(hres_linear_image[:, slice_idx+2, :]) > 0):
                    hole_counter += 1
                    hole_list.append(slice_idx)
                    hres_linear_image[:, slice_idx+1, :] = hres_linear_image[:, slice_idx+2, :]
                
            # Create NIfTI image for linearized data
            logger.info("Disk writing...")
            
            # Create header for the new image
            new_affine = nifti_img.affine.copy()
            # Adjust y-axis spacing
            new_affine[1, 1] = new_affine[1, 1] / resolution_increase
            new_affine[:3, :3] = new_affine[:3, :3] * np.eye(3)            
            
            # Create a new image with the linearized data
            lin_img = nib.Nifti1Image(hres_linear_image, new_affine, nifti_img.header)
            lin_img.header['pixdim'][2] = nifti_img.header['pixdim'][2] / resolution_increase  # Adjust y resolution
            
            # Save the linearized image
            nib.save(lin_img, os.path.join(outImPath, subject, f"on{side}{Lin4image}"))
            
            # Normalize the data
            slice_length = round(cc_value[-1]['length_on'] / (nifti_img.header['pixdim'][2] / 10))
            lengthfactor = max_slices / slice_length
            
            normalized_image = np.zeros_like(hres_linear_image)
            check_range = np.zeros(max_slices, dtype=int)
            
            logger.info("Normalization...")
            
            for ii in range(max_slices):
                # Figure out slice in aligned that needs to go 
                # into the ii-th slice of the normalized
                
                jj = round(ii / max_slices * slice_counter)  
                
                if jj < 1:
                    jj = 1
                    
                check_range[ii] = jj
                normalized_image[:, ii, :] = hres_linear_image[:, jj, :]
                
                # If a hole is found between slices, fill it with current
                if ii > 2:
                    if check_range[ii-3] > 0 and check_range[ii-2] == 0 and check_range[ii-1] > 0:
                        normalized_image[:, ii-2, :] = normalized_image[:, ii-3, :]
            
            # Create and save normalized image
            norm_img = nib.Nifti1Image(normalized_image, new_affine, lin_img.header)
            nib.save(norm_img, os.path.join(outImPath, subject, f"on{side}{Norm4image}"))
            
            
    logger.info("Processing complete!")

    dataframe = pd.concat(dataframe)
    dataframe.to_excel(DataFile, index=False)

                
if __name__ == "__main__":
    main()