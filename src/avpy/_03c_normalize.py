import os
import numpy as np
import nibabel as nib
from scipy import ndimage
import pandas as pd
import pickle
import subprocess
from skimage import measure
import sys
import psutil
import time
import gc
from datetime import datetime

NAME = "normalize_stats"

def main(path="./"):
    # Get current working directory
    #curwd = os.getcwd()
    
    segment_types = {
        16: "OT",
        8: "OC",
        4: "iCran",
        2: "iCan",
        1: "iOrb"
    }

    # Read study path from control file
    with open(os.path.join(path,'ONcontrol.txt'), 'r') as f:
        StudyPath = f.readline().strip()
        
    #StudyPath = "/home/robbis/git/aVP-toolbox/data/test/"

    inPath = os.path.join(StudyPath, 'data', 'proc')
    outImPath = os.path.join(StudyPath, 'data', 'proc')
    outResPath = os.path.join(StudyPath, 'results')

    ResampDataFile = os.path.join(outResPath, 'aVP_slice_data_iso.xlsx')
    ResampStretchFile = os.path.join(outResPath, 'aVP_section_CSA_length_iso.xlsx')

    sides = ['l', 'r']

    # Read subject list
    with open(os.path.join(StudyPath, 'data', 'sbj.list'), 'r') as fileID:
        subject_list = [line.strip() for line in fileID]

    image_types = ['linearize', 'normalized']
    segment_info = []
    dataframe = []

    for subject in subject_list:        
        for side_idx, side in enumerate(sides):
            for rr_idx, image_type in enumerate(image_types):
                # Keep track of the combined index
                bname = f"{subject}/on{side}_{image_type}_4bc_iso06"
                fname = os.path.join(inPath, bname)
                
                if not os.path.exists(f"{fname}.nii.gz"):
                    print(f"Could not find: {fname}.nii.gz")
                    continue
                    
                print(f"INFO: Processing resampled {subject} - {image_type} - {side}")
                    
                # Load the nifti file
                img = nib.load(f"{fname}.nii.gz")
                nifti_data = img.get_fdata()
                
                # Get dimensions and resolutions
                x_dim, dy, z_dim = nifti_data.shape
                x_resolution = img.header.get_zooms()[0]
                y_resolution = img.header.get_zooms()[1]
                z_resolution = img.header.get_zooms()[2]
                
                current_slice_idx = 0
                number_of_areas = 0
                sum_cross_section_area = 0
                total_length = 0
                partial_distance = 0
                cc_value = []
                
                # Process each slice
                for y in range(dy):
                    selected_y_slice = nifti_data[:, y, :]
                    max_voxel_value = int(np.round(np.max(selected_y_slice)))
                    
                    if max_voxel_value == 0:
                        continue
                                    
                    # Binarize the slice
                    binarized_slice = selected_y_slice > 0
                    
                    # Get region properties
                    labeled_image = measure.label(binarized_slice)
                    props = measure.regionprops(labeled_image)
                    
                    area = props[0].area * x_resolution * z_resolution
                    
                    number_of_areas += 1
                    sum_cross_section_area += area
                                        
                    if current_slice_idx > 1:
                        previous_voxel_value = cc_value[current_slice_idx-1]['max_voxel_value']
           
                        if max_voxel_value != previous_voxel_value:
                            
                            cc_value[current_slice_idx-1]['save_length'] = total_length
                            cc_value[current_slice_idx-1]['average_area'] = sum_cross_section_area / number_of_areas
                            cc_value[current_slice_idx-1]['segment_length'] = partial_distance
                            
                            segment_info.append(
                                {
                                    'subject': subject,
                                    'side': side,
                                    'image_type': image_type,
                                    'segment_type': previous_voxel_value,
                                    'segment_name': segment_types[previous_voxel_value],
                                    'lenght': total_length,
                                    'area': sum_cross_section_area / number_of_areas
                                }
                            )
                            
                            number_of_areas = 1
                            sum_cross_section_area = slice_data['area']
                            partial_distance = 0
                        else:
                            sum_cross_section_area += slice_data['area']
                            number_of_areas += 1
                            
                            
                    total_length += y_resolution
                    partial_distance += y_resolution
                    
                    current_slice_idx += 1
                    
                    slice_data = {
                        'subject': subject,
                        'side': side,
                        'image_type': image_type,
                        'current_slice_yz': current_slice_idx,
                        'original_slice_yz': y,
                        'max_voxel_value': max_voxel_value,
                        'segment_name': segment_types[max_voxel_value],
                        'distance': y_resolution,
                        'majaxis': props[0].major_axis_length * x_resolution,
                        'minaxis': props[0].minor_axis_length * z_resolution,
                        'area': props[0].area * x_resolution * z_resolution,
                        'eccent': props[0].eccentricity,
                        'total_length': total_length,
                        'save_length': 0,
                        'average_area': 0
                    }
                    
                        
                    cc_value.append(slice_data)
            
                # Process the last slice if we had data
                if current_slice_idx == 0:
                    print(f'No ON elements found for {bname} - skipping')
                    continue
                    
                cc_value[current_slice_idx-1]['save_length'] = cc_value[current_slice_idx-1]['total_length']
                cc_value[current_slice_idx-1]['average_area'] = sum_cross_section_area / number_of_areas

                sliceframe = pd.DataFrame(cc_value)
                dataframe.append(sliceframe)

                # Save the last segment info
                segment_info.append(
                    {
                        'subject': subject,
                        'side': side,
                        'image_type': image_type,
                        'segment_type': max_voxel_value,
                        'segment_name': segment_types[max_voxel_value],
                        'lenght': total_length,
                        'area': sum_cross_section_area / number_of_areas
                    }
                )
    # Build dataframe
    dataframe = pd.concat(dataframe)
    dataframe.to_excel(
        ResampDataFile, index=False, header=True
    )
    # Add informations for single segments
    segment_info = pd.DataFrame(segment_info)
    segment_info.to_excel(
        ResampStretchFile, index=False, header=True
    )
    
if __name__ == "__main__":
    main()
