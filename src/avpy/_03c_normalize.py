import os
import numpy as np
import nibabel as nib
import pandas as pd
from skimage import measure

import logging
logger = logging.getLogger(__name__)

NAME = "normalize_stats"

def main(path="./", debug=False):
    # Get current working directory
    #curwd = os.getcwd()
    
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    
    segment_types = {
        16: "OT",
        8: "OC",
        4: "iCran",
        2: "iCan",
        1: "iOrb"
    }

    # Read study path from control file
    StudyPath = path
        
    #StudyPath = "/home/robbis/git/aVP-toolbox/data/test/"

    inPath = os.path.join(StudyPath, 'data', 'proc')
    outImPath = os.path.join(StudyPath, 'data', 'proc')
    outResPath = os.path.join(StudyPath, 'results')

    ResampDataFile = os.path.join(outResPath, 'CSA_slice_iso.xlsx')
    ResampStretchFile = os.path.join(outResPath, 'CSA_section_iso06.xlsx')

    sides = ['l', 'r']

    # Read subject list
    with open(os.path.join(StudyPath, 'data', 'sbj.list'), 'r') as fileID:
        subject_list = [line.strip() for line in fileID]

    image_types = ['linearize', 'normalized']
    segment_info = []
    dataframe = []
    summary_stats = []

    for subject in subject_list:        
        for side_idx, side in enumerate(sides):
            for rr_idx, image_type in enumerate(image_types):
                
                # Keep track of the combined index
                bname = f"{subject}/on{side}_{image_type}_4bc_iso06"
                fname = os.path.join(inPath, bname)
                
                if not os.path.exists(f"{fname}.nii.gz"):
                    logger.error(f"Could not find: {fname}.nii.gz")
                    continue
                    
                logger.info(f"Processing resampled {subject} - {image_type} - {side}")
                    
                # Load the nifti file
                img = nib.load(f"{fname}.nii.gz")
                nifti_data = img.get_fdata()
                
                # Get dimensions and resolutions
                x_dim, dy, z_dim = nifti_data.shape
                x_resolution = img.header.get_zooms()[0]
                y_resolution = img.header.get_zooms()[1]
                z_resolution = img.header.get_zooms()[2]

                current_slice_idx = -1

                n_segment_area = 0
                n_total_area = 0
                total_area = 0
                segment_area = 0
                
                total_length = 0
                segment_length = 0
                
                cc_value = []
                
                nonzero_elements = np.nonzero(nifti_data)
                y_min = np.min(nonzero_elements[1])
                y_max = np.max(nonzero_elements[1])
                
                # Process each slice
                for y in range(y_min, y_max + 1):
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
                    
                    n_segment_area += 1
                    n_total_area += 1
                    total_area += area
                    segment_area += area
                                        
                    if current_slice_idx > 1:
                        
                        previous_voxel_value = cc_value[current_slice_idx-1]['max_voxel_value']
                        if max_voxel_value != previous_voxel_value:
                            
                            cc_value[current_slice_idx-1]['save_length'] = total_length
                            cc_value[current_slice_idx-1]['average_area'] = segment_area / n_segment_area
                            cc_value[current_slice_idx-1]['segment_length'] = segment_length
                            
                            segment_info.append(
                                {
                                    'subject': subject,
                                    'side': side,
                                    'image_type': image_type,
                                    'segment_type': previous_voxel_value,
                                    'segment_name': segment_types[previous_voxel_value],
                                    'segment_length': segment_length,
                                    'area': segment_area / n_segment_area
                                }
                            )
                            
                            n_segment_area = 1
                            segment_area = slice_data['area']
                            segment_length = 0
                        else:
                            segment_area += slice_data['area']
                            n_segment_area += 1
                            
                            
                    total_length += y_resolution
                    segment_length += y_resolution
                    
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
                        'average_area': 0,
                        'length_on': 0,
                    }
                    
                        
                    cc_value.append(slice_data)
            
                # Process the last slice if we had data
                if current_slice_idx == 0:
                    logger.info(f'No ON elements found for {bname} - skipping')
                    continue
                    
                cc_value[current_slice_idx-1]['segment_length'] = segment_length
                cc_value[current_slice_idx-1]['average_area'] = segment_area / n_segment_area
                cc_value[current_slice_idx-1]['length_on'] = total_length
                
                summary = {
                    'subject': subject,
                    'side': side,
                    'image_type': image_type,
                    'length_on': total_length,
                    'area': total_area / n_total_area,
                }
                
                summary_stats.append(summary)
                
                
                sliceframe = pd.DataFrame(cc_value)
                dataframe.append(sliceframe)

                # Save the last segment info
                voxel_value = cc_value[current_slice_idx-1]['max_voxel_value']
                segment_info.append(
                    {
                        'subject': subject,
                        'side': side,
                        'image_type': image_type,
                        'segment_type': voxel_value,
                        'segment_name': segment_types[voxel_value],
                        'length_on': total_length,
                        'segment_length': segment_length,
                        'area': segment_area / n_segment_area
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
    
    # Save summary statistics
    summary_stats_df = pd.DataFrame(summary_stats)
    summary_stats_df.to_excel(
        os.path.join(outResPath, 'CSA_total_iso06.xlsx'),
        index=False, header=True)
    
if __name__ == "__main__":
    main()
