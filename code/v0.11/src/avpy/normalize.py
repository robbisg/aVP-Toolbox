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
from scipy import ndimage
import pandas as pd
import pickle
import subprocess
from skimage import measure
import sys

# Get current working directory
curwd = os.getcwd()

# Read study path from control file
#with open(os.path.join(curwd, 'ONcontrol.txt'), 'r') as fileID:
#    StudyPath = fileID.readline().strip()
    
StudyPath = "/home/robbis/git/aVP-toolbox/data/test/"

inPath = os.path.join(StudyPath, 'data', 'proc')
outImPath = os.path.join(StudyPath, 'data', 'proc')
outResPath = os.path.join(StudyPath, 'results')

# Output file paths
DataFile = os.path.join(outResPath, 'py_aVP_slice_data.xlsx')
HoleFile = os.path.join(outResPath, 'py_log_check_hole.xlsx')
RangeFile = os.path.join(outResPath, 'py_log_check_range.xlsx')
LenStretchFile = os.path.join(outResPath, 'py_aVP_section_CSA_length.xlsx')

# Define column labels
tablabels = [
    'curr_sli_yz', 'orig_sli_yz', 'point_y', 'point_z', 'circshift_y', 'circshift_z',
    'dist', 'int_dist_x10', 'len_on', 'tot_len', 'mMax', 'save_len',
    'CSArea', 'Eccent', 'MajAxis', 'MinAxis', 'AvgCSA'
]

stretxt = [
    'Subject', 'ONsection', 'side', 'TotLength', 'OT_length', 'OC_length',
    'iCran_length', 'iCan_length', 'iOrb_length', 'OT_CSA', 'OC_CSA',
    'iCran_CSA', 'iCan_CSA', 'iOrb_CSA', 'SegmCode 1', 'SegmCode 2',
    'SegmCode 3', 'SegmCode 4', 'SegmCode 5'
]

rangetxt = ['Subject', 'ONsection', 'side', 'Slice...']
#pd.DataFrame([rangetxt]).to_excel(RangeFile, index=False, header=False)

holetxt = ['Subject', 'ONsection', 'side', 'Slice...']
#pd.DataFrame([holetxt]).to_excel(HoleFile, index=False, header=False)

# Define image output file names
Lin4image = '_py_linearize_4bc.nii.gz'
Lin4pkl = '_py_linearize_4bc.pkl'

Norm4image = '_py_normalized_4bc.nii.gz'
Norm4pkl = '_py_normalized_4bc.pkl'

fullNorm4image = '_py_full_normalized_4bc.nii.gz'

sides = ['l', 'r']
isbj = 0
tablelength = []

maxNslices = 2500
# 0: use combination of aVP segments. 1: use individual aVP segments (not tested)
ismask = 0

# Read subject list
with open(os.path.join(StudyPath, 'data', 'sbj.list'), 'r') as fileID:
    subject_list = [line.strip() for line in fileID]

resolution_increase = 10

loopRef = []
tablelength_data = []

dataframe_slice = []

# Main processing loop
# TODO: Consider to use multiprocessing for parallel processing
# TODO: Consider to use a progress bar for better user experience
# TODO: Consider to create a subject function to encapsulate the logic
for subject in subject_list:
    isbj += 1
        
    for side_idx, side in enumerate(sides):
        fname = os.path.join(inPath, subject, f'on_{side}')
        
        print(f"INFO: Analyzing {subject} - on_{side}")
        
        if not os.path.exists(f"{fname}.nii.gz"):
            print(f"could not find: {fname}.nii.gz")
            sys.exit(1)
        
        # Load the nifti file
        nifti_img = nib.load(f"{fname}.nii.gz")
        nifti_data = nifti_img.get_fdata()
        
        # Check if the image is empty
        if np.sum(nifti_data) == 0:
            print(f"WARNING: Image {fname}.nii.gz is empty.")
            continue
        
        # Take into account that the image considers the origin to be lower left, but numpy is upper left
        nifti_data = np.flip(nifti_data, axis=0)
        
        # Get dimensions and resolutions
        x_dim, y_dim, z_dim = nifti_data.shape
        x_resolution = nifti_img.header.get_zooms()[0]
        y_resolution = nifti_img.header.get_zooms()[1]
        z_resolution = nifti_img.header.get_zooms()[2]
        
        # Calculate the center of the slice to be able to shift the centroid of the ROI there
        image_center = [round(z_dim/2), round(x_dim/2)]
        active_slice = -1
        segment_type = 0
        
        table = []
        current_max_value = 0
        cc_value = []
        
        interpolation_data = []    

        # Process each slice along y axis
        for y in range(y_dim):
            
            # Take xz "slice", eliminating other ys
            selected_y_slice = nifti_data[:, y, :]  
            max_voxel_value = np.max(selected_y_slice)
            
            # Empty slice, go to the next
            if max_voxel_value == 0:
                continue
            
            if active_slice == -1:
                segment_type = 1
                current_max_value = max_voxel_value
            else:
                if not current_max_value == max_voxel_value:
                    segment_type += 1
            
            active_slice += 1
            
            # Initialize dictionary for this slice
            slice_data = {
                'current_slice_yz': active_slice,
                'original_slice_yz': y,
                'segment_type': segment_type,
                'max_voxel_value': max_voxel_value,
                'subject': subject,
                'side': side
            }
            
            # Deal with centering the mask in a new version of the image
            binarized_slice = selected_y_slice > 0  # Binarize
            
            # Find connected components and properties
            # TODO: Check
            labeled_image = measure.label(binarized_slice)
            props = measure.regionprops(labeled_image)
            
            if len(props) == 0:
                print(f"WARNING: No region found in slice {y}")
                continue
                        
            centroid = props[0].centroid
            orig_centroids = np.array(centroid)
            
            slice_data['majaxis'] = props[0].major_axis_length * x_resolution
            slice_data['minaxis'] = props[0].minor_axis_length * z_resolution
            slice_data['area'] =    props[0].area * x_resolution * z_resolution
            slice_data['eccent'] =  props[0].eccentricity
            
            # Shift the centroid of the region to the center of the image
            x_center_shift = int(image_center[1] - round(orig_centroids[1]))
            z_center_shift = int(image_center[0] - round(orig_centroids[0]))
            
            cc = np.roll(np.roll(selected_y_slice, 
                                 x_center_shift, axis=0), 
                         z_center_shift, axis=1)
            
            # TODO: Use two dictionary entries
            #slice_data['circshift'] = [x_center_shift, z_center_shift]
            slice_data['circshift_x'] = x_center_shift
            slice_data['circshift_z'] = z_center_shift
            
            # Save the mask or segments
            if ismask == 0:
                nifti_data[:, y, :] = cc
            else:
                nifti_data[:, y, :] = np.sign(cc)
            
            # Account for obliqueness of the optic nerve
            length_on = 0
            
            # TODO: Separate points in two fields and use different naming
            slice_data['orig_centroid_x'] = orig_centroids[1]
            slice_data['orig_centroid_z'] = orig_centroids[0]
            slice_data['distance'] = 0
            slice_data['length_on'] = length_on
            slice_data['length_on'] = y_resolution
            slice_data['total_length'] = slice_data['length_on']
            slice_data['save_length'] = 0
            slice_data['int_distance_x10'] = 0
            slice_data['avgCSA'] = 0
            
            # TODO: Do we need to store it in the slice_data?
            
            
            
            # Replicate the slice 10 times
            # TODO: Change variable name `vol`
            vol = np.zeros((x_dim, resolution_increase, z_dim))
            for kk in range(resolution_increase):
                vol[:, kk, :] = nifti_data[:, y, :]
            slice_data['vol'] = vol

            
            # Create an empty slice and start values for lengths
            slice_data['intra'] = np.zeros((x_dim, 1, z_dim))
            slice_data['slice'] = nifti_data[:, y, :]
            
            slice_vols = dict()
            slice_vols['vol'] = vol
            slice_vols['intra'] = np.zeros((x_dim, 1, z_dim))
            slice_vols['slice'] = nifti_data[:, y, :]
            
                
            if active_slice == 0:
                countCSA = 1
                sumCSA = slice_data['area']
            
            elif active_slice > 0:
                # Calculate the distance between original centroids
                previous_slice = active_slice - 1
                
                z_center_displacement = (orig_centroids[1] - 
                                        cc_value[previous_slice]['orig_centroid_z'])
                x_center_displacement = (orig_centroids[0] - 
                                        cc_value[previous_slice]['orig_centroid_x'])
                
                zz = z_resolution * z_center_displacement
                xx = x_resolution * x_center_displacement
                yy = y_resolution * 2  # Why times 2?
                
                slice_data['distance'] = np.sqrt(xx*xx + yy*yy + zz*zz)
                
                # Multiply by 10 and round to integer
                # TODO: Maybe it is bettere `slice_gap`
                distance_gap = round((slice_data['distance'] - 2*y_resolution) / 
                                    y_resolution * 10)
                
                # If negative something is strange - ignore it
                if distance_gap < 0:
                    current_slice_print = y
                    previous_slice_print = cc_value[active_slice]['original_slice_yz']
                    print(f"WARNING: Negative distance between slice {current_slice_print} {previous_slice_print}")
                    distance_gap = 0
                
                # If positive make insertions
                if distance_gap > 0:
                    slice_vols['intra'] = np.zeros((x_dim, distance_gap, z_dim))
                    
                    # For first half of interpolates use prior slice
                    # For the second half, use the current slice
                    for kk in range(distance_gap):
                        if kk < distance_gap // 2:
                            slice_vols['intra'][:, kk, :] = interpolation_data[previous_slice]['vol'][:, 0, :]
                        else:
                            slice_vols['intra'][:, kk, :] = slice_vols['vol'][:, 0, :]
                
                # Update length of ON
                slice_data['length_on'] += distance_gap / resolution_increase * y_resolution
                
                # And total length
                slice_data['total_length'] = cc_value[previous_slice]['total_length'] + slice_data['length_on']
                slice_data['int_distance_x10'] = distance_gap
                
                # Cross-sectional Area (CSA)
                if slice_data['max_voxel_value'] != cc_value[previous_slice]['max_voxel_value']:
                    # TODO: Is save_lenght equal to total_length?
                    cc_value[previous_slice]['save_length'] = cc_value[previous_slice]['total_length']
                    cc_value[previous_slice]['avgCSA'] = sumCSA / countCSA
                    countCSA = 1
                    sumCSA = slice_data['area']
                else:
                    sumCSA += slice_data['area']
                    countCSA += 1
                
                # Add data to table
                # TODO: Check previous slice
                table_row = [
                    slice_data['current_slice_yz'],
                    slice_data['original_slice_yz'],
                    slice_data['orig_centroid_x'],
                    slice_data['orig_centroid_z'],
                    slice_data['circshift_x'],
                    slice_data['circshift_z'],
                    slice_data['distance'],
                    slice_data['int_distance_x10'],
                    slice_data['length_on'],
                    slice_data['total_length'],
                    slice_data['max_voxel_value'],
                    slice_data['save_length'],
                    slice_data['area'],
                    slice_data['eccent'],
                    slice_data['majaxis'],
                    slice_data['minaxis'],
                    slice_data['avgCSA']
                ]
                table.append(table_row)
            
            
            cc_value.append(slice_data)
            interpolation_data.append(slice_vols)
        
        # Save processed info
        cc_value[active_slice]['save_length'] = cc_value[active_slice-1]['total_length']
        cc_value[active_slice]['avgCSA'] = sumCSA / countCSA
        
        # Add final row to table
        table_row = [
            cc_value[active_slice-1]['current_slice_yz'],
            cc_value[active_slice-1]['original_slice_yz'],
            cc_value[active_slice-1]['point'][0],
            cc_value[active_slice-1]['point'][1],
            cc_value[active_slice-1]['circshift'][0],
            cc_value[active_slice-1]['circshift'][1],
            cc_value[active_slice-1]['distance'],
            cc_value[active_slice-1]['int_distance_x10'],
            cc_value[active_slice-1]['length_on'],
            cc_value[active_slice-1]['total_length'],
            cc_value[active_slice-1]['max_voxel_value'],
            cc_value[active_slice-1]['save_length'],
            cc_value[active_slice-1]['area'],
            cc_value[active_slice-1]['eccent'],
            cc_value[active_slice-1]['majaxis'],
            cc_value[active_slice-1]['minaxis'],
            cc_value[active_slice-1]['avgCSA']
        ]
        table.append(table_row)
        
        # Extract relevant data for output
        length_values = [row[11] for row in table]
        length_values = [l for l in length_values if l != 0]
        
        carea_values = [row[16] for row in table]
        carea_values = [c for c in carea_values if c != 0]
        
        mNms_values = [row[10] for row in table]
        mNms_values = [m for m, c in zip(mNms_values, carea_values) if c != 0]
        
        subsidInd = (isbj - 1) * 2 + side_idx
        
        loopRef.append([subject, 'on', sides[side_idx]])
        
        # Prepare table length data
        tablelength_row = [
        #    cc_value[active_slice-1]['total_length'],
        #    length_values[0],
        #    length_values[1] - length_values[0],
        #    length_values[2] - length_values[1],
        #    length_values[3] - length_values[2],
        #    cc_value[active_slice-1]['total_length'] - length_values[3],
        #    carea_values[0],
        #    carea_values[1],
        #    carea_values[2],
        #    carea_values[3],
        #    carea_values[4],
        #    mNms_values[0],
        #    mNms_values[1],
        #    mNms_values[2],
        #    mNms_values[3],
        #    mNms_values[4]
        ]
        tablelength_data.append(tablelength_row)
        
        # Write data to excel files
        #for i, row in enumerate(table):
        #    row_idx = (subsidInd - 1) * len(table) + i + 1
        #    df = pd.DataFrame([loopRef[subsidInd-1] + [tablabels[j]] for j in range(len(tablabels))])
        #    with pd.ExcelWriter(DataFile, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        #        df.to_excel(writer, sheet_name='Sheet 1', startrow=row_idx-1, startcol=0, header=False, index=False)
        #        
        #        # Write table data
        #        df2 = pd.DataFrame([row]).transpose()
        #        df2.to_excel(writer, sheet_name='Sheet 1', startrow=row_idx-1, startcol=4, header=False, index=False)
        
        # Build the linearized volume
        print("INFO: Nerve Interpolation...")
        
        # Select the first upsampled slice
        bbb = cc_value[0]['vol']
        interpolation_counter = 10  # Minislice counter
        
        n_slice = len(cc_value)  # Count the nonzero original slices
        
        # Calculate the volumes for the remaining slices
        for ii in range(1, n_slice):
            # Get the length in terms of minislices needed for this slice
            # TODO: Do we have distance_gap?
            n_dist = cc_value[ii]['intra'].shape[1]
            
            # Insert the extra slices as needed
            new_interval = range(interpolation_counter+1, 
                                    interpolation_counter+n_dist+1)
            
            for idx, j in enumerate(new_interval):
                if j < bbb.shape[1]:
                    bbb[:, j, :] = cc_value[ii]['intra'][:, idx, :]
            
            # Advance the minislice counter
            interpolation_counter += n_dist
            
            # Make a range for the 10 minislices for this slice
            interval = range(interpolation_counter+1, interpolation_counter+11)
            
            # Insert the 10 minislices for this slice
            for idx, j in enumerate(interval):
                if j < bbb.shape[1]:
                    bbb[:, j, :] = cc_value[ii]['vol'][:, idx, :]
            
            # Advance the minislice counter
            interpolation_counter += 10
        
        original_linearized_slices = interpolation_counter
        
        print("INFO: Image creation...")
        # Create or pad the full array
        if interpolation_counter < maxNslices:
            # If we need to pad, create a new larger array
            full_bbb = np.zeros((x_dim, maxNslices, z_dim))
            full_bbb[:, :interpolation_counter, :] = bbb[:, :interpolation_counter, :]
            bbb = full_bbb
        else:
            print(f"Houston, we have a problem! More than {maxNslices} slices!!!")
            sys.exit(1)
        
        # Fill hole - if neighboring slices not zero, copy in slice after
        hole_list = []
        hole_counter = 0
        
        print("INFO: Hole filling...")
        
        for slice_idx in range(1, maxNslices-2):
            if (np.max(bbb[:, slice_idx+1, :]) == 0 and
                np.max(bbb[:, slice_idx, :]) > 0 and
                np.max(bbb[:, slice_idx+2, :]) > 0):
                hole_counter += 1
                hole_list.append(slice_idx)
                bbb[:, slice_idx+1, :] = bbb[:, slice_idx+2, :]
        
        # Write hole list to excel
        if hole_list:
            df = pd.DataFrame([loopRef[subsidInd-1] + hole_list])
            with pd.ExcelWriter(HoleFile, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                df.to_excel(writer, sheet_name='Sheet 1', startrow=subsidInd, startcol=0, header=False, index=False)
        
        # Create NIfTI image for linearized data
        print("INFO: Disk writing...")
        
        # Create header for the new image
        new_affine = nifti_img.affine.copy()
        # Adjust y-axis spacing
        new_affine[1, 1] = new_affine[1, 1] / 10
        
        # Create a new image with the linearized data
        lin_img = nib.Nifti1Image(bbb, new_affine, nifti_img.header)
        lin_img.header['pixdim'][2] = nifti_img.header['pixdim'][2] / 10  # Adjust y resolution
        
        # Save the linearized image
        nib.save(lin_img, os.path.join(outImPath, subject, f"on{side}{Lin4image}"))
        
        # Save the cc_value data
        with open(os.path.join(outImPath, subject, f"on_{side}{Lin4pkl}"), 'wb') as f:
            pickle.dump(cc_value, f)
        
        # Normalize the data
        lengthfactor = maxNslices / round(cc_value[active_slice-1]['total_length'] / 
                                            (nifti_img.header['pixdim'][2] / 10))
        
        zz = np.zeros_like(bbb)
        check_range = np.zeros(maxNslices, dtype=int)
        
        print("INFO: Normalization...")
        
        for ii in range(maxNslices):
            jj = round(ii / maxNslices * original_linearized_slices)  # Figure out slice in aligned that needs to go into the ii-th slice of the normalized
            if jj < 1:
                jj = 1
            check_range[ii] = jj
            zz[:, ii, :] = bbb[:, jj, :]
            
            # If a hole is found between slices, fill it with current
            if ii > 2:
                if check_range[ii-3] > 0 and check_range[ii-2] == 0 and check_range[ii-1] > 0:
                    zz[:, ii-2, :] = zz[:, ii-3, :]
        
        # Create and save normalized image
        norm_img = nib.Nifti1Image(zz, new_affine, lin_img.header)
        nib.save(norm_img, os.path.join(outImPath, subject, f"on{side}{Norm4image}"))
        
        # Save normalized cc_value data
        with open(os.path.join(outImPath, subject, f"on_{side}{Norm4pkl}"), 'wb') as f:
            pickle.dump(cc_value, f)
        
        # Write range data to excel
        df = pd.DataFrame([loopRef[subsidInd-1] + list(check_range)])
        with pd.ExcelWriter(RangeFile, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, sheet_name='Sheet 1', startrow=subsidInd+1, startcol=0, header=False, index=False)

# Write section data to excel
tablelength_df = pd.DataFrame(tablelength_data)
loopRef_df = pd.DataFrame(loopRef)

stretxt_df = pd.DataFrame([stretxt])
with pd.ExcelWriter(LenStretchFile, engine='openpyxl') as writer:
    stretxt_df.to_excel(writer, sheet_name='Sheet 1', startrow=0, startcol=0, header=False, index=False)
    loopRef_df.to_excel(writer, sheet_name='Sheet 1', startrow=1, startcol=0, header=False, index=False)
    tablelength_df.to_excel(writer, sheet_name='Sheet 1', startrow=1, startcol=3, header=False, index=False)
    
# Now write the resampled data
resamptablelength_df = pd.DataFrame(resamptablelength_data)
resamploopRef_df = pd.DataFrame(resamploopRef)

with pd.ExcelWriter(ResampStretchFile, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
    resamploopRef_df.to_excel(writer, sheet_name='Sheet 1', startrow=1, startcol=0, header=False, index=False)
    resamptablelength_df.to_excel(writer, sheet_name='Sheet 1', startrow=1, startcol=4, header=False, index=False)

print("INFO: Processing complete!")


# Ensure isotropicness...
# TODO: Separate in another script
print("INFO: Running resampling script...")
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
resample_script = os.path.join(curwd, 'aVP_resample.sh')
subprocess.run([resample_script], shell=True, check=True)

# Process the resampled data
# TODO: Separate in other script
ResampDataFile = os.path.join(outResPath, 'aVP_slice_data_iso.xlsx')
ResampStretchFile = os.path.join(outResPath, 'aVP_section_CSA_length_iso.xlsx')

resamptablelength = []
isbj = 0
resamp = ['linearize', 'normalized']

resamplabels = [
    'curr_sli_yz', 'orig_sli_yz', 'mMax', 'dist', 'tot_len', 
    'save_len', 'CSArea', 'Eccent', 'MajAxis', 'MinAxis', 'AvgCSA'
]

resampstretxt = [
    'Subject', 'Image', 'ONsection', 'side', 'TotLength', 
    'OT_length', 'OC_length', 'iCran_length', 'iCan_length', 'iOrb_length', 
    'OT_CSA', 'OC_CSA', 'iCran_CSA', 'iCan_CSA', 'iOrb_CSA', 
    'SegmCode 1', 'SegmCode 2', 'SegmCode 3', 'SegmCode 4', 'SegmCode 5'
]

# Create initial spreadsheet
pd.DataFrame([resampstretxt]).to_excel(ResampStretchFile, index=False, header=False)

resamploopRef = []
resamptablelength_data = []

for subject in subject_list:
    isbj += 1
    
    for side_idx, side in enumerate(sides):
        for rr_idx, rr in enumerate(resamp):
            # Keep track of the combined index
            subsidInd = (isbj-1)*4 + rr_idx*2 + side_idx + 1
            bname = f"{subject}/on{side}_{rr}_4bc_iso06"
            fname = os.path.join(inPath, bname)
            
            if not os.path.exists(f"{fname}.nii.gz"):
                print(f"Could not find: {fname}.nii.gz")
                continue
                
            print(f"INFO: Processing resampled {subject} - {rr} - {side}")
                
            # Load the nifti file
            aa = nib.load(f"{fname}.nii.gz")
            nifti_data = aa.get_fdata()
            
            # Get dimensions and resolutions
            x_dim, dy, z_dim = nifti_data.shape
            x_resolution = aa.header.get_zooms()[0]
            y_resolution = aa.header.get_zooms()[1]
            z_resolution = aa.header.get_zooms()[2]
            
            rtable = []
            incount = 0
            countCSA = 0
            sumCSA = 0
            cc_value = []
            
            # Process each slice
            for y in range(dy):
                selected_y_slice = nifti_data[:, y, :]
                max_voxel_value = np.max(selected_y_slice)
                
                if max_voxel_value > 0:
                    incount += 1
                    
                    # Binarize the slice
                    binarized_slice = selected_y_slice > 0
                    
                    # Get region properties
                    labeled_image = measure.label(binarized_slice)
                    props = measure.regionprops(labeled_image)
                    
                    if len(props) > 0:
                        slice_data = {
                            'current_slice_yz': incount,
                            'original_slice_yz': y,
                            'max_voxel_value': max_voxel_value,
                            'distance': y_resolution,
                            'majaxis': props[0].major_axis_length * x_resolution,
                            'minaxis': props[0].minor_axis_length * z_resolution,
                            'area': props[0].area * x_resolution * z_resolution,
                            'eccent': props[0].eccentricity,
                            'save_length': 0,
                            'avgCSA': 0
                        }
                        
                        countCSA += 1
                        sumCSA += slice_data['area']
                        
                        if incount > 1:
                            slice_data['total_length'] = cc_value[incount-2]['total_length'] + slice_data['distance']
                            
                            if not slice_data['max_voxel_value'] == cc_value[incount-2]['max_voxel_value']:
                                cc_value[incount-2]['save_length'] = cc_value[incount-2]['total_length']
                                cc_value[incount-2]['avgCSA'] = sumCSA / countCSA
                                countCSA = 1
                                sumCSA = slice_data['area']
                            else:
                                sumCSA += slice_data['area']
                                countCSA += 1
                            
                            rtable_row = [
                                cc_value[incount-2]['current_slice_yz'],
                                cc_value[incount-2]['original_slice_yz'],
                                cc_value[incount-2]['max_voxel_value'],
                                cc_value[incount-2]['distance'],
                                cc_value[incount-2]['total_length'],
                                cc_value[incount-2]['save_length'],
                                cc_value[incount-2]['area'],
                                cc_value[incount-2]['eccent'],
                                cc_value[incount-2]['majaxis'],
                                cc_value[incount-2]['minaxis'],
                                cc_value[incount-2]['avgCSA']
                            ]
                            rtable.append(rtable_row)
                        else:
                            slice_data['total_length'] = slice_data['distance']
                        
                        cc_value.append(slice_data)
            
            # Process the last slice if we had data
            if incount == 0:
                print(f'No ON elements found for {bname} - skipping')
                continue
                
            cc_value[incount-1]['save_length'] = cc_value[incount-1]['total_length']
            cc_value[incount-1]['avgCSA'] = sumCSA / countCSA
            
            # Add the final row to the table
            rtable_row = [
                cc_value[incount-1]['current_slice_yz'],
                cc_value[incount-1]['original_slice_yz'],
                cc_value[incount-1]['max_voxel_value'],
                cc_value[incount-1]['distance'],
                cc_value[incount-1]['total_length'],
                cc_value[incount-1]['save_length'],
                cc_value[incount-1]['area'],
                cc_value[incount-1]['eccent'],
                cc_value[incount-1]['majaxis'],
                cc_value[incount-1]['minaxis'],
                cc_value[incount-1]['avgCSA']
            ]
            rtable.append(rtable_row)
            
            # Extract relevant data for output
            length_values = [row[5] for row in rtable]
            length_values = [l for l in length_values if l != 0]
            
            carea_values = [row[10] for row in rtable]
            carea_values = [c for c in carea_values if c != 0]
            
            mNms_values = [row[2] for row in rtable]
            mNms_values = [m for m, c in zip(mNms_values, carea_values) if c != 0]
            
            # Record the subject info
            resamploopRef.append([subject, resamp[rr_idx], 'on', sides[side_idx]])
            
            # Prepare resampled table length data
            resamptablelength_row = [
                cc_value[incount-1]['total_length'],
                length_values[0],
                length_values[1] - length_values[0],
                length_values[2] - length_values[1],
                length_values[3] - length_values[2],
                cc_value[incount-1]['total_length'] - length_values[3],
                carea_values[0],
                carea_values[1], 
                carea_values[2],
                carea_values[3],
                carea_values[4],
                mNms_values[0],
                mNms_values[1],
                mNms_values[2],
                mNms_values[3],
                mNms_values[4]
            ]
            resamptablelength_data.append(resamptablelength_row)
            
            # Write data to Excel
            for typ in range(len(resamplabels)):
                resampnoopRef = resamploopRef[subsidInd-1] + [resamplabels[typ]]
                df = pd.DataFrame([resampnoopRef])
                row_idx = (subsidInd-1) * len(rtable) + typ + 1
                
                with pd.ExcelWriter(ResampDataFile, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                    df.to_excel(writer, sheet_name='Sheet 1', startrow=row_idx-1, startcol=0, header=False, index=False)
            
            # Write the data values
            rtable_df = pd.DataFrame(rtable)
            with pd.ExcelWriter(ResampDataFile, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                rtable_df.transpose().to_excel(writer, sheet_name='Sheet 1', 
                                              startrow=(subsidInd-1)*len(rtable), 
                                              startcol=5, header=False, index=False)