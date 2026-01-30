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
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial.distance import euclidean
import pandas as pd
import pickle
import subprocess
from skimage import measure
import sys
import psutil
import time
import gc
from datetime import datetime
sys.path.append("/home/robbis/git/aVP-toolbox/code/v0.11/src/")

from avpy.test.test_cylinder import main

# Memory profiling functions
def get_memory_usage():
    """Return the current memory usage of this process in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert to MB

def log_memory_usage(label):
    """Log the current memory usage with a label."""
    mem_usage = get_memory_usage()
    timestamp = datetime.now().strftime("%H:%M:%S")
    #print(f"[{timestamp}] MEMORY ({label}): {mem_usage:.2f} MB")

def memory_report(func):
    """Decorator to report memory usage before and after function execution."""
    def wrapper(*args, **kwargs):
        gc.collect()  # Force garbage collection before measurement
        label_before = f"Before {func.__name__}"
        log_memory_usage(label_before)
        
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        gc.collect()  # Force garbage collection before measurement
        label_after = f"After {func.__name__} (took {elapsed:.2f}s)"
        log_memory_usage(label_after)
        
        return result
    return wrapper

def log_variable_memory_usage(variables, label):
    """Log memory usage of specified variables."""
    #print(f"\n[{datetime.now().strftime('%H:%M:%S')}] MEMORY PROFILE ({label}):")
    for var_name, var_value in variables.items():
        size_mb = sys.getsizeof(var_value) / (1024 * 1024)  # Convert to MB
        print(f"  {var_name}: {size_mb:.2f} MB")
    #print()

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
max_slices = 120 * resolution_increase

loopRef = []
tablelength_data = []

dataframe_slice = []

# Log initial memory state
log_memory_usage("Script start")

# Main processing loop
# TODO: Consider to use multiprocessing for parallel processing
# TODO: Consider to use a progress bar for better user experience
# TODO: Consider to create a subject function to encapsulate the logic

explored_degrees = [0, 10, 20, 30, 40, 50, 60]
expected_length = 100

errors = []

for degrees in explored_degrees:
    
    nifti_img = main(degrees=degrees)
    
    for factor in [1, 1.5, 2]:
    
        
        nifti_data = nifti_img.get_fdata(dtype=np.float32)
        
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
                if current_max_value != max_voxel_value:
                    segment_type += 1
                    current_max_value = max_voxel_value
            
            active_slice += 1
            
            #f = plt.imshow(selected_y_slice, cmap='gray')
            #plt.show()
            
            # Deal with centering the mask in a new version of the image
            binarized_slice = selected_y_slice > 0  # Binarize
            
            # Find connected components and properties
            labeled_image = measure.label(binarized_slice)
            props = measure.regionprops(labeled_image)
            
            if len(props) == 0:
                print(f"WARNING: No region found in slice {y}")
                continue
                        
            centroid = props[0].centroid
            orig_centroids = np.array(centroid)
            
            # Shift the centroid of the region to the center of the image
            x_center_shift = int(np.round(image_center[0] - orig_centroids[0]))
            z_center_shift = int(np.round(image_center[1] - orig_centroids[1]))
                                    
            if active_slice == 0:
                distance = (y_resolution * factor)
                
            elif active_slice > 0:
                # Calculate the distance between original centroids
                previous_slice = active_slice - 1
                
                x_orig = cc_value[previous_slice]['orig_centroid_x']
                z_orig = cc_value[previous_slice]['orig_centroid_z']
                y_orig = cc_value[previous_slice]['original_slice_yz']
                
                x_curr = orig_centroids[0]
                z_curr = orig_centroids[1]
                y_curr = y
                
                z_center_displacement = (z_curr - z_orig)
                x_center_displacement = (x_curr - x_orig)
                y_center_displacement = (y_curr - y_orig)
                
                zz = z_resolution * z_center_displacement
                xx = x_resolution * x_center_displacement
                yy = y_resolution * (y_center_displacement * factor)
                                
                distance = np.sqrt(xx*xx + yy*yy + zz*zz)
                
            residual_distance = distance - (factor*y_resolution)
            
            n_slices = round(distance / y_resolution)
            n_slices_upsampled = round(distance / y_resolution * 10)
            
            slice_gap = round(residual_distance / y_resolution)
            
            slice_gap_upsampled = round(residual_distance / y_resolution * 10)
            
            length_optical_nerve_gap += (slice_gap_upsampled * y_resolution / 10 + y_resolution)
            length_optical_nerve += distance
                
            # Initialize dictionary for this slice
            slice_data = {}
            slice_data['current_slice_yz'] = active_slice
            slice_data['original_slice_yz'] = y
            slice_data['circshift_x'] = x_center_shift
            slice_data['circshift_z'] = z_center_shift
                        
            # Account for obliqueness of the optic n
            slice_data['orig_centroid_x'] = orig_centroids[0]
            slice_data['orig_centroid_z'] = orig_centroids[1]
            
            slice_data['distance'] = distance
            slice_data['residual_distance'] = residual_distance
            
            slice_data['slice_gap'] = slice_gap
            slice_data['slice_gap_upsampled'] = slice_gap_upsampled
            
            slice_data['length_on'] = length_optical_nerve
            slice_data['length_on_gap'] = length_optical_nerve_gap
        
            cc_value.append(slice_data)
            
        error_gap = expected_length - length_optical_nerve_gap
        error_distance = expected_length - length_optical_nerve
        
        error_data = {
            'degrees': degrees,
            'factor': factor,
            'length_on': length_optical_nerve,
            'length_on_gap': length_optical_nerve_gap,
            'error_gap': error_gap,
            'error_distance': error_distance    
        }
        
        errors.append(error_data)
          
        sliceframe = pd.DataFrame(cc_value)
        #sliceframe.to_csv(f"/home/robbis/sliceframe_degrees-{degrees}_factor-{factor}.csv", 
        #                  index=False)

errors = pd.DataFrame(errors)
errors.to_csv("/home/robbis/errors.csv", index=False)

print(errors)

import seaborn as sns
sns.lineplot(errors, x='degrees', y='error_gap', hue='factor')
