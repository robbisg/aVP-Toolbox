import os
import numpy as np
import nibabel as nib
from skimage.measure import regionprops
from scipy.ndimage import shift

study_path = '/media/robbis/DATA/fmri/optical_nerve/data/CISS_Manual_Segmentations_HC+PTS/PTS/'

inPath = os.path.join(study_path, 'data/proc')
outImPath = os.path.join(study_path, 'data/proc')
outResPath = os.path.join(study_path, 'results')

DataFile = os.path.join(outResPath, 'aVP_slice_data.xlsx')
HoleFile = os.path.join(outResPath, 'log_check_hole.xlsx')
RangeFile = os.path.join(outResPath, 'log_check_range.xlsx')
LenStretchFile = os.path.join(outResPath, 'aVP_section_CSA_length.xlsx')

tablabels = ['curr_sli_yz', 'orig_sli_yz', 'point_y', 'point_z', 'circshift_y', 'circshift_z', 'dist', 'int_dist_x10', 'len_on', 'tot_len', 'mMax', 'save_len', 'CSArea', 'Eccent', 'MajAxis', 'MinAxis', 'AvgCSA']

stretxt = ['Subject', 'ONsection', 'side', 'TotLength', 'OT_length', 'OC_length', 'iCran_length', 'iCan_length', 'iOrb_length', 'OT_CSA', 'OC_CSA', 'iCran_CSA', 'iCan_CSA', 'iOrb_CSA', 'SegmCode 1', 'SegmCode 2', 'SegmCode 3', 'SegmCode 4', 'SegmCode 5']

rangetxt = ['Subject', 'ONsection', 'side', 'Slice...']
holetxt = ['Subject', 'ONsection', 'side', 'Slice...']

Lin4image = '_linearize_4bc.nii.gz'
Lin4mat = '_linearize_4bc.mat'
Norm4image = '_normalized_4bc.nii.gz'
Norm4mat = '_normalized_4bc.mat'
fullNorm4image = '_full_normalized_4bc.nii.gz'

sheet = 1
sides = ['l', 'r']
isbj = 0
tablelength = []

maxNslices = 2500
ismask = 0  # 0: use the combination of aVP segments. 1: use the individual aVP segments (not tested)

with open(os.path.join(study_path, 'data/sbj.list'), 'r') as file:
    subject_list = file.read().splitlines()

resolution_increase = 10

for isbj, sbj in enumerate(subject_list):

    for side in range(2):
        slice_results = []
        fname = os.path.join(inPath, sbj, f'on_{sides[side]}')

        print(f'INFO: Analyzing {sbj} - on_{sides[side]}')

        if not os.path.exists(f'{fname}.nii.gz'):
            print(f'could not find: {fname}.nii.gz')
            continue

        aswap = nib.load(f'{fname}.nii.gz')
        aa = aswap
        #aa_img = np.flip(aswap.get_fdata(), axis=0)
        
        aa_img = aswap.get_fdata()
        oo = aa

        x_dim, y_dim, z_dim = aa_img.shape

        x_resolution = aa.header['pixdim'][1]
        y_resolution = aa.header['pixdim'][2]
        z_resolution = aa.header['pixdim'][3]

        image_center = [np.round(z_dim / 2), np.round(x_dim / 2)]
        active_slice_counter = 0
        segment_type = 0

        table = []

        current_max_value = 0
        
        nonzero_voxels = np.unique(np.nonzero(aa_img)[1])

        for y in nonzero_voxels:
            selected_y_slice = aa_img[:, y, :]

            max_voxel_value = np.max(selected_y_slice)
            
            nonzero_values = np.unique(selected_y_slice[selected_y_slice > 0])
            if len(nonzero_values) > 1:
                print(f'WARNING: More than one nonzero value in slice {y}')

            if active_slice_counter == 0:
                segment_type = 1
                current_max_value = max_voxel_value
            else:
                if current_max_value != max_voxel_value:
                    segment_type += 1

            active_slice_counter += 1

            slice_results.append({
                'current_slice_yz': active_slice_counter,
                'original_slice_yz': y,
                'flagC': segment_type,
                'mm': max_voxel_value
            })

            # Get the centroid of the optical nerve
            binarized_slice = np.int_(selected_y_slice > 0)
            st = regionprops(binarized_slice)
            temp_centroids = np.array([prop.centroid for prop in st])
            orig_centroids = temp_centroids[0][::-1]

            slice_properties = regionprops(binarized_slice)[0]

            slice_results[-1].update({
                'majaxis': slice_properties.major_axis_length * x_resolution,
                'minaxis': slice_properties.minor_axis_length * z_resolution,
                'area': slice_properties.area * x_resolution * z_resolution,
                'eccent': slice_properties.eccentricity
            })

            x_center_shift = int(image_center[1] - round(orig_centroids[1]))
            z_center_shift = int(image_center[0] - round(orig_centroids[0]))

            cc = np.roll(selected_y_slice, shift=(x_center_shift, z_center_shift), axis=(0, 1))
            slice_results[-1]['circshift'] = [x_center_shift, z_center_shift]

            if ismask == 0:
                oo_img[:, y, :] = cc
            else:
                oo_img[:, y, :] = np.sign(cc)

            length_on = 0
            slice_results[-1].update({
                'point': orig_centroids,
                'point_original': temp_centroids,
                'distance': 0,
                'length_on': length_on,
                'slice': oo_img[:, y, :]
            })

            slice_results[-1]['vol'] = np.repeat(oo_img[:, y, :][:, np.newaxis, :], 10, axis=1)
            slice_results[-1]['intra'] = np.zeros_like(slice_results[-1]['vol'])
            slice_results[-1]['length_on'] += y_resolution
            slice_results[-1]['total_length'] = slice_results[-1]['length_on']
            slice_results[-1]['save_length'] = 0
            slice_results[-1]['int_distance_x10'] = 0
            slice_results[-1]['avgCSA'] = 0
            
            
            _slice_data = {
                'subject': sbj,
                'section': 'on',
                'side': sides[side],                
                'current_slice_yz': active_slice_counter,
                'original_slice_yz': y,
                'flagC': segment_type,
                'mm': max_voxel_value,
                'majaxis': slice_properties.major_axis_length * x_resolution,
                'minaxis': slice_properties.minor_axis_length * z_resolution,
                'area': slice_properties.area * x_resolution * z_resolution,
                'eccent': slice_properties.eccentricity,
                'x_center_shift': x_center_shift,
                'z_center_shift': z_center_shift,
                'point': orig_centroids,
                'point_original': temp_centroids,
                'distance': 0,
                'length_on': length_on,
                'slice': oo_img[:, y, :]
            }
            
            

            if active_slice_counter > 1:
                previous_slice = active_slice_counter - 1

                z_center_displacement = slice_results[-1]['point'][0] - slice_results[previous_slice]['point'][0]
                x_center_displacement = slice_results[-1]['point'][1] - slice_results[previous_slice]['point'][1]

                zz = z_resolution * z_center_displacement
                xx = x_resolution * x_center_displacement
                yy = y_resolution * 2
                slice_results[-1]['distance'] = np.sqrt(xx**2 + yy**2 + zz**2)

                distance_gap = round((slice_results[-1]['distance'] - 2 * y_resolution) / y_resolution * 10)

                if distance_gap < 0:
                    print(f'WARNING: Negative distance between slice {y} and {slice_results[previous_slice]["original_slice_yz"]}')
                    distance_gap = 0

                if distance_gap > 0:
                    for kk in range(distance_gap):
                        if kk < distance_gap // 2:
                            slice_results[-1]['intra'][:, kk, :] = slice_results[previous_slice]['vol'][:, 0, :]
                        else:
                            slice_results[-1]['intra'][:, kk, :] = slice_results[-1]['vol'][:, 0, :]

                slice_results[-1]['length_on'] += distance_gap / 10 * y_resolution
                slice_results[-1]['total_length'] = slice_results[previous_slice]['total_length'] + slice_results[-1]['length_on']
                slice_results[-1]['int_distance_x10'] = distance_gap

                if slice_results[-1]['mm'] != slice_results[previous_slice]['mm']:
                    slice_results[previous_slice]['save_length'] = slice_results[previous_slice]['total_length']
                    slice_results[previous_slice]['avgCSA'] = sumCSA / countCSA
                    countCSA = 1
                    sumCSA = slice_results[-1]['area']
                else:
                    sumCSA += slice_results[-1]['area']
                    countCSA += 1

                table.append([
                    slice_results[previous_slice]['current_slice_yz'],
                    slice_results[previous_slice]['original_slice_yz'],
                    *slice_results[previous_slice]['point'],
                    *slice_results[previous_slice]['circshift'],
                    slice_results[previous_slice]['distance'],
                    slice_results[previous_slice]['int_distance_x10'],
                    slice_results[previous_slice]['length_on'],
                    slice_results[previous_slice]['total_length'],
                    slice_results[previous_slice]['mm'],
                    slice_results[previous_slice]['save_length'],
                    slice_results[previous_slice]['area'],
                    slice_results[previous_slice]['eccent'],
                    slice_results[previous_slice]['majaxis'],
                    slice_results[previous_slice]['minaxis'],
                    slice_results[previous_slice]['avgCSA']
                ])
                
                
                

            elif active_slice_counter == 1:
                countCSA = 1
                sumCSA = slice_results[-1]['area']

        if active_slice_counter == 0:
            print('no ON elements found  -  quitting')
        else:
            slice_results[-1]['save_length'] = slice_results[-1]['total_length']
            slice_results[-1]['avgCSA'] = sumCSA / countCSA
            table.append([
                slice_results[-1]['current_slice_yz'],
                slice_results[-1]['original_slice_yz'],
                *slice_results[-1]['point'],
                *slice_results[-1]['circshift'],
                slice_results[-1]['distance'],
                slice_results[-1]['int_distance_x10'],
                slice_results[-1]['length_on'],
                slice_results[-1]['total_length'],
                slice_results[-1]['mm'],
                slice_results[-1]['save_length'],
                slice_results[-1]['area'],
                slice_results[-1]['eccent'],
                slice_results[-1]['majaxis'],
                slice_results[-1]['minaxis'],
                slice_results[-1]['avgCSA']
            ])

            length = np.array(table)[:, 11]
            length = length[length != 0]

            careas = np.array(table)[:, 16]
            careas = careas[careas != 0]

            mNms = np.array(table)[:, 10]
            mNms = mNms[careas != 0]

            subsidInd = (isbj - 1) * 2 + side

            loopRef = [sbj, 'on', sides[side]]
            noopRef = [[*loopRef, tablabels[typ]] for typ in range(16)]
            tablelength.append([
                slice_results[-1]['total_length'],
                length[0],
                length[1] - length[0],
                length[2] - length[1],
                length[3] - length[2],
                slice_results[-1]['total_length'] - length[3],
                careas[0],
                careas[1],
                careas[2],
                careas[3],
                careas[4],
                mNms[0],
                mNms[1],
                mNms[2],
                mNms[3],
                mNms[4]
            ])

            xlRange = f'A{subsidInd * len(table[0]) + 1}'
            # Write to DataFile using xlwt or openpyxl

            bbb = slice_results[0]['vol']
            interpolation_counter = 10

            n_slice = len(slice_results)

            print('INFO: Nerve Interpolation...')

            for ii in range(1, n_slice):
                n_dist = slice_results[ii]['intra'].shape[1]
                new_interval = range(interpolation_counter + 1, interpolation_counter + n_dist + 1)
                bbb[:, new_interval, :] = slice_results[ii]['intra']
                interpolation_counter += n_dist
                interval = range(interpolation_counter + 1, interpolation_counter + 11)
                bbb[:, interval, :] = slice_results[ii]['vol']
                interpolation_counter += 10

            original_linearized_slices = interpolation_counter

            print('INFO: Image creation...')
            if interpolation_counter < maxNslices:
                for ii in range(interpolation_counter + 1, maxNslices + 1):
                    bbb[:, ii, :] = slice_results[-1]['vol'][:, 0, :] * 0
            else:
                print(f'Houston, we have a problem! More than {maxNslices} slices!!!')
                quit()

            hole_list = []
            hole_counter = 0

            print('INFO: Hole filling...')

            for slice in range(1, maxNslices - 2):
                if np.max(bbb[:, slice + 1, :]) == 0 and \
                   np.max(bbb[:, slice, :]) > 0 and \
                   np.max(bbb[:, slice + 2, :]) > 0:
                    hole_counter += 1
                    hole_list.append(slice)
                    bbb[:, slice + 1, :] = bbb[:, slice + 2, :]

            xlRange = f'A{subsidInd}'
            # Write to HoleFile using xlwt or openpyxl
            if hole_list:
                xlRange = f'D{subsidInd}'
                # Write to HoleFile using xlwt or openpyxl

            dd = aa
            dd.header['pixdim'][3] /= 10
            dd.header['dim'][3] = bbb.shape[1]
            dd_img = bbb

            print('INFO: Disk writing...')

            nib.save(nib.Nifti1Image(dd_img, dd.affine, dd.header), os.path.join(outImPath, sbj, f'on{sides[side]}{Lin4image}'))
            np.save(os.path.join(outImPath, sbj, f'on_{sides[side]}{Lin4mat}'), slice_results)

            lengthfactor = maxNslices / round(slice_results[-1]['total_length'] / dd.header['pixdim'][3])
            zz = np.zeros_like(slice_results[0]['vol'][:, 0, :])

            check_range = np.zeros(maxNslices)

            print('INFO: Normalization...')

            for ii in range(maxNslices):
                jj = round(ii / maxNslices * original_linearized_slices)
                if jj < 1:
                    jj = 1
                check_range[ii] = jj
                zz[:, ii, :] = bbb[:, jj, :]

                if ii > 2:
                    if check_range[ii - 2] > 0 and check_range[ii - 1] == 0 and check_range[ii] > 0:
                        zz[:, ii - 1, :] = zz[:, ii - 2, :]

            xx = dd
            xx_img = zz
            nib.save(nib.Nifti1Image(xx_img, xx.affine, xx.header), os.path.join(outImPath, sbj, f'on{sides[side]}{Norm4image}'))
            np.save(os.path.join(outImPath, sbj, f'on_{sides[side]}{Norm4mat}'), slice_results)

            xlRange = f'A{subsidInd + 1}'
            # Write to RangeFile using xlwt or openpyxl
            xlRange = f'D{subsidInd + 1}'
            # Write to RangeFile using xlwt or openpyxl