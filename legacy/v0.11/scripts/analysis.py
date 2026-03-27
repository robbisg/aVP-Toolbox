import pandas as pd
import nibabel as ni
import numpy as np
import os
from sekupy.results import apply_function, filter_dataframe
from scipy.stats import ttest_ind
from matplotlib.ticker import FormatStrFormatter

path_map = "/media/robbis/DATA/fmri/optical_nerve/data/CISS_Manual_Segmentations_HC+PTS/maps/"

fname = "/media/robbis/DATA/fmri/optical_nerve/data/CISS_Manual_Segmentations_HC+PTS/{group}/results/aVP_slice_data_iso.xlsx"

map_name = "sub-group_feature-{feature}_group-{group}_side-{side}_on.nii.gz"

groups = ["HC", "PTS"]

info_columns = 4
rows_per_block = 11

full_dataframe = []


def create_nerve_map(dataframe, feature):
    
    background_image = ni.load("/media/robbis/DATA/fmri/optical_nerve/toolbox/atlas/aVP-24_prob100.nii")
    atlas = background_image.get_fdata()
    
    nerve_map = np.zeros((atlas.shape[0], atlas.shape[1], atlas.shape[2]))
    
    for y in range(n_slices):
        nerve_map[:, y, :][atlas_data[:, y, :] != 0] = dataframe[feature].values[y]
        
    return nerve_map



def plot_nerve(nerve_map, 
               threshold, 
               comparison='equal', 
               colormap=pl.cm.magma,
               title="Nerve Map",
               vlim=None,
               figsize=(7, 18)
               ):
    
    background_image = ni.load("/media/robbis/DATA/fmri/optical_nerve/toolbox/atlas/aVP-24_prob100.nii")
    background_data = background_image.get_fdata()
    resolution = background_image.header['pixdim'][1]
    x_dim = background_data.shape[0]
    y_dim = background_data.shape[1]
    
    fig, ax = pl.subplots(figsize=figsize)
    ax.imshow(background_data[:, :, 35].T, 
              cmap=pl.cm.gray, 
              origin='lower', 
              aspect='equal')

    
    if comparison == 'equal':
        fx_comparison = np.equal
    elif comparison == 'greater':
        fx_comparison = np.less
    elif comparison == 'less':
        fx_comparison = np.greater
        
    if vlim is not None:
        vmin, vmax = vlim
    else:
        vmin, vmax = nerve_map.min(), nerve_map.max()
    
    # Masking 
    mask = fx_comparison(nerve_map, threshold)
    
    masked_nerve = np.ma.masked_where(mask,
                                      nerve_map)
    
    image = ax.imshow(masked_nerve[:, :, 35].T, 
                      cmap=colormap, 
                      alpha=0.9, 
                      origin='lower', 
                      aspect='equal',
                      vmin=vmin,
                      vmax=vmax
                      )
    
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    #cbar.set_label("Overlay Value")

    # Set plot title and labels (now with units)
    ax.set_title(title)
    ax.set_xlabel("x-length (mm)")  # Units added
    ax.set_ylabel("y-length (mm)")  # Units added
    

    # Set the ticks to be at the correct mm intervals
    x_ticks = np.arange(0, x_dim + 1, 25)
    y_ticks = np.arange(0, y_dim + 1, 25)[::-1]
    
    x_ticks_labels = np.arange(0, x_dim * resolution + resolution, 10 * resolution)
    y_ticks_labels = np.arange(0, y_dim * resolution + resolution, 10 * resolution)
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    ax.xaxis.set_major_formatter(lambda x, pos: f"{int(x*resolution):.1f}") # Example: 1 decimal place
    ax.yaxis.set_major_formatter(lambda x, pos: f"{int(x*resolution):.1f}")
    
    ax.set_xlim(100, 150)
    
    return fig, ax





for group in groups:
    
    df = pd.read_excel(fname.format(group=group), header=None)
        
    n_rows = df.shape[0]
     
    n_blocks = n_rows // rows_per_block
    
    dataframe = []
    
    for i in range(n_blocks):
        start = i * rows_per_block
        end = (i + 1) * rows_per_block
        block = df.iloc[start:end]
        
        column_names = block.iloc[:, 4].values
        data_values = block.iloc[:, 5:].dropna(axis=1).values.T
        
        print(data_values.shape)
        print(block.iloc[0, :4])
        
        data_frames = pd.DataFrame(data_values, columns=column_names)
        data_frames['subject_id'] = [block.iloc[0, 0]] * len(data_values)
        data_frames['type'] = [block.iloc[0, 1]] * len(data_values)
        data_frames['image_name'] = [block.iloc[0, 2]] * len(data_values)
        data_frames['side'] = [block.iloc[0, 3]] * len(data_values)
        data_frames['group'] = [group] * len(data_values)
                
        dataframe.append(data_frames)    

    dataframe = pd.concat(dataframe)

    full_dataframe.append(dataframe)
    
full_dataframe = pd.concat(full_dataframe)

###################################################################################
# Analysis

import nibabel as ni

n_slices = 102
image_type = 'normalized'
atlas = ni.load("/media/robbis/DATA/fmri/optical_nerve/toolbox/atlas/aVP-24_label.nii")
atlas_data = atlas.get_fdata()


# 1) Maps of features for each group

for side in ['r', 'l']:
    for group in groups:
        for feature in ['Eccent', 'CSArea']:
           
            df = filter_dataframe(full_dataframe, group=[group], side=[side], type=[image_type])        
            df = apply_function(df, keys=['orig_sli_yz'], attr=feature, fx=lambda x: x.mean(0))
            
            nerve_map = np.zeros((atlas.shape[0], atlas.shape[1], atlas.shape[2]))
            
            for y in range(n_slices):
                nerve_map[:, y, :][atlas_data[:, y, :] != 0] = df[feature].values[y]
                
            ni.save(ni.Nifti1Image(nerve_map, atlas.affine), os.path.join(path_map, 
                                                                          map_name.format(group=group, 
                                                                                          side=side, 
                                                                                          feature=feature)))
            
            pl.figure()
            pl.imshow(nerve_map[:,:,35], cmap=pl.cm.magma)
            pl.title(f"Group: {group} - Side: {side} - Feature: {feature}")
            pl.colorbar()
                
            
###################################################################################
# 3) Tests

bonferroni_value = 0.05 / n_slices

for side in ['r', 'l']:
    for feature in ['Eccent', 'CSArea']:
                
        df = filter_dataframe(full_dataframe, side=[side], type=[image_type])        
        #df = apply_function(df, keys=['orig_sli_yz'], attr=feature, fx=lambda x: x.mean(0))
        
        nerve_map_t = np.zeros((atlas.shape[0], atlas.shape[1], atlas.shape[2]))
        nerve_map_p = np.zeros((atlas.shape[0], atlas.shape[1], atlas.shape[2]))
        
        for y in range(n_slices):
            
            df_slice_patients = filter_dataframe(df, curr_sli_yz=[y+1], group=['PTS'])
            df_slice_healthy =  filter_dataframe(df, curr_sli_yz=[y+1], group=['HC'])
            
            t, p = ttest_ind(df_slice_healthy[feature].values, 
                             df_slice_patients[feature].values)
            
            nerve_map_t[:, y, :][atlas_data[:, y, :] != 0] = t
            nerve_map_p[:, y, :][atlas_data[:, y, :] != 0] = p
            
            threshold_image = nerve_map_t * (nerve_map_p < bonferroni_value)
            
            
        ni.save(ni.Nifti1Image(nerve_map_t, atlas.affine), 
                os.path.join(path_map, map_name.format(group='t',
                                                       side=side,
                                                       feature=feature)))
        
        ni.save(ni.Nifti1Image(nerve_map_p, atlas.affine), 
                os.path.join(path_map, map_name.format(group='p', 
                                                       side=side, 
                                                       feature=feature)))
        
        pl.figure()
        pl.imshow(threshold_image[:, :, 35], cmap=pl.cm.coolwarm, vmin=-5, vmax=5)
        pl.title(f"Side: {side} - Feature: {feature}")
        pl.colorbar()
        

##############################################################################################
# New analyses
import pengouin as pg




###################################################################################
# 1) Verify that left and right are not different

for feature in ['Eccent', 'CSArea']:
    for group in groups:
        df = filter_dataframe(full_dataframe, group=[group], type=[image_type])        
        #df = apply_function(df, keys=['orig_sli_yz'], attr=feature, fx=lambda x: x.mean(0))
        
        p_values = []
        
        for y in range(n_slices):
            
            df_left = filter_dataframe(df, side=['l'], curr_sli_yz=[y+1])
            df_right = filter_dataframe(df, side=['r'], curr_sli_yz=[y+1])
            
            t, p = ttest_ind(df_left[feature].values, df_right[feature].values)
            
            p_values.append(p)
            
            if p < bonferroni_value:
                print(f"Group: {group} - Feature: {feature} - Slice: {y} - T: {t} - P: {p}")
            
        pg.multicomp(p_values, method='fdr_bh')


###################################################################################
# 2) Plot different values

colormaps = [pl.cm.viridis, pl.cm.turbo]
limits = [(0.4, 1), 
          (5, 19)]


for f, feature in enumerate(['Eccent', 'CSArea']):
    for group in groups:
        df = filter_dataframe(full_dataframe, group=[group], side=[side], type=[image_type])        
        df = apply_function(df, 
                            keys=['orig_sli_yz', 'subject_id'], 
                            attr=feature, 
                            fx=lambda x: x.mean(0))
        
        df = apply_function(df,
                            keys=['orig_sli_yz'],
                            attr=feature,
                            fx=lambda x: x.mean(0))
        
           
        nerve_map = np.zeros((atlas.shape[0], atlas.shape[1], atlas.shape[2]))
            
        for y in range(n_slices):
            nerve_map[:, y, :][atlas_data[:, y, :] != 0] = df[feature].values[y]

        fig, ax = plot_nerve(nerve_map,
                             threshold=0,
                             comparison='equal',
                             colormap=colormaps[f],
                             title=f"{feature} map in {group}",
                             vlim=limits[f])
        
        fig.savefig(os.path.join(path_map, 
                                 f"sub-group_feature-{feature}_group-{group}_side-both_on.png"))

###################################################################################
# 2.1) Generate xls file


lap = 0
for feature in ['Eccent', 'CSArea']:
    for group in groups:
        
        
        df = filter_dataframe(full_dataframe, group=[group], side=[side], type=[image_type])        
        df_mean = apply_function(df, 
                            keys=['orig_sli_yz'], 
                            attr=feature, 
                            fx=lambda x: x.mean(0))
        df_std = apply_function(df,
                            keys=['orig_sli_yz'],
                            attr=feature,
                            fx=lambda x: x.std(0))
        
        
        if lap == 0:
            dfs = df_mean.copy()
            #dfs.rename(columns={feature: f"{feature}_mean_{group}"}, inplace=True)
        
        lap += 1
        
        dfs[f"{feature}_mean_{group}"] = df_mean[feature].values
        dfs[f"{feature}_std_{group}"] = df_std[feature].values
        
        
dfs.to_excel(os.path.join(path_map, "group_features.xlsx"))




###################################################################################
# 3) Plot different values

bonferroni_value = 0.05 / n_slices
extension_fig = 'png'


for feature in ['Eccent', 'CSArea']:
            
    df = filter_dataframe(full_dataframe, type=[image_type])        
    df = apply_function(df, 
                        keys=['curr_sli_yz', 'group', 'subject_id'], 
                        attr=feature, 
                        fx=lambda x: x.mean(0))
    
    nerve_map_t = np.zeros((atlas.shape[0], atlas.shape[1], atlas.shape[2]))
    nerve_map_p = np.zeros((atlas.shape[0], atlas.shape[1], atlas.shape[2]))
    
    ts = []
    ps = []
    
    for y in range(n_slices):
        
        df_slice_patients = filter_dataframe(df, curr_sli_yz=[y+1], group=['PTS'])
        df_slice_healthy =  filter_dataframe(df, curr_sli_yz=[y+1], group=['HC'])
        
        t, p = ttest_ind(df_slice_healthy[feature].values, 
                         df_slice_patients[feature].values)
        
        nerve_map_t[:, y, :][atlas_data[:, y, :] != 0] = t
        nerve_map_p[:, y, :][atlas_data[:, y, :] != 0] = p
        
        ts.append(t)
        ps.append(p)
        
    dfs[f"{feature}_t"] = ts
    dfs[f"{feature}_p"] = ps
        
        #threshold_image = nerve_map_t * (nerve_map_p < bonferroni_value)
    
    mask_p, p_fdr = pg.multicomp(nerve_map_p, method='fdr_bh')
    threshold_image = nerve_map_t * mask_p
    
    fig, ax = plot_nerve(threshold_image, 
                         threshold=0, 
                         comparison='equal', 
                         colormap=pl.cm.coolwarm,
                         title=f"FDR-corrected {feature} map in HC vs PTS",
                         vlim=(-5, 5))
    
    fig.savefig(os.path.join(path_map, 
                             f"sub-group_feature-{feature}_stats-ttestfdr_side-both_on.png"))
    
    
    fig, ax = plot_nerve(nerve_map_t,
                         threshold=0.,
                         comparison='equal',
                         colormap=pl.cm.coolwarm,
                         title=f"Unthresholded {feature} map in HC vs PTS",
                         vlim=(-5, 5))

    fig.savefig(os.path.join(path_map, 
                             f"sub-group_feature-{feature}_stats-ttestuncorrected_side-both_on.png"))        
    
    

    fig, ax = plot_nerve(nerve_map_t * (nerve_map_p < bonferroni_value),
                         threshold=0.,
                         comparison='equal',
                         colormap=pl.cm.coolwarm,
                         title=f"Bonferroni-corrected {feature} map in HC vs PTS",
                         vlim=(-5, 5))
    
    fig.savefig(os.path.join(path_map,
                             f"sub-group_feature-{feature}_stats-ttestbonferroni_side-both_on.png"))