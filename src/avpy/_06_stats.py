import pandas as pd
import nibabel as ni
import numpy as np
import os
from sekupy.results import apply_function, filter_dataframe
from scipy.stats import ttest_ind
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as pl
import pingouin as pg
import os.path as op

parent_dir = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
atlas_dir = op.join(parent_dir, "atlas")
print(f"Parent directory: {parent_dir}")

def create_nerve_map(dataframe, feature):
    
    background_image = ni.load(op.join(atlas_dir, "aVP-24_prob100.nii"))
    atlas = background_image.get_fdata()
    n_slices = atlas.shape[1]
    
    nerve_map = np.zeros((atlas.shape[0],
                          atlas.shape[1], 
                          atlas.shape[2]))
    
    for y in range(n_slices):
        nerve_map[:, y, :][atlas[:, y, :] != 0] = dataframe[feature].values[y]
        
    return nerve_map



def plot_nerve(nerve_map, 
               threshold,
               comparison='equal', 
               colormap=pl.cm.magma,
               title="Nerve Map",
               vlim=None,
               figsize=(7, 18)
               ):
    
    background_image = ni.load(op.join(atlas_dir, "aVP-24_prob100.nii"))
    background_data = background_image.get_fdata()
    resolution = background_image.header['pixdim'][1]
    x_dim = background_data.shape[0]
    y_dim = background_data.shape[1]
    
    fig, ax = pl.subplots(figsize=figsize)
    ax.imshow(background_data[:, :, 35].T, 
              cmap=pl.cm.gray, 
              origin='lower', 
              aspect='equal')

    
    fx_comparison_dict = {
        'equal': np.equal,
        'greater': np.less,
        'less': np.greater
    }
    
    fx_comparison = fx_comparison_dict.get(comparison, np.equal)       
    
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


def main(path="./", dataset_a="HC", dataset_b="PTS"):
    
    do_figures = False
    dataframe = []

    # TODO: turn this into parameters
    features = ['Eccent', 'CSArea']
    sides = ['r', 'l']
    image_type = 'normalized'
    
    path_map = op.join(path, "maps")
    results_fname = op.join(path, "{group}", "results", "aVP_slice_data_iso.xlsx")
    map_name = "sub-group_feature-{feature}_group-{group}_side-{side}_on.nii.gz"
    
    atlas = ni.load(op.join("aVP-24_label.nii"))
    atlas_data = atlas.get_fdata()
    x_dim, y_dim, z_dim = atlas_data.shape
    n_slices = atlas.shape[1]
    
    p_value = 0.05
    bonferroni_value = p_value / n_slices

    groups = [dataset_a, dataset_b]

    # Read data for each group
    for group in groups:
        df = pd.read_excel(results_fname.format(group=group))
        df['group'] = [group] * df.shape[0]
        
        dataframe.append(df)
    
    full_dataframe = pd.concat(dataframe, ignore_index=True)

    
    for feature in features:
        for side in sides:
            for group in groups:
            
                df = filter_dataframe(full_dataframe, 
                                      group=[group], 
                                      side=[side], 
                                      type=[image_type])        
                df = apply_function(df, keys=['original_slice_yz'], 
                                    attr=feature, fx=lambda x: x.mean(0))
                
                nerve_map = create_nerve_map(df, feature)
                
                map_fname = map_name.format(group=group,
                                            side=side, 
                                            feature=feature)
                
                nerve_image = ni.Nifti1Image(nerve_map, atlas.affine)
                ni.save(nerve_image, op.join(path_map, map_fname))  
                
                if do_figures:
                    pl.figure()
                    pl.imshow(nerve_map[:,:,35], cmap=pl.cm.magma)
                    pl.title(f"Group: {group} - Side: {side} - Feature: {feature}")
                    pl.colorbar()
                    
                
    ###################################################################################
    # 3) Tests


    for feature in features:
        for side in sides:
                    
            df = filter_dataframe(full_dataframe, 
                                  side=[side], 
                                  type=[image_type])        
            
            nerve_map_t = np.zeros((atlas.shape[0], atlas.shape[1], atlas.shape[2]))
            nerve_map_p = np.zeros((atlas.shape[0], atlas.shape[1], atlas.shape[2]))
            
            for y in range(n_slices):
                
                df_slice_a = filter_dataframe(df, curr_sli_yz=[y+1], group=[dataset_a])
                df_slice_b = filter_dataframe(df, curr_sli_yz=[y+1], group=[dataset_b])
                
                t, p = ttest_ind(df_slice_a[feature].values, 
                                 df_slice_b[feature].values)
                
                nerve_map_t[:, y, :][atlas_data[:, y, :] != 0] = t
                nerve_map_p[:, y, :][atlas_data[:, y, :] != 0] = p
                
                threshold_image = nerve_map_t * (nerve_map_p < bonferroni_value)
                
            
            map_fname = map_name.format(group='t',
                                        side=side, 
                                        feature=feature)
            nerve_image = ni.Nifti1Image(nerve_map_t, atlas.affine)
            ni.save(nerve_image, op.join(path_map, map_fname))
            
            map_fname = map_name.format(group='p',
                                        side=side, 
                                        feature=feature)
            nerve_image = ni.Nifti1Image(nerve_map_p, atlas.affine)
            ni.save(nerve_image, op.join(path_map, map_fname))
            
            if do_figures:
                pl.figure()
                pl.imshow(threshold_image[:, :, 35], cmap=pl.cm.coolwarm, vmin=-5, vmax=5)
                pl.title(f"Side: {side} - Feature: {feature}")
                pl.colorbar()
                

    ###################################################################################
    # 2) Plot different values

    colormaps = [pl.cm.viridis, pl.cm.turbo]
    limits = [(0.4, 1), (5, 19)]


    for f, feature in enumerate(features):
        for group in groups:
            df = filter_dataframe(full_dataframe, 
                                  group=[group], 
                                  side=[side], 
                                  type=[image_type])        
            df = apply_function(df, 
                                keys=['original_slice_yz', 'subject_id'], 
                                attr=feature, 
                                fx=lambda x: x.mean(0))
            
            df = apply_function(df,
                                keys=['original_slice_yz'],
                                attr=feature,
                                fx=lambda x: x.mean(0))
            
            
            nerve_map = create_nerve_map(df, feature)
            
            if do_figures:
                fig, ax = plot_nerve(nerve_map,
                                     threshold=0,
                                     comparison='equal',
                                     colormap=colormaps[f],
                                     title=f"{feature} map in {group}",
                                     vlim=limits[f])
            
                fig.savefig(
                    op.join(path_map, 
                            f"sub-group_feature-{feature}_group-{group}_side-both_on.png")
                    )

    ###################################################################################
    # 2.1) Generate xls file
    lap = 0
    for feature in features:
        for group in groups:            
            df = filter_dataframe(full_dataframe, 
                                  group=[group], 
                                  side=[side], 
                                  type=[image_type])        
            df_mean = apply_function(df, 
                                keys=['original_slice_yz'], 
                                attr=feature, 
                                fx=lambda x: x.mean(0))
            df_std = apply_function(df,
                                keys=['original_slice_yz'],
                                attr=feature,
                                fx=lambda x: x.std(0))
            
            
            if lap == 0:
                dfs = df_mean.copy()
            
            lap += 1
            
            dfs[f"{feature}_mean_{group}"] = df_mean[feature].values
            dfs[f"{feature}_std_{group}"] = df_std[feature].values
            
            

    ###################################################################################
    # 3) Plot different values


    extension_fig = 'png'


    for feature in features:
                
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
            
            df_slice_a = filter_dataframe(df, curr_sli_yz=[y+1], group=[dataset_a])
            df_slice_b = filter_dataframe(df, curr_sli_yz=[y+1], group=[dataset_b])
            
            t, p = ttest_ind(df_slice_a[feature].values, 
                             df_slice_b[feature].values)
            
            nerve_map_t[:, y, :][atlas_data[:, y, :] != 0] = t
            nerve_map_p[:, y, :][atlas_data[:, y, :] != 0] = p
            
            ts.append(t)
            ps.append(p)
            
        dfs[f"{feature}_t"] = ts
        dfs[f"{feature}_p"] = ps
                    
        mask_p, p_fdr = pg.multicomp(nerve_map_p, method='fdr_bh')
        threshold_image = nerve_map_t * mask_p
        
        if do_figures:
            fig, ax = plot_nerve(threshold_image, 
                                threshold=0, 
                                comparison='equal', 
                                colormap=pl.cm.coolwarm,
                                title=f"FDR-corrected {feature} map in {dataset_a} vs {dataset_b}",
                                vlim=(-5, 5))
            
            fig.savefig(op.join(path_map, 
                                    f"sub-group_feature-{feature}_stats-ttestfdr_side-both_on.png"))
            
            
            fig, ax = plot_nerve(nerve_map_t,
                                threshold=0.,
                                comparison='equal',
                                colormap=pl.cm.coolwarm,
                                title=f"Unthresholded {feature} map in {dataset_a} vs {dataset_b}",
                                vlim=(-5, 5))

            fig.savefig(op.join(path_map, 
                                    f"sub-group_feature-{feature}_stats-ttestuncorrected_side-both_on.png"))        
            
            

            fig, ax = plot_nerve(nerve_map_t * (nerve_map_p < bonferroni_value),
                                threshold=0.,
                                comparison='equal',
                                colormap=pl.cm.coolwarm,
                                title=f"Bonferroni-corrected {feature} map in {dataset_a} vs {dataset_b}",
                                vlim=(-5, 5))
            
            fig.savefig(op.join(path_map,
                                    f"sub-group_feature-{feature}_stats-ttestbonferroni_side-both_on.png"))
        
    dfs.to_excel(op.join(path_map, "aVP_feature_stats.xlsx"))

if __name__ == '__main__':
    main()