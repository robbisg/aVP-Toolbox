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
# Statistical analysis module for aVP-Toolbox: generates nerve maps and performs 
# statistical comparisons between groups with fully parametrized configuration.

import pandas as pd
import nibabel as ni
import numpy as np
import os
import os.path as op
from sekupy.results import apply_function, filter_dataframe
from scipy.stats import ttest_ind
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as pl
import pingouin as pg
import logging

logger = logging.getLogger(__name__)
NAME = "stats"

# Initialize atlas paths
parent_dir = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
atlas_dir = op.join(parent_dir, "atlas")
atlas_name = "aVP-24_label.nii.gz"


class StatsConfig:
    """Configuration class for statistical analysis parameters."""
    
    def __init__(self, **kwargs):
        # Default feature and analysis settings
        self.features = kwargs.get('features', ['Eccent', 'CSArea'])
        self.sides = kwargs.get('sides', ['r', 'l'])
        self.image_type = kwargs.get('image_type', 'normalized')
        
        # Statistical parameters
        self.p_value = kwargs.get('p_value', 0.05)
        self.correction_method = kwargs.get('correction_method', 'bonferroni')
        self.background_threshold = kwargs.get('background_threshold', 0)
        
        # File and directory parameters
        self.results_filename = kwargs.get('results_filename', 'aVP_slice_data_iso.xlsx')
        self.maps_subdir = kwargs.get('maps_subdir', 'maps')
        self.results_subdir = kwargs.get('results_subdir', 'results')
        self.output_filename = kwargs.get('output_filename', 'aVP_feature_stats.xlsx')
        self.map_filename_template = kwargs.get('map_filename_template', 
                                              'sub-group_feature-{feature}_group-{group}_side-{side}_on.nii.gz')
        
        # Visualization parameters
        self.slice_display_index = kwargs.get('slice_display_index', 35)
        self.stat_plot_vlim = kwargs.get('stat_plot_vlim', (-5, 5))
        self.figure_extension = kwargs.get('figure_extension', 'png')
        self.feature_colormaps = kwargs.get('feature_colormaps', [pl.cm.viridis, pl.cm.turbo])
        self.feature_limits = kwargs.get('feature_limits', [(0.4, 1), (5, 19)])
        
        # Data processing parameters
        self.slice_column = kwargs.get('slice_column', 'curr_sli_yz')
        self.aggregation_keys = kwargs.get('aggregation_keys', ['original_slice_yz'])
        
        # Atlas and plotting parameters
        self.atlas_path = kwargs.get('atlas_path', None)
        self.generate_figures = kwargs.get('generate_figures', False)
        
        # Advanced plotting parameters
        self.plot_figsize = kwargs.get('plot_figsize', (7, 18))
        self.plot_alpha = kwargs.get('plot_alpha', 0.9)
        self.plot_tick_interval = kwargs.get('plot_tick_interval', 25)
        self.plot_xlim = kwargs.get('plot_xlim', (100, 150))
        self.colorbar_fraction = kwargs.get('colorbar_fraction', 0.046)
        self.colorbar_pad = kwargs.get('colorbar_pad', 0.03)
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return {
            'features': self.features,
            'sides': self.sides,
            'image_type': self.image_type,
            'p_value': self.p_value,
            'correction_method': self.correction_method,
            'generate_figures': self.generate_figures,
            'atlas_path': self.atlas_path,
            'results_filename': self.results_filename,
            'maps_subdir': self.maps_subdir,
            'results_subdir': self.results_subdir,
            'map_filename_template': self.map_filename_template,
            'slice_display_index': self.slice_display_index,
            'stat_plot_vlim': self.stat_plot_vlim,
            'feature_colormaps': self.feature_colormaps,
            'feature_limits': self.feature_limits,
            'output_filename': self.output_filename,
            'figure_extension': self.figure_extension,
            'background_threshold': self.background_threshold,
            'slice_column': self.slice_column,
            'aggregation_keys': self.aggregation_keys
        }
    
    @classmethod
    def from_file(cls, config_file):
        """Load configuration from JSON or YAML file."""
        import json
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith('.json'):
                    config_dict = json.load(f)
                elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    import yaml
                    config_dict = yaml.safe_load(f)
                else:
                    raise ValueError("Config file must be .json or .yaml/.yml")
            return cls(**config_dict)
        except Exception as e:
            logger.error(f"Error loading config from {config_file}: {e}")
            return cls()  # Return default config

def create_nerve_map(dataframe, feature, atlas_path=None, background_threshold=0):
    """
    Create a nerve map from dataframe values.
    
    Args:
        dataframe: Input dataframe with feature values
        feature: Feature column name to map
        atlas_path: Path to atlas file (optional, uses default if None)
        background_threshold: Threshold for background voxels (default: 0)
    """
    if atlas_path is None:
        atlas_path = op.join(atlas_dir, atlas_name)
    
    background_image = ni.load(atlas_path)
    atlas = background_image.get_fdata()
    n_slices = atlas.shape[1]
    
    nerve_map = np.zeros((atlas.shape[0],
                          atlas.shape[1], 
                          atlas.shape[2]))
    
    for y in range(n_slices):
        nerve_map[:, y, :][atlas[:, y, :] != background_threshold] = dataframe[feature].values[y]
        
    return nerve_map



def plot_nerve(nerve_map, 
               threshold,
               comparison='equal', 
               colormap=pl.cm.magma,
               title="Nerve Map",
               vlim=None,
               figsize=(7, 18),
               atlas_path=None,
               slice_index=35,
               tick_interval=25,
               resolution_multiplier=10,
               xlim=(100, 150),
               alpha=0.9,
               colorbar_fraction=0.046,
               colorbar_pad=0.03,
               xlabel="x-length (mm)",
               ylabel="y-length (mm)",
               background_cmap=pl.cm.gray
               ):
    """
    Plot nerve map with configurable parameters.
    
    Args:
        nerve_map: Input nerve map array
        threshold: Threshold value for comparison
        comparison: Comparison type ('equal', 'greater', 'less')
        colormap: Colormap for overlay
        title: Plot title
        vlim: Value limits tuple (vmin, vmax)
        figsize: Figure size tuple
        atlas_path: Path to atlas file
        slice_index: Z-slice to display (default: 35)
        tick_interval: Interval for axis ticks (default: 25)
        resolution_multiplier: Multiplier for resolution labels (default: 10)
        xlim: X-axis limits tuple
        alpha: Overlay transparency
        colorbar_fraction: Colorbar size fraction
        colorbar_pad: Colorbar padding
        xlabel: X-axis label
        ylabel: Y-axis label
        background_cmap: Background colormap
    """
    if atlas_path is None:
        atlas_path = op.join(atlas_dir, atlas_name)
    
    background_image = ni.load(atlas_path)
    background_data = background_image.get_fdata()
    resolution = background_image.header['pixdim'][1]
    x_dim = background_data.shape[0]
    y_dim = background_data.shape[1]
    
    fig, ax = pl.subplots(figsize=figsize)
    ax.imshow(background_data[:, :, slice_index].T, 
              cmap=background_cmap, 
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
    
    masked_nerve = np.ma.masked_where(mask, nerve_map)
    
    image = ax.imshow(masked_nerve[:, :, slice_index].T, 
                      cmap=colormap, 
                      alpha=alpha, 
                      origin='lower', 
                      aspect='equal',
                      vmin=vmin,
                      vmax=vmax
                      )
    
    cbar = fig.colorbar(image, ax=ax, fraction=colorbar_fraction, pad=colorbar_pad)

    # Set plot title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Set the ticks to be at the correct intervals
    x_ticks = np.arange(0, x_dim + 1, tick_interval)
    y_ticks = np.arange(0, y_dim + 1, tick_interval)[::-1]
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    ax.xaxis.set_major_formatter(lambda x, pos: f"{int(x*resolution):.1f}")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{int(x*resolution):.1f}")
    
    if xlim is not None:
        ax.set_xlim(xlim)
    
    return fig, ax


def generate_nerve_maps(path, dataset_a, dataset_b, features=None, sides=None, 
                       image_type='normalized', p_value=0.05, correction_method='bonferroni', 
                       generate_figures=False, debug=False, atlas_path=None,
                       results_filename="aVP_slice_data_iso.xlsx", maps_subdir="maps",
                       results_subdir="results", map_filename_template=None,
                       slice_display_index=35, stat_plot_vlim=(-5, 5), 
                       feature_colormaps=None, feature_limits=None,
                       output_filename="aVP_feature_stats.xlsx", figure_extension='png',
                       background_threshold=0, slice_column='curr_sli_yz',
                       aggregation_keys=None):
    """
    Generate nerve maps and perform statistical analysis between two datasets.
    
    Args:
        path (str): Root path containing the data
        dataset_a (str): Name of first dataset group  
        dataset_b (str): Name of second dataset group
        features (list): List of features to analyze (default: ['Eccent', 'CSArea'])
        sides (list): List of sides to analyze (default: ['r', 'l'])
        image_type (str): Type of image data to process (default: 'normalized')
        p_value (float): P-value threshold for statistical tests (default: 0.05)
        correction_method (str): Correction method ('bonferroni' or 'fdr')
        generate_figures (bool): Whether to generate and save figures
        debug (bool): Enable debug logging
        atlas_path (str): Path to atlas file (optional, uses default if None)
        results_filename (str): Name of results Excel file (default: 'aVP_slice_data_iso.xlsx')
        maps_subdir (str): Subdirectory for output maps (default: 'maps')
        results_subdir (str): Subdirectory for input results (default: 'results')
        map_filename_template (str): Template for map filenames (optional)
        slice_display_index (int): Z-slice index for visualization (default: 35)
        stat_plot_vlim (tuple): Value limits for statistical plots (default: (-5, 5))
        feature_colormaps (list): Colormaps for each feature (optional)
        feature_limits (list): Value limits for each feature (optional)
        output_filename (str): Output Excel filename (default: 'aVP_feature_stats.xlsx')
        figure_extension (str): File extension for figures (default: 'png')
        background_threshold (int): Background threshold for atlas (default: 0)
        slice_column (str): Column name for slice indices (default: 'curr_sli_yz')
        aggregation_keys (list): Keys for data aggregation (optional)
        
    Returns:
        pd.DataFrame: Combined results dataframe with statistics
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    
    logger.info(f"Starting statistical analysis: {dataset_a} vs {dataset_b}")
    
    # Set default values for optional parameters
    if features is None:
        features = ['Eccent', 'CSArea']
    if sides is None:
        sides = ['r', 'l']
    if feature_colormaps is None:
        feature_colormaps = [pl.cm.viridis, pl.cm.turbo]
    if feature_limits is None:
        feature_limits = [(0.4, 1), (5, 19)]
    if map_filename_template is None:
        map_filename_template = "sub-group_feature-{feature}_group-{group}_side-{side}_on.nii.gz"
    if aggregation_keys is None:
        aggregation_keys = ['original_slice_yz']
    
    dataframe = []
    do_figures = generate_figures
    
    # Create output directory
    path_map = op.join(path, maps_subdir)
    os.makedirs(path_map, exist_ok=True)
    
    results_fname = op.join(path, "{group}", results_subdir, results_filename)
    
    # Load atlas with error handling
    if atlas_path is None:
        atlas_path = op.join(atlas_dir, atlas_name)
    if not op.exists(atlas_path):
        # Fallback to local atlas
        fallback_atlas = op.join(op.dirname(atlas_path), "aVP-24_label.nii")
        if op.exists(fallback_atlas):
            atlas_path = fallback_atlas
            logger.warning(f"Using fallback atlas: {atlas_path}")
        else:
            logger.warning(f"Atlas not found at {atlas_path}, trying local atlas")
            atlas_path = "aVP-24_label.nii"
        
    if not op.exists(atlas_path):
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")
    
    atlas = ni.load(atlas_path)
    atlas_data = atlas.get_fdata()
    x_dim, y_dim, z_dim = atlas_data.shape
    n_slices = atlas.shape[1]
    
    bonferroni_value = p_value / n_slices
    logger.info(f"Using Bonferroni correction: {bonferroni_value}")

    groups = [dataset_a, dataset_b]

    # Read data for each group with error handling
    for group in groups:
        group_file = results_fname.format(group=group)
        if not op.exists(group_file):
            raise FileNotFoundError(f"Results file not found for group {group}: {group_file}")
        
        logger.info(f"Loading data for group: {group}")
        df = pd.read_excel(group_file)
        df['group'] = [group] * df.shape[0]
        dataframe.append(df)
    
    full_dataframe = pd.concat(dataframe, ignore_index=True)
    logger.info(f"Loaded data for {len(groups)} groups with {len(full_dataframe)} total samples")

    
    for feature in features:
        for side in sides:
            for group in groups:
            
                df = filter_dataframe(full_dataframe, 
                                      group=[group], 
                                      side=[side], 
                                      type=[image_type])        
                df = apply_function(df, keys=aggregation_keys, 
                                    attr=feature, fx=lambda x: x.mean(0))
                
                nerve_map = create_nerve_map(df, feature, atlas_path, background_threshold)
                
                map_fname = map_filename_template.format(group=group,
                                                        side=side, 
                                                        feature=feature)
                
                nerve_image = ni.Nifti1Image(nerve_map, atlas.affine)
                ni.save(nerve_image, op.join(path_map, map_fname))  
                
                if do_figures:
                    pl.figure()
                    pl.imshow(nerve_map[:,:,slice_display_index], cmap=pl.cm.magma)
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
                
                slice_filter = {slice_column: [y+1]}
                df_slice_a = filter_dataframe(df, group=[dataset_a], **slice_filter)
                df_slice_b = filter_dataframe(df, group=[dataset_b], **slice_filter)
                
                t, p = ttest_ind(df_slice_a[feature].values, 
                                 df_slice_b[feature].values)
                
                nerve_map_t[:, y, :][atlas_data[:, y, :] != background_threshold] = t
                nerve_map_p[:, y, :][atlas_data[:, y, :] != background_threshold] = p
                
                threshold_image = nerve_map_t * (nerve_map_p < bonferroni_value)
                
            
            map_fname = map_filename_template.format(group='t',
                                                     side=side, 
                                                     feature=feature)
            nerve_image = ni.Nifti1Image(nerve_map_t, atlas.affine)
            ni.save(nerve_image, op.join(path_map, map_fname))
            
            map_fname = map_filename_template.format(group='p',
                                                    side=side, 
                                                    feature=feature)
            nerve_image = ni.Nifti1Image(nerve_map_p, atlas.affine)
            ni.save(nerve_image, op.join(path_map, map_fname))
            
            if do_figures:
                pl.figure()
                pl.imshow(threshold_image[:, :, slice_display_index], 
                         cmap=pl.cm.coolwarm, 
                         vmin=stat_plot_vlim[0], vmax=stat_plot_vlim[1])
                pl.title(f"Side: {side} - Feature: {feature}")
                pl.colorbar()
                

    ###################################################################################
    # 2) Plot different values

    for f, feature in enumerate(features):
        for group in groups:
            df = filter_dataframe(full_dataframe, 
                                  group=[group], 
                                  side=[side], 
                                  type=[image_type])        
            df = apply_function(df, 
                                keys=aggregation_keys + ['subject_id'], 
                                attr=feature, 
                                fx=lambda x: x.mean(0))
            
            df = apply_function(df,
                                keys=aggregation_keys,
                                attr=feature,
                                fx=lambda x: x.mean(0))
            
            
            nerve_map = create_nerve_map(df, feature, atlas_path, background_threshold)
            
            if do_figures:
                colormap = feature_colormaps[f] if f < len(feature_colormaps) else pl.cm.magma
                vlim = feature_limits[f] if f < len(feature_limits) else None
                
                fig, ax = plot_nerve(nerve_map,
                                     threshold=0,
                                     comparison='equal',
                                     colormap=colormap,
                                     title=f"{feature} map in {group}",
                                     vlim=vlim,
                                     atlas_path=atlas_path,
                                     slice_index=slice_display_index)
            
                fig.savefig(
                    op.join(path_map, 
                            f"sub-group_feature-{feature}_group-{group}_side-both_on.{figure_extension}")
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
                                keys=aggregation_keys, 
                                attr=feature, 
                                fx=lambda x: x.mean(0))
            df_std = apply_function(df,
                                keys=aggregation_keys,
                                attr=feature,
                                fx=lambda x: x.std(0))
            
            
            if lap == 0:
                dfs = df_mean.copy()
            
            lap += 1
            
            dfs[f"{feature}_mean_{group}"] = df_mean[feature].values
            dfs[f"{feature}_std_{group}"] = df_std[feature].values
            
            

    ###################################################################################
    # 3) Plot different values and statistical comparisons

    for feature in features:
                
        df = filter_dataframe(full_dataframe, type=[image_type])        
        df = apply_function(df, 
                            keys=[slice_column, 'group', 'subject_id'], 
                            attr=feature, 
                            fx=lambda x: x.mean(0))
        
        nerve_map_t = np.zeros((atlas.shape[0], atlas.shape[1], atlas.shape[2]))
        nerve_map_p = np.zeros((atlas.shape[0], atlas.shape[1], atlas.shape[2]))
        
        ts = []
        ps = []
        
        for y in range(n_slices):
            
            slice_filter = {slice_column: [y+1]}
            df_slice_a = filter_dataframe(df, group=[dataset_a], **slice_filter)
            df_slice_b = filter_dataframe(df, group=[dataset_b], **slice_filter)
            
            t, p = ttest_ind(df_slice_a[feature].values, 
                             df_slice_b[feature].values)
            
            nerve_map_t[:, y, :][atlas_data[:, y, :] != background_threshold] = t
            nerve_map_p[:, y, :][atlas_data[:, y, :] != background_threshold] = p
            
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
                                vlim=stat_plot_vlim,
                                atlas_path=atlas_path,
                                slice_index=slice_display_index)
            
            fig.savefig(op.join(path_map, 
                                    f"sub-group_feature-{feature}_stats-ttestfdr_side-both_on.{figure_extension}"))
            
            
            fig, ax = plot_nerve(nerve_map_t,
                                threshold=0.,
                                comparison='equal',
                                colormap=pl.cm.coolwarm,
                                title=f"Unthresholded {feature} map in {dataset_a} vs {dataset_b}",
                                vlim=stat_plot_vlim,
                                atlas_path=atlas_path,
                                slice_index=slice_display_index)

            fig.savefig(op.join(path_map, 
                                    f"sub-group_feature-{feature}_stats-ttestuncorrected_side-both_on.{figure_extension}"))        
            
            

            fig, ax = plot_nerve(nerve_map_t * (nerve_map_p < bonferroni_value),
                                threshold=0.,
                                comparison='equal',
                                colormap=pl.cm.coolwarm,
                                title=f"Bonferroni-corrected {feature} map in {dataset_a} vs {dataset_b}",
                                vlim=stat_plot_vlim,
                                atlas_path=atlas_path,
                                slice_index=slice_display_index)
            
            fig.savefig(op.join(path_map,
                                    f"sub-group_feature-{feature}_stats-ttestbonferroni_side-both_on.{figure_extension}"))
        
    # Save results and return dataframe
    output_file = op.join(path_map, output_filename)
    dfs.to_excel(output_file)
    logger.info(f"Results saved to: {output_file}")
    
    return dfs


def generate_nerve_maps_with_config(path, dataset_a, dataset_b, config=None, debug=False):
    """
    Generate nerve maps using a configuration object.
    
    Args:
        path (str): Root path containing the data
        dataset_a (str): Name of first dataset group  
        dataset_b (str): Name of second dataset group
        config (StatsConfig): Configuration object (optional, uses defaults if None)
        debug (bool): Enable debug logging
        
    Returns:
        pd.DataFrame: Combined results dataframe with statistics
    """
    if config is None:
        config = StatsConfig()
    
    # Convert config to keyword arguments
    config_dict = config.to_dict()
    
    return generate_nerve_maps(
        path=path,
        dataset_a=dataset_a,
        dataset_b=dataset_b,
        debug=debug,
        **config_dict
    )


def main(path="./", dataset_a="HC", dataset_b="PTS", debug=False, config_file=None):
    """
    Main function for statistical analysis - compatible with CLI interface.
    
    Args:
        path (str): Root path containing the data
        dataset_a (str): Name of first dataset group
        dataset_b (str): Name of second dataset group  
        debug (bool): Enable debug logging
        config_file (str): Path to configuration file (JSON/YAML) - optional
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    
    logger.info("Starting aVP-Toolbox statistical analysis")
    logger.info(f"Atlas directory: {atlas_dir}")
    logger.info(f"Parent directory: {parent_dir}")
    
    try:
        # Load configuration if provided
        if config_file:
            logger.info(f"Loading configuration from: {config_file}")
            config = StatsConfig.from_file(config_file)
        else:
            # Use default configuration
            config = StatsConfig()
        
        results_df = generate_nerve_maps_with_config(
            path=path,
            dataset_a=dataset_a, 
            dataset_b=dataset_b,
            config=config,
            debug=debug
        )
        
        logger.info("Statistical analysis completed successfully")
        return results_df
        
    except Exception as e:
        logger.error(f"Error during statistical analysis: {e}")
        raise


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Statistical analysis for aVP-Toolbox")
    parser.add_argument("--path", type=str, default="./", help="Root path containing the data")
    parser.add_argument("--dataset-a", type=str, default="HC", help="Name of first dataset group")
    parser.add_argument("--dataset-b", type=str, default="PTS", help="Name of second dataset group")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--config", type=str, help="Path to configuration file (JSON/YAML)")
    parser.add_argument("--generate-config", type=str, help="Generate example configuration file at specified path")
    
    args = parser.parse_args()
    
    # Generate example configuration file if requested
    if args.generate_config:
        import json
        config = StatsConfig()
        config_dict = config.to_dict()
        
        with open(args.generate_config, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        print(f"Example configuration file generated at: {args.generate_config}")
        print("Edit this file to customize your statistical analysis parameters.")
    else:
        main(path=args.path, dataset_a=args.dataset_a, dataset_b=args.dataset_b, 
             debug=args.debug, config_file=args.config)