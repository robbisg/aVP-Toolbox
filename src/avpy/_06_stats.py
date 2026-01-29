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
# statistical comparisons between groups.

import pandas as pd
import nibabel as ni
import numpy as np
import os
import os.path as op
from sekupy.results import apply_function, filter_dataframe
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as pl
import pingouin as pg
import logging
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)
NAME = "stats"

# Initialize atlas paths
parent_dir = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
atlas_dir = op.join(parent_dir, "atlas")
atlas_name = "aVP-24_prob50.nii.gz"

percentage_threshold = 50

segments = [
    ('iOrb', 0, 36),
    ('iCan', 37, 47),
    ('iCran', 48, 73),
    ('OC', 74, 84),
    ('OT', 85, 101)
]

def create_nerve_map(dataframe, feature):
    
    background_image = ni.load(op.join(atlas_dir, atlas_name))
    atlas = background_image.get_fdata()[:, ::-1, :]
    n_slices = atlas.shape[1]
    
    nerve_map = np.zeros((atlas.shape[0],
                          atlas.shape[1], 
                          atlas.shape[2]))
    
    for y in range(n_slices):
        nerve_map[:, y, :][atlas[:, y, :] >= percentage_threshold] = dataframe[feature].values[y]
        
    return nerve_map



def plot_nerve(nerve_map, 
               threshold,
               comparison='equal', 
               colormap=pl.cm.magma,
               title="Nerve Map",
               vlim=None,
               figsize=(10, 16)
               ):
    
    background_image = ni.load(op.join(atlas_dir, atlas_name))
    background_data = background_image.get_fdata()[:, ::-1, :]
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
    
    for segment_name, start_slice, end_slice in segments:
        slice_position = (start_slice + end_slice) / 2
        y_pos = start_slice - .5
        ax.hlines(y=y_pos, xmin=0, xmax=x_dim, colors='white', linestyles='dashed', linewidth=1, zorder=50)
        ax.text(110, slice_position, segment_name, color='white', fontsize=15, zorder=100)
    

    # Set the ticks to be at the correct mm intervals
    x_ticks = np.arange(0, x_dim + 1, 25)
    y_ticks = np.arange(0, y_dim + 1, 25)[::-1]
        
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    ax.xaxis.set_major_formatter(lambda x, pos: f"{int(x*resolution):.1f}") # Example: 1 decimal place
    ax.yaxis.set_major_formatter(lambda x, pos: f"{int(x*resolution):.1f}")
    
    ax.set_xlim(100, 150)
    ax.grid(False)
    
    return fig, ax


def calculate_segment_statistics(full_dataframe, dataset_a, dataset_b, features, sides, image_type='normalized'):
    """
    Calculate statistical tests for each anatomical segment.
    
    Args:
        full_dataframe (pd.DataFrame): Combined dataframe with all data
        dataset_a (str): Name of first dataset group
        dataset_b (str): Name of second dataset group
        features (list): List of features to analyze
        sides (list): List of sides to analyze
        image_type (str): Type of image data to process
        
    Returns:
        pd.DataFrame: Statistics summary for each segment
    """
    logger.info("Calculating segment-based statistics...")
    
    segment_stats = []
    
    for feature in features:
        for side in sides:
            # Filter data for this feature and side
            df = filter_dataframe(full_dataframe, 
                                  side=[side], 
                                  image_type=[image_type])
            
            fsegments = np.unique(df['segment_name'].values)
            # Calculate statistics for each segment
            for segment_name in fsegments:
                # Filter data for this segment (slice range)
                df_segment = df[df['segment_name'] == segment_name]

                # Average across slices for each subject
                df_segment_avg = df_segment.groupby(['subject', 'group'])[feature].mean().reset_index()
                
                # Separate groups
                group_a = df_segment_avg[df_segment_avg['group'] == dataset_a][feature].values
                group_b = df_segment_avg[df_segment_avg['group'] == dataset_b][feature].values
                
                if len(group_a) == 0 or len(group_b) == 0:
                    logger.warning(f"Insufficient data for {segment_name}, {side}, {feature}")
                    continue
                
                # Perform t-test
                t_stat, p_value = ttest_ind(group_a, group_b)
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt((np.std(group_a, ddof=1)**2 + np.std(group_b, ddof=1)**2) / 2)
                cohens_d = (np.mean(group_a) - np.mean(group_b)) / pooled_std if pooled_std > 0 else 0
                
                # Store results
                segment_stats.append({
                    'segment': segment_name,
                    'side': side,
                    'feature': feature,
                    f'{dataset_a}_mean': np.mean(group_a),
                    f'{dataset_a}_std': np.std(group_a, ddof=1),
                    f'{dataset_a}_n': len(group_a),
                    f'{dataset_b}_mean': np.mean(group_b),
                    f'{dataset_b}_std': np.std(group_b, ddof=1),
                    f'{dataset_b}_n': len(group_b),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < 0.05
                })
    
    segment_stats_df = pd.DataFrame(segment_stats)
    
    logger.debug(f"Significant (uncorrected p<0.05): {segment_stats_df['significant'].sum()}")
    
    return segment_stats_df


def plot_segment_statistics(segment_stats_df, dataset_a, dataset_b, path_map):
    """
    Create visualizations for segment-based statistics.
    
    Args:
        segment_stats_df (pd.DataFrame): Segment statistics dataframe
        dataset_a (str): Name of first dataset group
        dataset_b (str): Name of second dataset group
        path_map (str): Output directory for figures
    """
    logger.info("Generating segment statistics plots...")
    
    features = segment_stats_df['feature'].unique()
    sides = segment_stats_df['side'].unique()
    
    for feature in features:
        # Create figure with subplots for each side
        fig, axes = pl.subplots(1, len(sides), figsize=(6*len(sides), 8))
        if len(sides) == 1:
            axes = [axes]
        
        for ax, side in zip(axes, sides):
            df_plot = segment_stats_df[
                (segment_stats_df['feature'] == feature) & 
                (segment_stats_df['side'] == side)
            ].copy()
            
            if len(df_plot) == 0:
                continue
            
            # Prepare data for plotting
            segments_plot = df_plot['segment'].values
            x_pos = np.arange(len(segments_plot))
            
            means_a = df_plot[f'{dataset_a}_mean'].values
            stds_a = df_plot[f'{dataset_a}_std'].values
            means_b = df_plot[f'{dataset_b}_mean'].values
            stds_b = df_plot[f'{dataset_b}_std'].values
            
            width = 0.35
            
            # Create bars
            bars1 = ax.bar(x_pos - width/2, means_a, width, 
                          yerr=stds_a, label=dataset_a, 
                          color='#e74c3c', alpha=0.8, capsize=5)
            bars2 = ax.bar(x_pos + width/2, means_b, width, 
                          yerr=stds_b, label=dataset_b, 
                          color='#3498db', alpha=0.8, capsize=5)
            
            # Add significance markers
            y_max = max(means_a.max() + stds_a.max(), means_b.max() + stds_b.max())
            for i, row in df_plot.iterrows():
                y_pos = np.max([row[f'{dataset_a}_mean'] + row[f'{dataset_a}_std'],
                                row[f'{dataset_b}_mean'] + row[f'{dataset_b}_std']])
                if row['p_value'] < 0.05:
                    # FDR significant
                    ax.text(x_pos[i % len(x_pos)], y_pos * 1.05, '*', 
                           ha='center', va='bottom', fontsize=18, fontweight='bold')
                elif row['p_value'] < 0.01:
                    # Bonferroni significant
                    ax.text(x_pos[i % len(x_pos)], y_pos * 1.05, '**', 
                           ha='center', va='bottom', fontsize=18, fontweight='bold')
                elif row['p_value'] < 0.005:
                    # Uncorrected significant
                    ax.text(x_pos[i % len(x_pos)], y_pos * 1.05, '***', 
                           ha='center', va='bottom', fontsize=18, fontweight='bold')

            # Formatting
            ax.set_xlabel('Segment', fontsize=12)
            ax.set_ylabel(f'{feature.capitalize()}', fontsize=12)
            ax.set_title(f'Side: {side.upper()}', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(segments_plot, rotation=45, ha='right')
            
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, y_max * 1.15)
        ax.legend()
        
        fig.suptitle(f'{feature.capitalize()} - Segment Comparison: {dataset_a} vs {dataset_b}', 
                    fontsize=16, fontweight='bold', y=0.98)
        pl.tight_layout()
        
        # Save figure
        output_file = op.join(path_map, f'segment_stats_{feature}_{dataset_a}_vs_{dataset_b}.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        #logger.info(f"Saved segment statistics plot: {output_file}")
        
        pl.close(fig)



def calculate_segment_statistics_lm(full_dataframe, features, sides, 
                                    covariates=None, image_type='normalized', formula=None,
                                    correction_method='fdr_bh'):
    """
    Calculate statistical tests for each anatomical segment using linear models.
    
    Args:
        full_dataframe (pd.DataFrame): Combined dataframe with all data
        features (list): List of features to analyze
        sides (list): List of sides to analyze
        covariates (list): List of covariate column names (e.g., ['age', 'gender', 'group'])
        image_type (str): Type of image data to process
        formula (str): Custom formula for the linear model (e.g., 'feature ~ group + age + C(gender)')
                      If None, uses 'feature ~ group' or 'feature ~ group + covariates'
        correction_method (str): Multiple comparison correction method 
                                ('bonferroni', 'fdr_bh', 'fdr_by', 'holm', etc.)
        
    Returns:
        pd.DataFrame: Statistics summary for each segment including all effects
    """
    logger.info("Calculating segment-based statistics with linear models...")
    
    if covariates is None:
        covariates = []
    
    # Ensure 'group' is in covariates if not already
    if 'group' not in covariates:
        covariates = ['group'] + covariates
    
    segment_stats = []
    
    for feature in features:
        for side in sides:
            # Filter data for this feature and side
            df = filter_dataframe(full_dataframe, 
                                  side=[side], 
                                  image_type=[image_type])
            
            fsegments = np.unique(df['segment_name'].values)
            
            for segment_name in fsegments:
                # Filter data for this segment
                df_segment = df[df['segment_name'] == segment_name]

                # Average across slices for each subject
                groupby_cols = ['subject'] + covariates
                df_segment_avg = df_segment.groupby(groupby_cols)[feature].mean().reset_index()
                
                # Build formula for linear model
                if formula is None:
                    # Auto-generate formula
                    formula_str = f'{feature} ~ ' + ' + '.join(covariates)
                else:
                    # Use custom formula, replacing 'feature' placeholder
                    formula_str = formula.replace('feature', feature)
                
                try:
                    # Fit linear model
                    model = ols(formula_str, data=df_segment_avg).fit()
                    
                    # Store main results
                    result = {
                        'segment': segment_name,
                        'side': side,
                        'feature': feature,
                        'formula': formula_str,
                        'n_samples': len(df_segment_avg),
                        'r_squared': model.rsquared,
                        'adj_r_squared': model.rsquared_adj,
                        'f_statistic': model.fvalue,
                        'f_pvalue': model.f_pvalue
                    }
                    
                    # Add group means (for all groups)
                    groups = df_segment_avg['group'].unique()
                    for group in groups:
                        group_data = df_segment_avg[df_segment_avg['group'] == group][feature]
                        result[f'{group}_mean'] = group_data.mean()
                        result[f'{group}_std'] = group_data.std()
                        result[f'{group}_n'] = len(group_data)
                    
                    # Add all parameter estimates
                    for param_name in model.params.index:
                        result[f'{param_name}_coef'] = model.params[param_name]
                        result[f'{param_name}_se'] = model.bse[param_name]
                        result[f'{param_name}_t'] = model.tvalues[param_name]
                        result[f'{param_name}_p'] = model.pvalues[param_name]
                    
                    segment_stats.append(result)
                    
                except Exception as e:
                    logger.warning(f"Failed to fit model for {segment_name}, {side}, {feature}: {e}")
                    continue
    
    segment_stats_df = pd.DataFrame(segment_stats)
    
    if len(segment_stats_df) > 0:
        # Apply multiple comparison correction for each parameter
        param_cols = [col for col in segment_stats_df.columns if col.endswith('_p')]
        
        for p_col in param_cols:
            param_name = p_col.replace('_p', '')
            
            # Get p-values
            p_values = segment_stats_df[p_col].values
            
            # Apply correction
            reject, p_corrected, _, _ = multipletests(p_values, method=correction_method)
            
            # Add corrected values to dataframe
            segment_stats_df[f'{param_name}_p_corrected'] = p_corrected
            segment_stats_df[f'{param_name}_significant'] = reject
            segment_stats_df[f'{param_name}_significant_uncorrected'] = p_values < 0.05
        
        # Log significant effects for all parameters
        for col in segment_stats_df.columns:
            if col.endswith('_significant') and not col.endswith('_uncorrected'):
                param_name = col.replace('_significant', '')
                n_sig = segment_stats_df[col].sum()
                n_sig_uncorr = segment_stats_df[f'{param_name}_significant_uncorrected'].sum()
                logger.info(f"{param_name}: {n_sig_uncorr} uncorrected, {n_sig} corrected ({correction_method})")
    
    return segment_stats_df


def create_statistical_nerve_maps(segment_stats_df, param_name='group[T.PTS]', stat_type='coef'):
    """
    Create nerve maps from statistical results.
    
    Args:
        segment_stats_df (pd.DataFrame): Segment statistics dataframe
        param_name (str): Parameter name to visualize (e.g., 'group[T.PTS]', 'age')
        stat_type (str): Type of statistic ('coef', 'p', 'p_corrected', 't')
        
    Returns:
        dict: Dictionary of nerve maps for each feature and side
    """
    logger.info(f"Creating nerve maps for {param_name}_{stat_type}")
    
    nerve_maps = {}
    
    # Load atlas to get dimensions
    atlas_path = op.join(atlas_dir, atlas_name)
    atlas = ni.load(atlas_path)
    atlas_data = atlas.get_fdata()[:, ::-1, :]
    n_slices = atlas.shape[1]
    
    features = segment_stats_df['feature'].unique()
    sides = segment_stats_df['side'].unique()
    
    for feature in features:
        for side in sides:
            # Create empty nerve map
            nerve_map = np.zeros((atlas_data.shape[0], atlas_data.shape[1], atlas_data.shape[2]))
            
            # Filter data for this feature and side
            df_plot = segment_stats_df[
                (segment_stats_df['feature'] == feature) & 
                (segment_stats_df['side'] == side)
            ].copy()
            
            if len(df_plot) == 0:
                continue
            
            # Check if parameter exists
            col_name = f'{param_name}_{stat_type}'
            if col_name not in df_plot.columns:
                logger.warning(f"Column {col_name} not found in results")
                continue
            
            # Fill nerve map by segment
            for _, row in df_plot.iterrows():
                segment_name = row['segment']
                value = row[col_name]
                
                # Find segment slice range
                segment_info = next((s for s in segments if s[0] == segment_name), None)
                if segment_info is None:
                    continue
                
                _, start_slice, end_slice = segment_info
                
                # Fill slices for this segment
                for y in range(start_slice, end_slice + 1):
                    if y < n_slices:
                        nerve_map[:, y, :][atlas_data[:, y, :] >= percentage_threshold] = value
            
            nerve_maps[f'{feature}_{side}'] = nerve_map
    
    return nerve_maps


def plot_nerve_maps_with_stats(nerve_maps, param_name, stat_type, path_map):
    """
    Plot nerve maps for statistical results.
    
    Args:
        nerve_maps (dict): Dictionary of nerve maps
        param_name (str): Parameter name being visualized
        stat_type (str): Type of statistic being visualized
        path_map (str): Output directory for figures
    """
    logger.info(f"Plotting nerve maps for {param_name}_{stat_type}")
    
    for map_name, nerve_map in nerve_maps.items():
        feature, side = map_name.split('_')
        
        # Determine colormap and threshold based on stat type
        if stat_type in ['p', 'p_corrected']:
            # For p-values, show significant regions
            threshold = 0.05
            comparison = 'less'
            colormap = pl.cm.hot_r
            title = f'{feature.capitalize()} - {param_name}\n{stat_type} (p < 0.05)'
            vlim = (0, 0.1)
        elif stat_type == 'coef':
            # For coefficients, show all values
            threshold = -np.inf
            comparison = 'greater'
            colormap = pl.cm.RdBu_r
            title = f'{feature.capitalize()} - {param_name}\nCoefficient'
            # Center colormap on 0
            max_abs = np.max(np.abs(nerve_map[nerve_map != 0]))
            vlim = (-max_abs, max_abs) if max_abs > 0 else None
        elif stat_type == 't':
            # For t-statistics, show all values
            threshold = -np.inf
            comparison = 'greater'
            colormap = pl.cm.RdBu_r
            title = f'{feature.capitalize()} - {param_name}\nT-statistic'
            max_abs = np.max(np.abs(nerve_map[nerve_map != 0]))
            vlim = (-max_abs, max_abs) if max_abs > 0 else None
        else:
            threshold = -np.inf
            comparison = 'greater'
            colormap = pl.cm.viridis
            title = f'{feature.capitalize()} - {param_name}\n{stat_type}'
            vlim = None
        
        # Plot nerve map
        fig, ax = plot_nerve(nerve_map, 
                            threshold=threshold,
                            comparison=comparison,
                            colormap=colormap,
                            title=title,
                            vlim=vlim)
        
        # Save figure
        output_file = op.join(path_map, f'nerve_map_{feature}_{side}_{param_name}_{stat_type}.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved nerve map: {output_file}")
        pl.close(fig)


def plot_segment_statistics_lm(segment_stats_df, path_map, contrasts=None):
    """
    Create visualizations for segment-based statistics with all effects.
    
    Args:
        segment_stats_df (pd.DataFrame): Segment statistics dataframe
        path_map (str): Output directory for figures
        contrasts (dict): Dictionary of contrasts to plot, e.g.,
                         {'HC vs PTS': 'group[T.PTS]', 'age_effect': 'age'}
                         If None, plots all available parameters
    """
    logger.info("Generating segment statistics plots with linear model effects...")
    
    features = segment_stats_df['feature'].unique()
    sides = segment_stats_df['side'].unique()
    
    # Get all parameter columns (those ending in _coef)
    param_cols = [col.replace('_coef', '') for col in segment_stats_df.columns 
                  if col.endswith('_coef') and col != 'Intercept_coef']
    
    if contrasts is None:
        # Plot all parameters
        contrasts = {param: param for param in param_cols}
    
    for feature in features:
        for contrast_name, param_name in contrasts.items():
            # Check if this parameter exists in the data
            if f'{param_name}_coef' not in segment_stats_df.columns:
                logger.warning(f"Parameter {param_name} not found in model results")
                continue
            
            fig, axes = pl.subplots(1, len(sides), figsize=(6*len(sides), 8))
            if len(sides) == 1:
                axes = [axes]
            
            for ax, side in zip(axes, sides):
                df_plot = segment_stats_df[
                    (segment_stats_df['feature'] == feature) & 
                    (segment_stats_df['side'] == side)
                ].copy()
                
                if len(df_plot) == 0:
                    continue
                
                segments_plot = df_plot['segment'].values
                x_pos = np.arange(len(segments_plot))
                
                coefs = df_plot[f'{param_name}_coef'].values
                ses = df_plot[f'{param_name}_se'].values
                
                # Color bars based on significance (corrected)
                colors = ['#2ecc71' if sig else '#95a5a6' 
                         for sig in df_plot[f'{param_name}_significant'].values]
                
                bars = ax.bar(x_pos, coefs, yerr=ses, 
                             color=colors, alpha=0.8, capsize=5)
                
                # Add significance markers (using corrected p-values)
                for i, (idx, row) in enumerate(df_plot.iterrows()):
                    p_corr = row[f'{param_name}_p_corrected']
                    
                    if p_corr < 0.001:
                        marker = '***'
                    elif p_corr < 0.01:
                        marker = '**'
                    elif p_corr < 0.05:
                        marker = '*'
                    else:
                        marker = ''
                    
                    if marker:
                        y_pos = row[f'{param_name}_coef'] + row[f'{param_name}_se']
                        ax.text(x_pos[i], y_pos * 1.05 if y_pos > 0 else y_pos * 0.95, marker,
                               ha='center', va='bottom' if y_pos > 0 else 'top',
                               fontsize=18, fontweight='bold', color='black')
                
                ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
                ax.set_xlabel('Segment', fontsize=12)
                ax.set_ylabel(f'{contrast_name} coefficient', fontsize=12)
                ax.set_title(f'Side: {side.upper()}', fontsize=14, fontweight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(segments_plot, rotation=45, ha='right')
                ax.grid(axis='y', alpha=0.3)
            
            fig.suptitle(f'{feature.capitalize()} - {contrast_name} Effect (corrected)', 
                        fontsize=16, fontweight='bold', y=0.98)
            pl.tight_layout()
            
            output_file = op.join(path_map, f'segment_stats_lm_{feature}_{param_name}_effect.png')
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            pl.close(fig)


def generate_nerve_maps(path, features=None, sides=None, groups=None,
                       image_type='normalized', p_value=0.05, 
                       generate_figures=False, debug=False, 
                       use_linear_model=False, covariates=None, formula=None,
                       contrasts=None, correction_method='fdr_bh'):
    """
    Generate nerve maps and perform statistical analysis.
    
    Args:
        path (str): Root path containing the data
        features (list): List of features to analyze (default: ['eccent', 'area'])
        sides (list): List of sides to analyze (default: ['r', 'l'])
        groups (list): List of group names to load (e.g., ['HC', 'PTS', 'MS'])
        image_type (str): Type of image data to process (default: 'normalized')
        p_value (float): P-value threshold for statistical tests (default: 0.05)
        generate_figures (bool): Whether to generate and save figures
        debug (bool): Enable debug logging
        use_linear_model (bool): Use linear models instead of t-tests
        covariates (list): List of covariate column names (e.g., ['age', 'gender'])
                          Note: 'group' will be added automatically if not present
        formula (str): Custom formula for linear model (e.g., 'feature ~ C(group) + age + age:C(group)')
        contrasts (dict): Dictionary of contrasts to plot (only for linear models)
        correction_method (str): Multiple comparison correction method ('fdr_bh', 'bonferroni', etc.)
        
    Returns:
        tuple: (slice_results_df, segment_stats_df) - Combined results dataframes
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    
    if groups is None:
        raise ValueError("groups parameter is required")
    
    logger.info(f"Starting statistical analysis with groups: {groups}")
    if use_linear_model:
        logger.info(f"Using linear models with formula: {formula if formula else 'auto-generated'}")
        logger.info(f"Covariates: {covariates}")
        logger.info(f"Multiple comparison correction: {correction_method}")
    
    # Initialize empty dataframe list
    dataframe = []
    do_figures = generate_figures
    
    # Create output directory
    group_str = '-'.join(groups)
    path_map = op.join(path, f"maps_{group_str}")
    os.makedirs(path_map, exist_ok=True)
    
    results_fname = op.join(path, "{group}", "results", "CSA_slice_iso.xlsx")
    
    # Load atlas
    atlas_path = op.join(atlas_dir, atlas_name)
    if not op.exists(atlas_path):
        atlas_path = "aVP-24_prob50.nii"
        logger.warning(f"Atlas not found at {atlas_path}, using fallback")
        
    if not op.exists(atlas_path):
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")
    
    atlas = ni.load(atlas_path)
    atlas_data = atlas.get_fdata()
    n_slices = atlas.shape[1]

    # Read data for each group
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
    
    # Calculate statistics
    if use_linear_model:
        segment_stats_df = calculate_segment_statistics_lm(
            full_dataframe, features, sides, covariates, image_type, formula, correction_method
        )
        
        # Save segment statistics
        segment_stats_file = op.join(path_map, f"segment_statistics_lm_{group_str}.xlsx")
        segment_stats_df.to_excel(segment_stats_file, index=False)
        logger.info(f"Saved segment statistics: {segment_stats_file}")
        
        # Generate plots
        if do_figures:
            plot_segment_statistics_lm(segment_stats_df, path_map, contrasts)
            
            # Generate nerve maps for each contrast
            if contrasts:
                for contrast_name, param_name in contrasts.items():
                    # Create nerve maps for coefficients
                    nerve_maps_coef = create_statistical_nerve_maps(
                        segment_stats_df, param_name, 'coef'
                    )
                    plot_nerve_maps_with_stats(
                        nerve_maps_coef, param_name, 'coef', path_map
                    )
                    
                    # Create nerve maps for corrected p-values
                    nerve_maps_p = create_statistical_nerve_maps(
                        segment_stats_df, param_name, 'p_corrected'
                    )
                    plot_nerve_maps_with_stats(
                        nerve_maps_p, param_name, 'p_corrected', path_map
                    )
    else:
        # For t-tests, need exactly 2 groups
        if len(groups) != 2:
            raise ValueError("t-test mode requires exactly 2 groups. Use --use-lm for more than 2 groups.")
        
        segment_stats_df = calculate_segment_statistics(
            full_dataframe, groups[0], groups[1], features, sides, image_type
        )
        
        segment_stats_file = op.join(path_map, f"segment_statistics_{groups[0]}_vs_{groups[1]}.xlsx")
        segment_stats_df.to_excel(segment_stats_file, index=False)
        
        if do_figures:
            plot_segment_statistics(segment_stats_df, groups[0], groups[1], path_map)
    
    return None, segment_stats_df


def main(path="./", groups=None, debug=False, use_linear_model=False, 
         covariates=None, formula=None, contrasts=None, correction_method='fdr_bh'):
    """
    Main function for statistical analysis.
    
    Args:
        path (str): Root path containing the data
        groups (list): List of group names (e.g., ['HC', 'PTS'])
        debug (bool): Enable debug logging
        use_linear_model (bool): Use linear models
        covariates (list): List of covariate column names
        formula (str): Custom formula for linear model
        contrasts (dict): Dictionary of contrasts to plot
        correction_method (str): Multiple comparison correction method
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    
    if groups is None or len(groups) < 2:
        raise ValueError("At least 2 groups are required")
    
    logger.info("Starting aVP-Toolbox statistical analysis")
    
    try:
        _, segment_stats_df = generate_nerve_maps(
            path=path,
            groups=groups,
            sides=['r', 'l'],
            features=['eccent', 'area'],
            image_type='normalized',
            p_value=0.05,
            generate_figures=True,
            debug=debug,
            use_linear_model=use_linear_model,
            covariates=covariates,
            formula=formula,
            contrasts=contrasts,
            correction_method=correction_method
        )
        
        logger.info("Statistical analysis completed successfully")
        return segment_stats_df
        
    except Exception as e:
        logger.error(f"Error during statistical analysis: {e}")
        raise


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Statistical analysis for aVP-Toolbox")
    parser.add_argument("--path", type=str, default="./", help="Root path containing the data")
    parser.add_argument("--groups", type=str, nargs='+', required=True, 
                       help="List of group names (e.g., HC PTS MS)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--use-lm", action="store_true", help="Use linear models")
    parser.add_argument("--covariates", type=str, nargs='+', 
                       help="List of covariate column names (e.g., age gender)")
    parser.add_argument("--formula", type=str, 
                       help="Custom formula for linear model (e.g., 'feature ~ C(group) + age')")
    parser.add_argument("--contrasts", type=str, nargs='+',
                       help="Contrasts to plot in format 'name:param' (e.g., 'HC_vs_PTS:group[T.PTS]')")
    parser.add_argument("--correction", type=str, default='fdr_bh',
                       choices=['bonferroni', 'fdr_bh', 'fdr_by', 'holm', 'hommel'],
                       help="Multiple comparison correction method")
    
    args = parser.parse_args()
    
    # Parse contrasts
    contrasts_dict = None
    if args.contrasts:
        contrasts_dict = {}
        for contrast in args.contrasts:
            name, param = contrast.split(':')
            contrasts_dict[name] = param
    
    main(path=args.path, groups=args.groups, debug=args.debug, 
         use_linear_model=args.use_lm, covariates=args.covariates,
         formula=args.formula, contrasts=contrasts_dict,
         correction_method=args.correction)