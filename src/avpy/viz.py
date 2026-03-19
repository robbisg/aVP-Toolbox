import os.path as op
import numpy as np
import nibabel as ni
import matplotlib.pyplot as pl
import pandas as pd
import logging

logger = logging.getLogger(__name__)

parent_dir = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
atlas_dir = op.join(parent_dir, "atlas")
atlas_name = "aVP-24_prob50.nii.gz"

segments = [
    ('iOrb', 0, 36),
    ('iCan', 37, 47),
    ('iCran', 48, 73),
    ('OC', 74, 84),
    ('OT', 85, 101)
]


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
              aspect='equal',
              )

    
    fx_comparison_dict = {
        'equal': np.equal,
        'greater': np.less,
        'less': np.greater
    }
    
    fx_comparison = fx_comparison_dict.get(comparison, np.equal)       
    
    if vlim is not None:
        vmin, vmax = vlim
    else:
        max_absolute = max(abs(nerve_map.min()), abs(nerve_map.max()))
        vmin, vmax = -max_absolute, max_absolute
    
    # Masking
    logger.debug(f"Map limits {vmin},{vmax}; Threshold {threshold}")
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
                          yerr=stds_a, label=str(dataset_a), 
                          color='#e74c3c', alpha=0.8, capsize=5)
            bars2 = ax.bar(x_pos + width/2, means_b, width, 
                          yerr=stds_b, label=str(dataset_b), 
                          color='#3498db', alpha=0.8, capsize=5)
            
            # Add significance markers
            y_max = max(means_a.max() + stds_a.max(), means_b.max() + stds_b.max())
            for i, row in df_plot.iterrows():
                y_pos = np.max([row[f'{dataset_a}_mean'] + row[f'{dataset_a}_std'],
                                row[f'{dataset_b}_mean'] + row[f'{dataset_b}_std']])
                
                if row['group_p_corrected'] < 0.05:
                    # FDR significant
                    ax.text(x_pos[i % len(x_pos)], y_pos * 1.05, '*', 
                           ha='center', va='bottom', fontsize=18, fontweight='bold')
                elif row['group_p_corrected'] < 0.01:
                    # Bonferroni significant
                    ax.text(x_pos[i % len(x_pos)], y_pos * 1.05, '**', 
                           ha='center', va='bottom', fontsize=18, fontweight='bold')
                elif row['group_p_corrected'] < 0.005:
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


def plot_nerve_maps_with_stats(nerve_map, param_name, key, stat_type, path_map):
    """
    Plot nerve maps for statistical results.
    
    Args:
        nerve_maps (dict): Dictionary of nerve maps
        param_name (str): Parameter name being visualized
        stat_type (str): Type of statistic being visualized
        path_map (str): Output directory for figures
    """
    logger.info(f"Plotting nerve maps for {param_name}_{stat_type}")
    
    
    colormaps = {
        'group': pl.cm.coolwarm,
        'age': pl.cm.RdGy,
        'sex': pl.cm.Spectral,
    }
    
    feature, side = key.split('_')
    
    if stat_type == 'p_uncorrected':
        title = f'{feature.capitalize()} - {param_name}\nUncorrected p-value'
    elif stat_type == 'p_corrected':
        title = f'{feature.capitalize()} - {param_name}\nCorrected p-value'
    elif stat_type == 'coef':
        title = f'{feature.capitalize()} - {param_name}\nCoefficient'
    else:
        logger.warning(f"Unknown stat_type: {stat_type}. Skipping.")
        return

    
    colormap = colormaps.get(param_name, pl.cm.seismic)
    
    # Plot nerve map
    fig, ax = plot_nerve(nerve_map, 
                        threshold=0,
                        comparison='equal',
                        colormap=colormap,
                        title=title)
    
    # Save figure
    output_file = op.join(path_map, f'nerve_map_{feature}_{side}_{param_name}_{stat_type}.png')
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.debug(f"Saved nerve map: {output_file}")
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
                colors = []
                for i, sig in enumerate(df_plot[f'{param_name}_significant'].values):
                    if not sig:
                        colors.append('#95a5a6')  # Gray for non-significant
                    else:
                        if coefs[i] > 0:
                            colors.append('royalblue')  # Green for positive significant
                        else:
                            colors.append('indianred')  # Red for negative significant
                
                
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
                        y_pos = np.abs(row[f'{param_name}_coef']) + 1.3 * row[f'{param_name}_se']
                        y_pos = np.sign(row[f'{param_name}_coef']) * y_pos                        
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
            
            

