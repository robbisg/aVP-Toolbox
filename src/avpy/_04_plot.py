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

Plot features from normalized aVP data.

Reads in:
- CSA_slice.xlsx file from results directory

Produces:
- Line plots of features grouped by subject with averages
- Plots saved as PNG files in results directory
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
NAME = "plot_features"

def create_feature_lineplot(df, feature, x_axis='current_slice_yz', group_by='subject', 
                            image_type='linearized', output_path=None):
    """
    Create line plot showing individual subjects (transparent) and average (thick) with segments separated.
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature (str): Feature column name to plot
        x_axis (str): X-axis column name
        group_by (str): Grouping column name (default: 'subject')
        output_path (str): Path to save the plot
    """
    logger.debug(f"Creating line plot for feature: {feature}")
    
    plot_df = df.copy()
    plot_df = plot_df[plot_df['image_type'] == image_type]
    
    if len(plot_df) == 0:
        logger.warning(f"No data available for {feature}")
        return
    
    
    # Plot single subject l/r lines
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot individual subjects as transparent lines
    subjects = plot_df[group_by].unique()
    logger.debug(f"Plotting {len(subjects)} subjects")
    
    for subject in subjects:
        subject_data = plot_df[plot_df[group_by] == subject]
        if len(subject_data) > 0:
            ax.plot(subject_data[x_axis], subject_data[feature], 
                   alpha=0.3, linewidth=1, color='gray')
    
    # Calculate and plot average line
    avg_data = plot_df.groupby('plot_x')[feature].agg(['mean', 'std']).reset_index()
    
    # Plot average line with seaborn
    ax.plot(avg_data['plot_x'], avg_data['mean'], 
           linewidth=4, color='darkblue', label='Average', 
           marker='o', markersize=6, markerfacecolor='white', 
           markeredgecolor='darkblue', markeredgewidth=2)
    
    
    
    
    # Average between sides for each subject and slice
    if 'side' in plot_df.columns:
        logger.debug("Averaging between left and right sides")
        plot_df = plot_df.groupby([group_by, x_axis])[feature].mean().reset_index()
    else:
        # If no side column, just ensure we have the right grouping
        plot_df = plot_df.groupby([group_by, x_axis])[feature].mean().reset_index()
    
    # Get segment information for x-axis separation
    segments = plot_df['segment_name'].unique()
    segments = [seg for seg in segments if pd.notna(seg)]  # Remove NaN segments
    segments = sorted(segments)  # Sort for consistent ordering
    logger.debug(f"Found segments: {segments}")
    
    # Create a continuous x-axis position for plotting
    plot_df_with_x = []
    segment_info = {}
    current_x = 0
    
    for i, segment in enumerate(segments):
        segment_data = plot_df[plot_df['segment_name'] == segment].copy()
        if len(segment_data) == 0:
            continue
            
        # Sort by original slice position
        segment_data = segment_data.sort_values(x_axis)
        unique_slices = sorted(segment_data[x_axis].unique())
        
        # Map original slice positions to continuous x positions
        slice_to_x = {slice_pos: current_x + j for j, slice_pos in enumerate(unique_slices)}
        segment_data['plot_x'] = segment_data[x_axis].map(slice_to_x)
        
        # Store segment info for later use
        segment_info[segment] = {
            'start': current_x,
            'end': current_x + len(unique_slices) - 1,
            'center': current_x + (len(unique_slices) - 1) / 2
        }
        
        plot_df_with_x.append(segment_data)
        current_x += len(unique_slices) + 3  # Add gap between segments
    
    if not plot_df_with_x:
        logger.warning(f"No valid segment data found for {feature}")
        return
        
    plot_df = pd.concat(plot_df_with_x, ignore_index=True)
    
    # Set up the plot with seaborn style
    sns.set_palette("husl")

    
    # Add vertical lines to separate segments
    segment_boundaries = []
    for i, segment in enumerate(segments[1:], 1):  # Skip first segment
        prev_segment = segments[i-1]
        boundary = (segment_info[prev_segment]['end'] + segment_info[segment]['start']) / 2
        segment_boundaries.append(boundary)
        ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.6, linewidth=2)
    
    # Customize x-axis with segment labels
    segment_centers = [info['center'] for info in segment_info.values()]
    ax.set_xticks(segment_centers)
    ax.set_xticklabels(segments, fontsize=12, fontweight='bold')
    ax.set_xlabel('Anatomical Segments', fontsize=14, fontweight='bold')
    
    # Add secondary x-axis showing slice numbers
    ax2 = ax.twiny()
    
    # Create slice position labels (show a few key positions)
    slice_positions = []
    slice_labels = []
    
    for segment in segments:
        if segment in segment_info:
            segment_data = plot_df[plot_df['segment_name'] == segment]
            if len(segment_data) > 0:
                # Show start, middle, and end slice for each segment
                start_x = segment_info[segment]['start']
                end_x = segment_info[segment]['end']
                mid_x = segment_info[segment]['center']
                
                start_slice = segment_data[segment_data['plot_x'] == start_x][x_axis].iloc[0] if len(segment_data[segment_data['plot_x'] == start_x]) > 0 else None
                end_slice = segment_data[segment_data['plot_x'] == end_x][x_axis].iloc[0] if len(segment_data[segment_data['plot_x'] == end_x]) > 0 else None
                
                if start_slice is not None:
                    slice_positions.append(start_x)
                    slice_labels.append(f'{int(start_slice)}')
                if end_slice is not None and end_x != start_x:
                    slice_positions.append(end_x)
                    slice_labels.append(f'{int(end_slice)}')
    
    ax2.set_xlim(ax.get_xlim())
    if slice_positions:
        ax2.set_xticks(slice_positions)
        ax2.set_xticklabels(slice_labels, fontsize=10, alpha=0.7)
        ax2.set_xlabel('Slice Position', fontsize=12, alpha=0.7)
    
    # Customize plot appearance
    ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    ax.set_title(f'{feature.replace("_", " ").title()} Along Optic Nerve Segments', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Customize legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate "Individual subjects" labels
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    ax.legend(unique_handles, unique_labels, loc='upper right', fontsize=12, 
             frameon=True, fancybox=True, shadow=True)
    
    # Add segment background colors for better visualization
    colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))
    for i, (segment, info) in enumerate(segment_info.items()):
        ax.axvspan(info['start'] - 1, info['end'] + 1, 
                  alpha=0.1, color=colors[i], zorder=0)
        
        # Add segment labels at the top
        ax.text(info['center'], ax.get_ylim()[1] * 0.95, segment, 
               ha='center', va='top', fontweight='bold', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7))
    
    plt.tight_layout()
    
    # Save plot if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Plot saved to: {output_path}")
    
    return fig

def main(path="./", features=['area', 'eccent'], debug=False, image_type='linearize'):
    """
    Main function to generate feature plots from CSA_slice.xlsx
    
    Args:
        path (str): Study path containing results directory
        features (list): List of features to plot (default: ['area', 'eccent'])
        debug (bool): Enable debug logging
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    
    logger.info("Starting feature plotting")
    
    # image_type can be 'linearized' or 'normalized' or both
    if image_type in ['linearize', 'normalized']:
        image_type = [image_type]
    elif image_type == 'both':
        image_type = ['linearize', 'normalized']
    else:
        logger.warning(f"Unknown image_type '{image_type}', defaulting to 'linearize'")
        image_type = ['linearize']
    
    # Set up paths
    StudyPath = path
    results_path = os.path.join(StudyPath, 'results')
    data_file = os.path.join(results_path, 'CSA_slice_iso.xlsx')
    plots_path = os.path.join(StudyPath, 'plots')
    
    # Check if data file exists
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        logger.error("Please run avpy to generate CSA_slice.xlsx")
        return
    
    # Load data
    logger.info(f"Loading data from: {data_file}")
    
    df = pd.read_excel(data_file)
    
    logger.info(f"Loaded {len(df)} rows of data")
    logger.debug(f"Available columns: {df.columns.tolist()}")
    logger.debug(f"Available features: {[col for col in df.columns if col in ['area', 'eccent', 'majaxis', 'minaxis']]}")
    logger.debug(f"Subjects: {df['subject'].unique()}")
    logger.debug(f"Sides: {df['side'].unique() if 'side' in df.columns else 'No side column'}")
    logger.debug(f"Segments: {df['segment_name'].unique() if 'segment_name' in df.columns else 'No segment_name column'}")
    
    # Create plots directory
    os.makedirs(plots_path, exist_ok=True)
    
    # Generate plots for each feature
    for feature in features:
        if feature not in df.columns:
            logger.warning(f"Feature '{feature}' not found in data. Available features: {df.columns.tolist()}")
            continue
        
        for im_type in image_type:
        
            logger.info(f"Generating plot for feature: {feature}")
            
            # Single clean plot with segments separated
            output_file = os.path.join(plots_path, f'{feature}_segments.png')
            create_feature_lineplot(
                df, feature, image_type=im_type,
                output_path=output_file
            )
        
    logger.info(f"Feature plotting completed. Plots saved in: {plots_path}")
    
    return None

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        study_path = sys.argv[1]
    else:
        study_path = "./"
    
    debug_mode = '--debug' in sys.argv
    
    # Run main function
    stats = main(path=study_path, debug=debug_mode)
    
    if stats:
        print("\nFeature Summary Statistics:")
        for feature, values in stats.items():
            print(f"{feature}: mean={values['mean']:.3f}, std={values['std']:.3f}")