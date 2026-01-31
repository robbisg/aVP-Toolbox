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

plt.style.use('seaborn-v0_8')

fixed_segments = [
    ('iOrb', 0, 36),
    ('iCan', 37, 47),
    ('iCran', 48, 73),
    ('OC', 74, 84),
    ('OT', 85, 101)
]

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
    #plot_df = plot_df[plot_df['image_type'] == image_type]
    
    # TODO: A possible implementation is to extract the line from each subject and
    # produce a single plot and a total plot.
    
    mapping = {
        'linearize': 'linearized',
        'normalized': 'normalized',
        'l': 'left',
        'r': 'right',
        'area': 'Cross-Sectional Area (mm²)',
        'eccent': 'Eccentricity',
    }
    
    if len(plot_df) == 0:
        logger.warning(f"No data available for {feature}")
        return
    
    #####################################################################
    # Plot of all subjects with average line
    #####################################################################
    figure = sns.relplot(data=plot_df, x=x_axis, y=feature, hue=group_by, 
                         col='side', row='image_type', kind='line', alpha=0.3,
                         color='dimgray', height=4, aspect=2, legend=False)
        
    #plt.show()
   
    # Add average line to each subplot in the seaborn figure
    for i, ax in enumerate(figure.axes.flat):
        # Get data for this subplot
        subplot_data = ax.lines[0].get_data() if ax.lines else ([], [])
        if len(subplot_data[0]) == 0:
            continue
        
        # Calculate average across subjects for each x position
        x_vals = []
        y_vals = []
        for line in ax.lines:
            x, y = line.get_data()
            x_vals.extend(x)
            y_vals.extend(y)
        
        if x_vals:
            df_temp = pd.DataFrame({'x': x_vals, 'y': y_vals})
            avg = df_temp.groupby('x')['y'].mean().reset_index()
            ax.plot(avg['x'], avg['y'], linewidth=3, color='darkblue', label='Average')
            
        if 'normalized' in ax.get_title().lower():
            for segment_name, start_slice, end_slice in fixed_segments:
                boundary = end_slice + 0.5
                ax.axvline(x=boundary, color='gray', linewidth=.8, linestyle='--')
                text_x = (start_slice + end_slice) * .5
                
                line_max = avg['y'].values[start_slice:end_slice].max()
                text_y = line_max + (line_max * 0.1)
                
                ax.text(text_x-3, text_y, segment_name, color='darkblue', fontweight='bold', fontsize=14)
        
        # Change title font size
        ax.set_title(ax.get_title(), fontsize=16)
        ax.set_xlabel("y length (mm)", fontsize=14)
        ax.xaxis.set_major_formatter(lambda x, pos: f"{int(x*0.6):.1f}") # Example: 1 decimal place

        ax.set_ylabel(feature, fontsize=14)
        #ax.legend()

    fname = os.path.join(output_path, 'plots', f'sub-all_feature-{feature}_desc-allsubjects_plot.png')
    figure.savefig(fname, dpi=300)

    ########################################################################
    # Plot of individual subjects in separate plots
    ########################################################################
    subjects = plot_df[group_by].unique()
    logger.debug(f"Plotting {len(subjects)} subjects")
        
    for subject in subjects:
        
        figure_lr, ax_lr = plt.subplots(2, 2, figsize=(16, 8))
        figure_avg, ax_avg = plt.subplots(2, 1, figsize=(10, 6))

        for t, img_type in enumerate(['linearize', 'normalized']):
        
            to_be_averaged = []

            for s, side in enumerate(['l', 'r']):
                            
                data = plot_df[(plot_df[group_by] == subject) & 
                                (plot_df['side'] == side) & 
                                (plot_df['image_type'] == img_type)]
                if data.empty:
                    continue
                    
                x = data[x_axis]
                y = data[feature]

                to_be_averaged.append([x, y])
                
                ax_lr[t, s].plot(x, y, linewidth=1.5, color='salmon')
                ax_lr[t, s].set_title(f'{feature} in {subject} {img_type} image - {side}', fontsize=14)

                ax_lr[t, s].set_xlabel(x_axis, fontsize=12)
                ax_lr[t, s].set_ylabel(feature, fontsize=12)

                # Separate segments with vertical lines
                segment_names = data['segment_name'].unique()
                for i, segment in enumerate(segment_names):  # Skip first segment
                    boundary = data[data['segment_name'] == segment][x_axis].min()
                    ax_lr[t, s].axvline(x=boundary, color='gray', alpha=0.6, linewidth=.5, linestyle='--')
                    
                    # Add segment labels
                    line_max_segment = y[data['segment_name'] == segment].max()
                    text_x = boundary + (data[data['segment_name'] == segment][x_axis].max() - boundary) * 0.5
                    text_y = line_max_segment + (line_max_segment * 0.02)
                    ax_lr[t, s].text(text_x-3, text_y, segment, color='salmon', fontweight='bold', fontsize=14)

            
            min_elements = np.min([len(item[1]) for item in to_be_averaged])
            arg_min = np.argmin([len(item[1]) for item in to_be_averaged])
            x_avg = to_be_averaged[arg_min][0][:min_elements]
            y_avg = np.mean([item[1][:min_elements] for item in to_be_averaged], axis=0)

            ax_avg[t].plot(x_avg, y_avg, linewidth=1, color='salmon')
            ax_avg[t].set_title(f'{feature} in {subject} {img_type} image - Average Both Sides', fontsize=14)
            ax_avg[t].set_xlabel(x_axis, fontsize=12)
            ax_avg[t].set_ylabel(feature, fontsize=12)
        
            for i, segment in enumerate(segment_names):  # Skip first segment
                boundary = data[data['segment_name'] == segment][x_axis].min()
                ax_avg[t].axvline(x=boundary, color='gray', alpha=0.6, linewidth=.5, linestyle='--')
                
                # Add segment labels
                line_max_segment = y_avg[(x_avg >= boundary) & (x_avg <= data[data['segment_name'] == segment][x_axis].max())].max()
                text_x = boundary + (data[data['segment_name'] == segment][x_axis].max() - boundary) * 0.5
                text_y = line_max_segment + (line_max_segment * 0.02)
                ax_avg[t].text(text_x-3, text_y, segment, color='salmon', fontweight='bold', fontsize=14)
        
        figure_lr.tight_layout()
        figure_avg.tight_layout()
        
        # Save individual subject plots
        individual_plots_path = None
        if output_path:
            base_dir = os.path.dirname(output_path)
            subject_ = str(subject).zfill(2)
            individual_plots_path = os.path.join(base_dir, 'plots', f'{subject_}')
            os.makedirs(individual_plots_path, exist_ok=True)
       
            figure_lr.savefig(os.path.join(individual_plots_path, 
                            f"sub-{subject}_feature-{feature}_desc-singleside_plot.png"), 
                            dpi=300, bbox_inches='tight', facecolor='white')
            figure_avg.savefig(os.path.join(individual_plots_path, 
                            f"sub-{subject}_feature-{feature}_desc-avg_plot.png"), 
                            dpi=300, bbox_inches='tight', facecolor='white')
            logger.debug(f"Saved individual plots for subject {subject} in {individual_plots_path}")
        
        plt.close('all')        
    
    return

def main(path="./", features=['area'], debug=False, image_type='linearize'):
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
            create_feature_lineplot(
                df, feature, image_type=im_type,
                output_path=path
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