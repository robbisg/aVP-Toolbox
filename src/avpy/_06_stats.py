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
from avpy.viz import plot_segment_statistics, plot_segment_statistics_lm, plot_nerve_maps_with_stats

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


def load_participants(path):
    participant_file = op.join(path, "participants.xlsx")
    logger.info(f"Loading participant data from: {participant_file}")
    participant_df = pd.read_excel(participant_file)
    participant_df['subject'] = participant_df['subject'].astype(str)
    return participant_df


def load_data(path, participant_df):
    results_fname = op.join(path, "results", "CSA_slice_iso.xlsx")
    logger.info(f"Loading results from: {results_fname}")
    dataframe = pd.read_excel(results_fname)
    
    # Join dataframe by id
    full_dataframe = pd.merge(dataframe, participant_df, on='subject', how='inner')
    
    missing_subjects = set(dataframe['subject'].unique()) - set(full_dataframe['subject'].unique())
    if missing_subjects:
        logger.warning(f"Missing participant data for subjects: {missing_subjects}")
        
        # Remove subjects without participant data
        full_dataframe = full_dataframe[full_dataframe['group'].isna() == False]
    
    return full_dataframe




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


def calculate_statistics_lm(full_dataframe, features, sides, kind='segment',
                            covariates=None, image_type='normalized', formula=None,
                            correction_method='fdr_bh'):
    
    logger.info(f"Calculating {kind}-based statistics with linear models...")
    
    # This should thorow and error?
    if covariates is None:
        covariates = []
    
    stats = []
    
    
    if kind == 'segment':
        grouping_var = 'segment_name'
    elif kind == 'slice':
        grouping_var = 'current_slice_yz'
    else:
        kind = 'segment'
        logger.warning(f"Invalid kind '{kind}' specified. Defaulting to 'segment'.")
    
    
    for feature in features:
        for side in sides:
            df = filter_dataframe(full_dataframe, 
                                  side=[side], 
                                  image_type=[image_type])
            
            unique_portions = np.unique(df[grouping_var].values)
            
            for portion_name in unique_portions:
                df_portion = df[df[grouping_var] == portion_name]

                
                if kind == 'segment':
                    groupby_cols = ['subject'] + covariates
                    df_portion = df_portion.groupby(groupby_cols)[feature].mean().reset_index()
                
                if formula is None:
                    formula_str = f'{feature} ~ ' + ' + '.join(covariates)
                else:
                    formula_str = formula.replace('feature', feature)
                
                try:
                    model = ols(formula_str, data=df_portion).fit()
                    
                    result = {
                        kind : portion_name,
                        'side': side,
                        'feature': feature,
                        'n_samples': len(df_portion),
                        'r_squared': model.rsquared,
                        'adj_r_squared': model.rsquared_adj
                    }
                    
                    if 'group' in df_portion.columns:
                        groups = df_portion['group'].unique()
                        for group in groups:
                            group_data = df_portion[df_portion['group'] == group][feature]
                            result[f'{group}_mean'] = group_data.mean()
                            result[f'{group}_std'] = group_data.std()
                            result[f'{group}_n'] = len(group_data)
                    
                    for param_name in model.params.index:
                        result[f'{param_name}_coef'] = model.params[param_name]
                        result[f'{param_name}_se'] = model.bse[param_name]
                        result[f'{param_name}_t'] = model.tvalues[param_name]
                        result[f'{param_name}_p'] = model.pvalues[param_name]
                    
                    stats.append(result)
                    
                except Exception as e:
                    logger.warning(f"Failed to fit model for {portion_name}, {side}, {feature}: {e}")
                    continue
    
    stats_df = pd.DataFrame(stats)
    
    if len(stats_df) > 0:
        param_cols = [col for col in stats_df.columns if col.endswith('_p')]
        
        for p_col in param_cols:
            param_name = p_col.replace('_p', '')
            p_values = stats_df[p_col].values
            reject, p_corrected, _, _ = multipletests(p_values, method=correction_method)
            
            stats_df[f'{param_name}_p_corrected'] = p_corrected
            stats_df[f'{param_name}_significant'] = reject
        
        for col in stats_df.columns:
            if col.endswith('_significant'):
                param_name = col.replace('_significant', '')
                n_sig = stats_df[col].sum()
                logger.info(f"{param_name}: {n_sig} significant ({correction_method})")
    
    return stats_df



def create_statistical_nerve_maps(segment_stats_df, param_name='group[T.PTS]', stat_type='coef'):
    
    logger.info(f"Creating nerve maps for {param_name}_{stat_type}")
    
    nerve_maps = {}
    
    atlas_path = op.join(atlas_dir, atlas_name)
    atlas = ni.load(atlas_path)
    atlas_data = atlas.get_fdata()[:, ::-1, :]
    n_slices = atlas.shape[1]
    
    features = segment_stats_df['feature'].unique()
    sides = segment_stats_df['side'].unique()
    
    for feature in features:
        for side in sides:
            nerve_map = np.zeros((atlas_data.shape[0], atlas_data.shape[1], atlas_data.shape[2]))
            
            df_plot = segment_stats_df[
                (segment_stats_df['feature'] == feature) & 
                (segment_stats_df['side'] == side)
            ].copy()
            
            if len(df_plot) == 0:
                continue
            
            col_name = f'{param_name}_{stat_type}'
            if col_name not in df_plot.columns:
                logger.warning(f"Column {col_name} not found in results")
                continue
            
            for _, row in df_plot.iterrows():
                segment_name = row['segment']
                value = row[col_name]
                
                segment_info = next((s for s in segments if s[0] == segment_name), None)
                if segment_info is None:
                    continue
                
                _, start_slice, end_slice = segment_info
                
                for y in range(start_slice, end_slice + 1):
                    if y < n_slices:
                        nerve_map[:, y, :][atlas_data[:, y, :] >= percentage_threshold] = value
            
            nerve_maps[f'{feature}_{side}'] = nerve_map
    
    return nerve_maps


def create_slice_nerve_maps(slice_stats_df, param_name, stat_type='coef'):
    
    nerve_maps = {}
    
    atlas_path = op.join(atlas_dir, atlas_name)
    atlas = ni.load(atlas_path)
    atlas_data = atlas.get_fdata()[:, ::-1, :]
    
    features = slice_stats_df['feature'].unique()
    sides = slice_stats_df['side'].unique()
    
    for feature in features:
        for side in sides:
            nerve_map = np.zeros(atlas_data.shape)
            
            df_plot = slice_stats_df[
                (slice_stats_df['feature'] == feature) & 
                (slice_stats_df['side'] == side)
            ].copy()
            
            if len(df_plot) == 0:
                continue
            
            col_name = f'{param_name}_{stat_type}'
            if col_name not in df_plot.columns:
                logger.warning(f"Column {col_name} not found")
                continue
            
            for _, row in df_plot.iterrows():
                y = int(row['slice'])
                value = row[col_name]
                if y < atlas_data.shape[1]:
                    nerve_map[:, y, :][atlas_data[:, y, :] >= percentage_threshold] = value
            
            nerve_maps[f'{feature}_{side}'] = nerve_map
    
    return nerve_maps


def generate_statistics(dataframe, features, sides, results_path, covariates=None, image_type='normalized',
                        formula=None, correction_method='fdr_bh', debug=False,
                        kind='segment', p_value=0.05, use_linear_model=True, 
                        do_figures=True):
    
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    
    logger.info(f"Starting analysis with groups: {groups}")
    if use_linear_model:
        logger.info(f"Formula: {formula if formula else 'auto-generated'}")
        logger.info(f"Covariates: {covariates if covariates else 'all'}")
        logger.info(f"Correction: {correction_method}")
    
    
    # Get all parameter columns (exclude Intercept)
    param_cols = [col.replace('_coef', '') for col in stats_df.columns 
                  if col.endswith('_coef') and col != 'Intercept_coef']
    
    
    if use_linear_model:
        
        stats_df = calculate_statistics_lm(
            dataframe, features, sides, covariates, image_type, formula, correction_method, 
            kind=kind
        )
        
    else:
        
        if 'group' not in dataframe.columns:
            raise ValueError("Dataframe must contain 'group' column for t-test mode. Use --use-lm for linear models.")
                
        groups = dataframe['group'].unique()
        if len(groups) != 2:
            raise ValueError("t-test mode requires exactly 2 groups. Use --use-lm for more.")
        
        segment_stats_df = calculate_segment_statistics(
            dataframe, groups[0], groups[1], features, sides, image_type
        )
        
        segment_stats_file = op.join(results_path, f"segment_statistics_{groups[0]}_vs_{groups[1]}.csv")
        segment_stats_df.to_csv(segment_stats_file, index=False)
        
        if do_figures:
            plot_segment_statistics(segment_stats_df, groups[0], groups[1], results_path)
            
        return
    

    
    if kind == 'segment':
        plot_segment_statistics_lm(stats_df, results_path)

        if 'group' in param_cols:
            groups = dataframe['group'].unique()
            if len(groups) == 2:
                plot_segment_statistics(stats_df, groups[0], groups[1], results_path)
        
    else:
        # Generate nerve maps for each parameter
        for param_name in param_cols:
            # Slice-wise nerve maps
            nerve_maps_coef = create_slice_nerve_maps(stats_df, param_name, 'coef')
            nerve_maps_p = create_slice_nerve_maps(stats_df, param_name, 'p_corrected')
            nerve_maps_punc = create_slice_nerve_maps(stats_df, param_name, 'p')
            
            for key, nerve_data_coef in nerve_maps_coef.items():
                nerve_data_p = nerve_maps_p[key]
                nerve_data_punc = nerve_maps_punc[key]
                                
                nerve_thresholded_p = nerve_data_coef * (nerve_data_p <= p_value)
                nerve_thresholded_punc = nerve_data_coef * (nerve_data_punc <= p_value)
                
            
                plot_nerve_maps_with_stats(nerve_data_coef, param_name, key,
                                            'coef', results_path)
                plot_nerve_maps_with_stats(nerve_thresholded_p, param_name, key,
                                            'p_corrected', results_path)
                plot_nerve_maps_with_stats(nerve_thresholded_punc, param_name, key,
                                            'p_uncorrected', results_path)
                
                
    return
        

def main(path="./", groups=None, debug=False, use_linear_model=True, 
         features=None, sides=None, covariates=None, 
         formula=None, correction_method='fdr_bh'):
    
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        
    logger.info("Starting aVP-Toolbox statistical analysis")
    
    try:
        
        participant_df = load_participants(path)
        full_dataframe = load_data(path, participant_df)
        
        result_path = op.join(path, "stats")
        os.makedirs(result_path, exist_ok=True)
        
        # TODO: change covariates with variables
        covariates = participant_df.columns.tolist()
        covariates.remove('subject')
        
        
        generate_statistics(dataframe=full_dataframe, features=features, sides=sides, results_path=result_path,
                            covariates=covariates, image_type='normalized', formula=formula,
                            correction_method=correction_method, kind='segment', p_value=0.05)
        
        generate_statistics(dataframe=full_dataframe, features=features, sides=sides, results_path=result_path,
                            covariates=covariates, image_type='normalized', formula=formula,
                            correction_method=correction_method, kind='slice', p_value=0.05)
        

        
        logger.info("Analysis completed successfully")
        return 
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Statistical analysis for aVP-Toolbox")
    parser.add_argument("--path", type=str, default="./")
    parser.add_argument("--groups", type=str, nargs='+')
    parser.add_argument("--features", type=str, nargs='+', default=['area'], help="Features to analyze (e.g. area, diameter).")
    parser.add_argument("--side", type=str, nargs='+', choices=['r', 'l', 'both'], default='both', help="Side to analyze (r, l, or both).")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use-lm", action="store_true")
    parser.add_argument("--covariates", type=str, nargs='+')
    parser.add_argument("--formula", type=str)
    parser.add_argument("--correction", type=str, default='fdr_bh',
                       choices=['bonferroni', 'fdr_bh', 'fdr_by', 'holm', 'hommel'])
    
    args = parser.parse_args()
    
    main(path=args.path, groups=args.groups, debug=args.debug, features=args.features, 
         sides=args.side, use_linear_model=args.use_lm, covariates=args.covariates,
         formula=args.formula, correction_method=args.correction)