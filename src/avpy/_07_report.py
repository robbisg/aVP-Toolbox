#!/usr/bin/env python3
"""
Simple MRI Report Generator
Creates a clean report with:
- Side menu navigation (one entry per subject)
- Line plots from processed data showing area and eccentricity
- NIfTI image viewers
"""

import jinja2
import plotly.graph_objects as go
from jinja2 import Template
import numpy as np
import logging
import os
from pathlib import Path
import nibabel as nib
import pandas as pd
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)
NAME = "report"


def load_nifti_data(filepath):
    """
    Load NIfTI file and return volume data
    
    Args:
        filepath: Path to .nii or .nii.gz file
    
    Returns:
        3D numpy array
    """
    img = nib.load(filepath)
    return img.get_fdata()


def create_combined_feature_plot(subject_df, subject_name):
    """
    Create 4 separate plots: area/eccentricity × linearized/normalized
    
    Args:
        subject_df: DataFrame filtered for single subject with columns:
                   ['subject', 'side', 'image_type', 'current_slice_yz', 'area', 'eccent', ...]
        subject_name: Name of the subject
    
    Returns:
        HTML string with Plotly figure
    """
    # Define color mapping: Left = Red, Right = Blue
    side_colors = {'l': '#e74c3c', 'r': '#3498db'}  # Red for left, Blue for right
    
    # Create 2x2 subplots
    fig = make_subplots(
        rows=2, 
        cols=2,
        subplot_titles=[
            'Area - Linearized',
            'Area - Normalized', 
            'Eccentricity - Linearized',
            'Eccentricity - Normalized'
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    # Map for subplot positions
    subplot_map = {
        ('area', 'linearize'): (1, 1),
        ('area', 'normalized'): (1, 2),
        ('eccent', 'linearize'): (2, 1),
        ('eccent', 'normalized'): (2, 2)
    }
    
    # Process each combination
    for feature in ['area', 'eccent']:
        for image_type in ['linearize', 'normalized']:
            row, col = subplot_map[(feature, image_type)]
            
            # Plot left and right sides
            for side in ['l', 'r']:
                # Filter data
                subset = subject_df[
                    (subject_df['side'] == side) & 
                    (subject_df['image_type'] == image_type)
                ].sort_values('current_slice_yz')
                
                if len(subset) == 0:
                    continue
                
                # Create legend label
                side_label = 'Left' if side == 'l' else 'Right'
                
                # Determine units and format
                if feature == 'area':
                    y_values = subset['area']
                    hover_template = 'Slice: %{x}<br>Area: %{y:.2f} mm²<extra></extra>'
                else:
                    y_values = subset['eccent']
                    hover_template = 'Slice: %{x}<br>Eccentricity: %{y:.3f}<extra></extra>'
                
                # Add trace
                fig.add_trace(
                    go.Scatter(
                        x=subset['current_slice_yz'],
                        y=y_values,
                        mode='lines+markers',
                        name=side_label,
                        line=dict(
                            color=side_colors[side],
                            width=2
                        ),
                        marker=dict(size=4),
                        legendgroup=side_label,
                        showlegend=(row == 1 and col == 1),  # Only show legend once
                        hovertemplate=hover_template
                    ),
                    row=row, col=col
                )
    
    # Update x-axes labels
    fig.update_xaxes(title_text="Slice Position", row=2, col=1)
    fig.update_xaxes(title_text="Slice Position", row=2, col=2)
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Area (mm²)", row=1, col=1)
    fig.update_yaxes(title_text="Area (mm²)", row=1, col=2)
    fig.update_yaxes(title_text="Eccentricity", row=2, col=1)
    fig.update_yaxes(title_text="Eccentricity", row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title=f'{subject_name} - Optic Nerve Morphometry',
        height=800,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        )
    )
    
    subject_str = str(subject_name).lower().replace(" ", "_").replace(".", "_")
    
    return fig.to_html(include_plotlyjs='cdn', div_id=f'features_{subject_str}')


def create_combined_viewers(scans_data, subject_name):
    """
    Create a combined figure with multiple slice viewers
    
    Args:
        scans_data: List of dictionaries with 'name', 'volume', 'filepath'
        subject_name: Name of the subject
    
    Returns:
        HTML string with combined Plotly figure
    """
    n_scans = len(scans_data)
    
    if n_scans == 0:
        return "<p>No image data available</p>"
    
    # Create subplots
    fig = make_subplots(
        rows=1, 
        cols=n_scans,
        subplot_titles=[scan['name'] for scan in scans_data],
        horizontal_spacing=0.05
    )
    
    for idx, scan in enumerate(scans_data, 1):
        volume = scan['volume']
        slice_idx = volume.shape[2] // 2
        slice_data = volume[:, :, slice_idx]
        
        fig.add_trace(
            go.Heatmap(
                z=slice_data,
                colorscale='Gray',
                showscale=(idx == n_scans),  # Only show colorbar for last plot
                hovertemplate='X: %{x}<br>Y: %{y}<br>Intensity: %{z:.1f}<extra></extra>'
            ),
            row=1, col=idx
        )
        
        # Update axes
        fig.update_xaxes(showticklabels=False, row=1, col=idx)
        fig.update_yaxes(showticklabels=False, row=1, col=idx)
    
    fig.update_layout(
        title=f'{subject_name} - All Scans',
        height=400,
        template='plotly_white'
    )
    
    subject_str = str(subject_name).lower().replace(" ", "_").replace(".", "_")
    return fig.to_html(include_plotlyjs=False, div_id=f'slices_{subject_str}')


def get_scan_info(volume, scan_name, filepath=None):
    """
    Generate text information about a single scan
    
    Args:
        volume: 3D numpy array
        scan_name: Name for the scan
        filepath: Path to original NIfTI file (optional)
    
    Returns:
        Dictionary with text information
    """
    info = {
        'Scan Name': scan_name,
        'Dimensions': f'{volume.shape[0]} × {volume.shape[1]} × {volume.shape[2]}',
        'Voxel Count': f'{volume.size:,}',
        'Data Type': str(volume.dtype),
        'Mean Intensity': f'{np.mean(volume):.2f}',
        'Std Deviation': f'{np.std(volume):.2f}',
        'Min Value': f'{np.min(volume):.2f}',
        'Max Value': f'{np.max(volume):.2f}'
    }
    
    # Add metadata from NIfTI header if available
    if filepath and os.path.exists(filepath):
        try:
            img = nib.load(filepath)
            header = img.header
            info['Voxel Size'] = f'{header.get_zooms()[0]:.2f} × {header.get_zooms()[1]:.2f} × {header.get_zooms()[2]:.2f} mm'
        except Exception as e:
            logger.warning(f"Could not read header info from {filepath}: {e}")
    
    return info


def load_processed_data(path):
    """
    Load the processed data from CSA_slice_iso.xlsx
    
    Args:
        path: Root directory of the study
        
    Returns:
        DataFrame with processed data or None if file not found
    """
    excel_path = Path(path) / "results" / "CSA_slice_iso.xlsx"
    
    if not excel_path.exists():
        logger.warning(f"Processed data file not found: {excel_path}")
        return None
    
    try:
        df = pd.read_excel(excel_path)
        logger.info(f"Loaded processed data: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Failed to load processed data: {e}")
        return None


def generate_report(subjects_data, processed_df, output_file='mri_report.html'):
    """
    Generate the HTML report
    
    Args:
        subjects_data: Dictionary with format:
            {
                'Subject1': [
                    {
                        'name': 'scan_name',
                        'volume': volume_array,
                        'filepath': 'path/to/file.nii.gz'
                    },
                    ...
                ],
                ...
            }
        processed_df: DataFrame with processed morphometry data
        output_file: Output filename
    """
    # Prepare data for each subject
    subjects = []
    
    # Get list of subjects from processed data if available
    if processed_df is not None:
        subjects_in_df = processed_df['subject'].unique()
    else:
        subjects_in_df = []
    
    for subject_name in subjects_in_df:
        # Filter data for this subject
        subject_df = processed_df[processed_df['subject'] == subject_name]
        
        # Create feature plots from processed data
        feature_plot = create_combined_feature_plot(subject_df, subject_name)
        
        # Here subject may be a int or str - ensure consistent key type
        if isinstance(subject_name, int):
            subject_key = f"{subject_name:02d}"
            logger.debug(f"INT: Processing subject: {subject_name} as key {subject_key}")
        else:
            subject_key = f"00{subject_name}"
        
        # Get image data if available
        logger.debug(f"Processing subject: {subject_name} (key: {subject_key})")
        logger.debug(f"Subjects data keys: {list(subjects_data.keys())}")
        scans_list = subjects_data.get(subject_key, [])
        
        logger.debug(f"Subject {subject_name}: Found {len(scans_list)} scans")
        
        # Create combined image viewers
        if scans_list:
            logger.debug(f"Creating image viewers for {subject_name}")
            combined_images = create_combined_viewers(scans_list, subject_name)
            
            # Collect scan information
            scans_info = []
            for scan in scans_list:
                scan_info = get_scan_info(
                    scan['volume'], 
                    scan['name'], 
                    scan.get('filepath')
                )
                scans_info.append(scan_info)
                logger.debug(f"  - Added info for: {scan['name']}")
        else:
            logger.warning(f"No image scans found for subject {subject_name}")
            combined_images = "<p>No image data available for this subject</p>"
            scans_info = []
        
        
        subject_str = str(subject_name).lower().replace(" ", "_").replace(".", "_")
        subjects.append({
            'name': subject_name,
            'id': subject_str,
            'feature_plot': feature_plot,
            'images': combined_images,
            'scans_info': scans_info,
            'n_scans': len(scans_list),
            'has_data': True
        })
        
        logger.info(f"Added subject {subject_name} with {len(scans_list)} scans and {len(scans_info)} info entries")
    
    # HTML Template
    current_dir = Path(__file__).parent
    template_loader = jinja2.FileSystemLoader(searchpath=current_dir / "templates")
    template_env = jinja2.Environment(loader=template_loader)
    TEMPLATE_FILE = "report.html.jinja"
    template = template_env.get_template(TEMPLATE_FILE)
    
    # Render and save
    html_content = template.render(
        title="aVP-Toolbox MRI Report",
        subjects=subjects
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"✅ Report generated: {output_file}")
    logger.info(f"   Total subjects: {len(subjects)}")
    
    return output_file


def collect_subject_data(path="./", debug=False):
    """
    Collect all subject data from the pipeline output directory.
    Groups scans by subject.
    
    Args:
        path: Root directory of the study
        debug: Enable debug logging
        
    Returns:
        Dictionary with subject data grouped by subject
    """
    path = Path(path)
    subjects_data = {}
    
    # Standard file patterns for aVP-Toolbox
    file_patterns = [
        ('onl_linearize_4bc_iso06.nii.gz', 'Left - Linearized'),
        ('onr_linearize_4bc_iso06.nii.gz', 'Right - Linearized'),
        ('onl_normalized_4bc_iso06.nii.gz', 'Left - Normalized'),
        ('onr_normalized_4bc_iso06.nii.gz', 'Right - Normalized')
    ]
    
    # Look for processed data in common locations
    search_dirs = [
        path / "data" / "proc",
        path / "data",
        path
    ]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
            
        logger.info(f"Searching for subjects in: {search_dir}")
        
        # Find subject directories
        subject_dirs = [d for d in search_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        for subject_dir in subject_dirs:
            subject_name = subject_dir.name
            
            # Initialize list for this subject if not exists
            if subject_name not in subjects_data:
                subjects_data[subject_name] = []
            
            # Find relevant NIfTI files
            for pattern, scan_name in file_patterns:
                nifti_file = subject_dir / pattern
                if nifti_file.exists():
                    logger.debug(f"Found: {subject_name} - {scan_name}")
                    
                    try:
                        volume = load_nifti_data(str(nifti_file))
                        subjects_data[subject_name].append({
                            'name': scan_name,
                            'volume': volume,
                            'filepath': str(nifti_file)
                        })
                    except Exception as e:
                        logger.error(f"Failed to load {nifti_file}: {e}")
        
        # If we found data, stop searching
        if subjects_data:
            break
    
    # Remove subjects with no scans
    subjects_data = {k: v for k, v in subjects_data.items() if v}
    
    if subjects_data:
        logger.info(f"Found image data for {len(subjects_data)} subjects")
        for subject_name, scans in subjects_data.items():
            logger.info(f"  - {subject_name}: {len(scans)} scans")
    else:
        logger.warning("No image data found in directories")
    
    return subjects_data


def main(path="./", debug=False):
    """
    Main report generation function compatible with aVP-Toolbox pipeline.
    
    Args:
        path: Root directory of the study
        debug: Enable debug logging
    """
    logger.info("Starting report generation...")
    
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load processed morphometry data
    processed_df = load_processed_data(path)
    
    if processed_df is None:
        logger.error("Cannot generate report without processed data (CSA_slice_iso.xlsx)")
        logger.error("Please run the normalization pipeline first (steps: normalize, resample, normalize_stats)")
        return
    
    # Collect subject image data (optional - for image viewers)
    subjects_data = collect_subject_data(path, debug)
    
    if subjects_data:
        total_scans = sum(len(scans) for scans in subjects_data.values())
        logger.info(f"Collected {total_scans} image files from {len(subjects_data)} subjects")
    
    # Generate output path
    output_path = Path(path) / "reports"
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "avp_toolbox_report.html"
    
    # Generate report
    generate_report(subjects_data, processed_df, str(output_file))
    
    logger.info("Report generation completed successfully.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate aVP-Toolbox HTML report")
    parser.add_argument("--path", type=str, default="./", help="Root directory of the study")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    main(path=args.path, debug=args.debug)