"""
Simple MRI Report Generator
Creates a clean report with:
- Side menu navigation
- NIfTI image viewer (one slice view)
- Basic statistics plot
- Text information section
"""

import jinja2
import plotly.graph_objects as go
from jinja2 import Template
import numpy as np
import logging
import os
from pathlib import Path
import nibabel as nib

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
    
    


def create_slice_viewer(volume, subject_name, slice_idx=None):
    """
    Create simple 2D slice viewer
    
    Args:
        volume: 3D numpy array
        subject_name: Name for the scan
        slice_idx: Which slice to show (default: middle)
    
    Returns:
        HTML string with Plotly figure
    """
    if slice_idx is None:
        slice_idx = volume.shape[2] // 2
    
    slice_data = volume[:, :, slice_idx]
    
    fig = go.Figure(data=go.Heatmap(
        z=slice_data,
        colorscale='Gray',
        hovertemplate='X: %{x}<br>Y: %{y}<br>Intensity: %{z:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{subject_name} - Axial Slice {slice_idx}',
        height=400,
        template='plotly_white',
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )
    
    return fig.to_html(include_plotlyjs='cdn', div_id=f'slice_{subject_name.lower().replace(" ", "_")}')


def create_statistics_plot(volume, subject_name):
    """
    Create bar plot with basic statistics
    
    Args:
        volume: 3D numpy array
        subject_name: Name for the scan
    
    Returns:
        HTML string with Plotly figure
    """
    stats = {
        'Mean': np.mean(volume),
        'Std Dev': np.std(volume),
        'Median': np.median(volume),
        'Min': np.min(volume),
        'Max': np.max(volume)
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(stats.keys()),
            y=list(stats.values()),
            marker_color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'],
            hovertemplate='%{x}: %{y:.2f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=f'{subject_name} - Basic Statistics',
        yaxis_title='Value',
        height=350,
        template='plotly_white'
    )
    
    return fig.to_html(include_plotlyjs=False, div_id=f'stats_{subject_name.lower().replace(" ", "_")}')


def get_text_info(volume, subject_name, filepath=None):
    """
    Generate text information about the scan
    
    Args:
        volume: 3D numpy array
        subject_name: Name for the scan
        filepath: Path to original NIfTI file (optional)
    
    Returns:
        Dictionary with text information
    """
    info = {
        'Subject ID': subject_name,
        'Dimensions': f'{volume.shape[0]} × {volume.shape[1]} × {volume.shape[2]}',
        'Voxel Count': f'{volume.size:,}',
        'Data Type': str(volume.dtype),
        'Mean Intensity': f'{np.mean(volume):.2f}',
        'Std Deviation': f'{np.std(volume):.2f}'
    }
    
    # Add metadata from NIfTI header if available
    if filepath and os.path.exists(filepath):
        try:
            img = nib.load(filepath)
            header = img.header
            info['Voxel Size'] = f'{header.get_zooms()[0]:.2f} × {header.get_zooms()[1]:.2f} × {header.get_zooms()[2]:.2f} mm'
            info['Data Type (Header)'] = str(header.get_data_dtype())
        except Exception as e:
            logger.warning(f"Could not read header info from {filepath}: {e}")
    
    return info


def generate_report(subjects_data, output_file='mri_report.html'):
    """
    Generate the HTML report
    
    Args:
        subjects_data: Dictionary with format:
            {
                'Subject 1': {
                    'volume': volume_array,
                    'filepath': 'path/to/file.nii.gz' (optional)
                },
                ...
            }
        output_file: Output filename
    """
    # Prepare data for each subject
    subjects = []
    for subject_name, data in subjects_data.items():
        volume = data.get('volume') if isinstance(data, dict) else data
        filepath = data.get('filepath') if isinstance(data, dict) else None
        
        subjects.append({
            'name': subject_name,
            'id': subject_name.lower().replace(' ', '_').replace('.', '_'),
            'image': create_slice_viewer(volume, subject_name),
            'stats_plot': create_statistics_plot(volume, subject_name),
            'info': get_text_info(volume, subject_name, filepath)
        })
    
    # HTML Template
    # Path of this file
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
    return output_file


def collect_subject_data(path="./", debug=False):
    """
    Collect all subject data from the pipeline output directory.
    
    Args:
        path: Root directory of the study
        debug: Enable debug logging
        
    Returns:
        Dictionary with subject data
    """
    path = Path(path)
    subjects_data = {}
    
    # Standard file patterns for aVP-Toolbox
    file_patterns = ['onl_linearize_4bc_iso06.nii.gz', 'onr_linearize_4bc_iso06.nii.gz']
    
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
        subject_dirs = [d for d in search_dir.iterdir() if d.is_dir()]
        
        for subject_dir in subject_dirs:
            subject_name = subject_dir.name
            
            # Find relevant NIfTI files
            for pattern in file_patterns:
                nifti_file = subject_dir / pattern
                if nifti_file.exists():
                    key = f"{subject_name}_{pattern.replace('.nii.gz', '')}"
                    logger.info(f"Found: {key}")
                    
                    try:
                        volume = load_nifti_data(str(nifti_file))
                        subjects_data[key] = {
                            'volume': volume,
                            'filepath': str(nifti_file)
                        }
                    except Exception as e:
                        logger.error(f"Failed to load {nifti_file}: {e}")
        
        # If we found data, stop searching
        if subjects_data:
            break
    
    if not subjects_data:
        logger.warning("No subject data found. Creating demo report.")
        # Create demo data
        subjects_data = {
            #'Demo Subject 001': {'volume': load_nifti_data(None)},
            #'Demo Subject 002': {'volume': load_nifti_data(None)},
        }
    
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
    
    # Collect subject data
    subjects_data = collect_subject_data(path, debug)
    
    logger.info(f"Collected data for {len(subjects_data)} subjects/scans")
    
    # Generate output path
    output_path = Path(path) / "reports"
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "avp_toolbox_report.html"
    
    # Generate report
    generate_report(subjects_data, str(output_file))
    
    logger.info("Report generation completed successfully.")


if __name__ == "__main__":
    main()