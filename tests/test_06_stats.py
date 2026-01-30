"""
Tests for the statistical analysis module (_06_stats.py)
"""

import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from avpy import _06_stats


@pytest.fixture
def stats_test_environment(tmp_path):
    """Create a test environment for stats testing."""
    study_path = tmp_path / "study"
    
    # Create directory structure
    results_dir = study_path / "results"
    results_dir.mkdir(parents=True)
    
    # Create group directories
    for group in ['HC', 'PTS']:
        group_dir = study_path / group
        group_data_dir = group_dir / "data"
        group_results_dir = group_dir / "results"
        group_data_dir.mkdir(parents=True)
        group_results_dir.mkdir(parents=True)
    
    # Create atlas directory and mock atlas file
    atlas_dir = tmp_path / "atlas"
    atlas_dir.mkdir(parents=True)
    
    return {
        'study_path': str(study_path),
        'results_dir': str(results_dir),
        'atlas_dir': str(atlas_dir),
        'groups': ['HC', 'PTS']
    }


@pytest.fixture
def mock_atlas(stats_test_environment):
    """Create a mock atlas file."""
    import nibabel as nib
    
    # Create a simple 3D array as mock atlas
    atlas_data = np.ones((100, 85, 100))  # Shape with 85 slices in y-dimension
    atlas_img = nib.Nifti1Image(atlas_data, np.eye(4))
    
    # Patch the atlas path in the module
    atlas_path = os.path.join(stats_test_environment['atlas_dir'], 'aVP-24_prob50.nii.gz')
    nib.save(atlas_img, atlas_path)
    
    # Patch the module's atlas_dir
    with patch.object(_06_stats, 'atlas_dir', stats_test_environment['atlas_dir']):
        yield atlas_path


def create_sample_dataframe(subjects, groups):
    """Create a sample dataframe similar to CSA_slice_iso.xlsx structure."""
    data = []
    for i, (subject, group) in enumerate(zip(subjects, groups)):
        for side in ['l', 'r']:
            for slice_num in range(10):
                for image_type in ['linearize', 'normalized']:
                    data.append({
                        'subject': subject,
                        'group': group,
                        'side': side,
                        'image_type': image_type,
                        'current_slice_yz': slice_num,
                        'segment': 'iOrb',
                        'eccent': np.random.rand(),
                        'area': np.random.rand() * 10,
                        'major_axis': np.random.rand() * 5,
                        'minor_axis': np.random.rand() * 3,
                    })
    return pd.DataFrame(data)


def test_consolidated_file_with_group_column(stats_test_environment, mock_atlas):
    """Test loading from consolidated file that already has group column."""
    study_path = stats_test_environment['study_path']
    groups = stats_test_environment['groups']
    
    # Create consolidated results file with group column
    subjects = ['sub-001', 'sub-002', 'sub-003', 'sub-004']
    subject_groups = ['HC', 'HC', 'PTS', 'PTS']
    df = create_sample_dataframe(subjects, subject_groups)
    
    consolidated_file = os.path.join(stats_test_environment['results_dir'], 'CSA_slice_iso.xlsx')
    df.to_excel(consolidated_file, index=False)
    
    # Mock the stats calculations to avoid complex dependencies
    with patch.object(_06_stats, 'calculate_segment_statistics') as mock_calc:
        mock_calc.return_value = pd.DataFrame()
        
        # Call generate_nerve_maps
        _, _ = _06_stats.generate_nerve_maps(
            path=study_path,
            features=['eccent', 'area'],
            sides=['l', 'r'],
            groups=groups,
            image_type='normalized',
            generate_figures=False,
            debug=True
        )
    
    # Verify that the consolidated file was used (check logs or mock calls)
    assert mock_calc.called


def test_consolidated_file_without_group_column(stats_test_environment, mock_atlas):
    """Test loading from consolidated file without group column and merging with subject lists."""
    study_path = stats_test_environment['study_path']
    groups = stats_test_environment['groups']
    
    # Create consolidated results file WITHOUT group column
    subjects = ['sub-001', 'sub-002', 'sub-003', 'sub-004']
    df = create_sample_dataframe(subjects, [''] * len(subjects))
    df = df.drop(columns=['group'])  # Remove group column
    
    consolidated_file = os.path.join(stats_test_environment['results_dir'], 'CSA_slice_iso.xlsx')
    df.to_excel(consolidated_file, index=False)
    
    # Create subject list files for each group
    for i, group in enumerate(groups):
        group_subjects = subjects[i*2:(i+1)*2]  # Split subjects between groups
        sbj_list_file = os.path.join(study_path, group, 'data', 'sbj.list')
        with open(sbj_list_file, 'w') as f:
            f.write('\n'.join(group_subjects))
    
    # Mock the stats calculations
    with patch.object(_06_stats, 'calculate_segment_statistics') as mock_calc:
        mock_calc.return_value = pd.DataFrame()
        
        # Call generate_nerve_maps
        _, _ = _06_stats.generate_nerve_maps(
            path=study_path,
            features=['eccent', 'area'],
            sides=['l', 'r'],
            groups=groups,
            image_type='normalized',
            generate_figures=False,
            debug=True
        )
    
    assert mock_calc.called


def test_fallback_to_group_specific_files(stats_test_environment, mock_atlas):
    """Test fallback to reading from group-specific files when consolidated file doesn't exist."""
    study_path = stats_test_environment['study_path']
    groups = stats_test_environment['groups']
    
    # Don't create consolidated file, only create group-specific files
    for i, group in enumerate(groups):
        subjects = [f'sub-{i*2+1:03d}', f'sub-{i*2+2:03d}']
        df = create_sample_dataframe(subjects, [group] * len(subjects))
        df = df.drop(columns=['group'])  # Group will be added by the function
        
        group_file = os.path.join(study_path, group, 'results', 'CSA_slice_iso.xlsx')
        df.to_excel(group_file, index=False)
    
    # Mock the stats calculations
    with patch.object(_06_stats, 'calculate_segment_statistics') as mock_calc:
        mock_calc.return_value = pd.DataFrame()
        
        # Call generate_nerve_maps
        _, _ = _06_stats.generate_nerve_maps(
            path=study_path,
            features=['eccent', 'area'],
            sides=['l', 'r'],
            groups=groups,
            image_type='normalized',
            generate_figures=False,
            debug=True
        )
    
    assert mock_calc.called


def test_consolidated_file_missing_subject_column(stats_test_environment, mock_atlas):
    """Test that error is raised when consolidated file doesn't have subject column."""
    study_path = stats_test_environment['study_path']
    groups = stats_test_environment['groups']
    
    # Create consolidated results file WITHOUT subject column
    df = pd.DataFrame({
        'group': ['HC'] * 10,
        'area': np.random.rand(10)
    })
    
    consolidated_file = os.path.join(stats_test_environment['results_dir'], 'CSA_slice_iso.xlsx')
    df.to_excel(consolidated_file, index=False)
    
    # This should raise ValueError about missing subject column
    with pytest.raises(ValueError, match="subject"):
        _06_stats.generate_nerve_maps(
            path=study_path,
            features=['eccent', 'area'],
            sides=['l', 'r'],
            groups=groups,
            image_type='normalized',
            generate_figures=False,
            debug=True
        )


def test_consolidated_file_unmapped_subjects(stats_test_environment, mock_atlas):
    """Test that error is raised when subjects cannot be mapped to groups."""
    study_path = stats_test_environment['study_path']
    groups = stats_test_environment['groups']
    
    # Create consolidated results file without group column
    subjects = ['sub-001', 'sub-002', 'sub-999']  # sub-999 won't be in any group
    df = create_sample_dataframe(subjects, [''] * len(subjects))
    df = df.drop(columns=['group'])
    
    consolidated_file = os.path.join(stats_test_environment['results_dir'], 'CSA_slice_iso.xlsx')
    df.to_excel(consolidated_file, index=False)
    
    # Create subject list files with only sub-001 and sub-002
    for i, group in enumerate(groups):
        group_subjects = [subjects[i]]  # Only one subject per group
        sbj_list_file = os.path.join(study_path, group, 'data', 'sbj.list')
        with open(sbj_list_file, 'w') as f:
            f.write(group_subjects[0])
    
    # This should raise ValueError about unmapped subjects
    with pytest.raises(ValueError, match="Could not determine group"):
        _06_stats.generate_nerve_maps(
            path=study_path,
            features=['eccent', 'area'],
            sides=['l', 'r'],
            groups=groups,
            image_type='normalized',
            generate_figures=False,
            debug=True
        )


def test_consolidated_file_missing_specified_groups(stats_test_environment, mock_atlas):
    """Test that error is raised when specified groups are not in the consolidated file."""
    study_path = stats_test_environment['study_path']
    
    # Create consolidated results file with only HC group
    subjects = ['sub-001', 'sub-002']
    df = create_sample_dataframe(subjects, ['HC', 'HC'])
    
    consolidated_file = os.path.join(stats_test_environment['results_dir'], 'CSA_slice_iso.xlsx')
    df.to_excel(consolidated_file, index=False)
    
    # Request both HC and PTS, but PTS is not in the data
    with pytest.raises(ValueError, match="not found in data"):
        _06_stats.generate_nerve_maps(
            path=study_path,
            features=['eccent', 'area'],
            sides=['l', 'r'],
            groups=['HC', 'PTS', 'MS'],  # MS not in data
            image_type='normalized',
            generate_figures=False,
            debug=True
        )
