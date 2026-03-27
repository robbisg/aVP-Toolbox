#!/usr/bin/env python3
"""
Pytest configuration file for aVP-Toolbox tests.

This file contains shared fixtures and configuration for all tests.
"""

import pytest
import os
import sys
import tempfile
import shutil
import numpy as np
import nibabel as nib
from unittest.mock import MagicMock, patch

# Add src to path for importing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="session")
def mock_dependencies():
    """Mock external dependencies that might not be available in test environment."""
    with patch.dict('sys.modules', {
        'sekupy': MagicMock(),
        'sekupy.results': MagicMock(),
        'pingouin': MagicMock(),
        'matplotlib': MagicMock(),
        'matplotlib.pyplot': MagicMock(),
        'nilearn': MagicMock(),
        'nilearn.image': MagicMock(),
    }):
        yield


@pytest.fixture
def temp_avpy_structure():
    """Create a temporary aVP-Toolbox directory structure for testing."""
    temp_dir = tempfile.mkdtemp(prefix="avpy_test_")
    
    # Create standard aVP directory structure
    directories = [
        "data/orig",
        "data/proc", 
        "results",
        "maps",
        "atlas"
    ]
    
    for dir_path in directories:
        full_path = os.path.join(temp_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
    
    # Create mock atlas file
    atlas_data = np.random.randint(0, 25, size=(256, 256, 72))
    atlas_affine = np.eye(4) * 0.6
    atlas_affine[3, 3] = 1
    
    atlas_img = nib.Nifti1Image(atlas_data, atlas_affine)
    atlas_path = os.path.join(temp_dir, "atlas", "aVP-24_label.nii.gz")
    nib.save(atlas_img, atlas_path)
    
    yield temp_dir
    
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_subject_data():
    """Create mock subject data for testing."""
    def _create_subject_data(base_dir, subject_id):
        subject_dir = os.path.join(base_dir, "data", "orig", subject_id)
        os.makedirs(subject_dir, exist_ok=True)
        
        # Create mock NIfTI files with appropriate labels
        files_config = {
            "otr.nii.gz": {"shape": (64, 64, 32), "values": [10]},
            "otl.nii.gz": {"shape": (64, 64, 32), "values": [10]},
            "onc.nii.gz": {"shape": (64, 64, 32), "values": [8, 9]},
            "onr.nii.gz": {"shape": (64, 64, 32), "values": [2, 4, 6]},
            "onl.nii.gz": {"shape": (64, 64, 32), "values": [2, 4, 6]},
        }
        
        affine = np.eye(4) * 0.6
        affine[3, 3] = 1
        
        created_files = []
        for filename, config in files_config.items():
            shape = config["shape"]
            values = config["values"]
            
            # Create data with specific label values
            data = np.random.choice(values, size=shape).astype(np.float32)
            
            img = nib.Nifti1Image(data, affine)
            filepath = os.path.join(subject_dir, filename)
            nib.save(img, filepath)
            created_files.append(filepath)
        
        return created_files
    
    return _create_subject_data


@pytest.fixture
def mock_stats_data():
    """Create mock statistical analysis data."""
    def _create_stats_data(base_dir, groups=None):
        if groups is None:
            groups = ["HC", "PTS"]
        
        import pandas as pd
        
        created_files = []
        for group in groups:
            results_dir = os.path.join(base_dir, group, "results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Create mock statistical data
            n_subjects = 20
            n_slices = 10
            
            data = []
            for subject_idx in range(n_subjects):
                for slice_idx in range(1, n_slices + 1):
                    for side in ['r', 'l']:
                        row = {
                            'subject_id': f'sub{subject_idx:03d}',
                            'side': side,
                            'type': 'normalized', 
                            'curr_sli_yz': slice_idx,
                            'original_slice_yz': slice_idx,
                            'Eccent': np.random.normal(0.7 if group == "HC" else 0.8, 0.1),
                            'CSArea': np.random.normal(12 if group == "HC" else 10, 2)
                        }
                        data.append(row)
            
            df = pd.DataFrame(data)
            filepath = os.path.join(results_dir, "aVP_slice_data_iso.xlsx")
            df.to_excel(filepath, index=False)
            created_files.append(filepath)
        
        return created_files
    
    return _create_stats_data


@pytest.fixture
def sample_nifti_image():
    """Create a sample NIfTI image for testing."""
    # Create sample 3D data
    data = np.random.rand(64, 64, 32).astype(np.float32)
    
    # Create affine transformation matrix
    affine = np.array([
        [0.6, 0, 0, -74.4],
        [0, 0.6, 0, -60.6], 
        [0, 0, 0.6, -21.0],
        [0, 0, 0, 1]
    ])
    
    img = nib.Nifti1Image(data, affine)
    return img


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add 'unit' marker to unit test functions
        if "test_unit" in item.name or "Unit" in str(item.cls):
            item.add_marker(pytest.mark.unit)
        
        # Add 'integration' marker to integration test functions  
        if "test_integration" in item.name or "Integration" in str(item.cls):
            item.add_marker(pytest.mark.integration)
        
        # Add 'slow' marker to tests that might be slow
        if "test_multiple" in item.name or "parallel" in item.name:
            item.add_marker(pytest.mark.slow)