import os
import shutil
import pytest
import nibabel as nib
import numpy as np
from unittest.mock import mock_open, patch, MagicMock
from pathlib import Path

@pytest.fixture
def test_environment():
    """Create a test environment with necessary directory structure."""
    test_dir = os.path.join(os.path.dirname(__file__), 'test_env')
    
    # Create directory structure
    study_path = os.path.join(test_dir, 'study')
    proc_dir = os.path.join(study_path, 'data', 'proc')
    orig_dir = os.path.join(study_path, 'data', 'orig')
    results_dir = os.path.join(study_path, 'results')
    
    # Ensure clean start
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create an ONcontrol.txt file within the test environment
    control_file_path = os.path.join(test_dir, 'ONcontrol.txt')
    with open(control_file_path, 'w') as f:
        f.write(study_path)
    
    # Create a test environment dictionary
    env = {
        'test_dir': test_dir,
        'study_path': study_path,
        'orig_dir': orig_dir,
        'proc_dir': proc_dir,
        'results_dir': results_dir,
        'control_file': control_file_path
    }
    
    yield env
    
    # Clean up after test
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

@pytest.fixture
def test_subjects(test_environment):
    """Create test subject directories."""
    env = test_environment
    
    # Create test subject directories
    subjects = ['sub-001', 'sub-002']
    
    for sbj in subjects:
        os.makedirs(os.path.join(env['orig_dir'], sbj), exist_ok=True)
        os.makedirs(os.path.join(env['proc_dir'], sbj), exist_ok=True)
    
    return subjects

def create_test_nifti(filepath, shape=(10, 10, 10), data=None, value=1):
    """Helper function to create a test NIfTI file."""
    if data is None:
        data = np.zeros(shape)
        data[3:7, 3:7, 3:7] = value
    
    img = nib.Nifti1Image(data, np.eye(4))
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    nib.save(img, filepath)
    return img

@pytest.fixture
def real_test_files():
    """Fixture to copy real test files to test environment."""
    def _copy_files(env, real_files_dir):
        if os.path.exists(real_files_dir):
            for item in os.listdir(real_files_dir):
                src = os.path.join(real_files_dir, item)
                if os.path.isdir(src):
                    dest = os.path.join(env['orig_dir'], item)
                    shutil.copytree(src, dest)
                else:
                    shutil.copy2(src, env['orig_dir'])
    return _copy_files

def mock_open_for_oncontrol(real_open, study_path):
    """Creates a mock for the open function that returns study path for ONcontrol.txt."""
    def mocked_open(filename, *args, **kwargs):
        if str(filename) == './ONcontrol.txt' or str(filename).endswith('/ONcontrol.txt'):
            mock_file = MagicMock()
            mock_file.read.return_value = study_path
            mock_file.__enter__.return_value = mock_file
            return mock_file
        return real_open(filename, *args, **kwargs)
    return mocked_open

@pytest.fixture
def patch_oncontrol(monkeypatch, test_environment):
    """Patch the open function for ONcontrol.txt."""
    monkeypatch.setattr('builtins.open', 
                      mock_open_for_oncontrol(open, test_environment["study_path"]))
    return test_environment["study_path"]