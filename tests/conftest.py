import os
import shutil
import pytest
import nibabel as nib
import numpy as np
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