import os
import pytest
import nibabel as nib
import numpy as np
from unittest.mock import patch, mock_open
from pathlib import Path

from avpy import _03b_resample

@pytest.fixture
def test_data_dir(tmp_path):
    """Setup test directory structure with test data"""
    # Create directory structure
    study_dir = tmp_path / "study"
    proc_dir = study_dir / "data" / "proc"
    proc_dir.mkdir(parents=True, exist_ok=True)
    results_dir = study_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test subjects
    subjects = ["sub01", "sub02"]
    for subject in subjects:
        subj_dir = proc_dir / subject
        subj_dir.mkdir(parents=True)
        
        # Create normalized files to resample
        for structure in ["on", "ot", "oc"]:
            for side in ["r", "l"]:
                # Create normalized files with different shapes and resolutions
                data = np.zeros((10, 10, 10))
                
                # Make each structure have a different shape
                offset = 2 if subject == "sub01" else 4
                length = 5 if subject == "sub01" else 3
                
                if structure == "on":
                    data[offset:offset+length, 3:6, 3:6] = 1
                elif structure == "ot":
                    data[3:6, offset:offset+length, 3:6] = 1
                elif structure == "oc":
                    data[3:6, 3:6, offset:offset+length] = 1
                
                # Different affine for different subjects to test resampling
                affine = np.eye(4)
                if subject == "sub01":
                    # 1mm isotropic
                    pass
                else:
                    # Non-isotropic resolution
                    affine[0, 0] = 0.8  # 0.8mm x-resolution
                    affine[1, 1] = 1.2  # 1.2mm y-resolution
                    affine[2, 2] = 1.0  # 1.0mm z-resolution
                
                img = nib.Nifti1Image(data, affine)
                nib.save(img, subj_dir / f"{structure}_{side}_norm.nii.gz")
    
    # Create subject list file
    sbj_list_path = study_dir / "data" / "sbj.list"
    with open(sbj_list_path, "w") as f:
        for subject in subjects:
            f.write(f"{subject}\n")
    
    # Write study path to control file
    control_file = tmp_path / "ONcontrol.txt"
    with open(control_file, "w") as f:
        f.write(str(study_dir))
    
    yield {
        "tmp_path": tmp_path,
        "study_dir": study_dir,
        "proc_dir": proc_dir,
        "results_dir": results_dir,
        "subjects": subjects,
        "control_file": control_file
    }

def test_process_images(test_data_dir):
    """Test the process_images function"""
    # Test directly if the function exists
    if hasattr(_03b_resample, 'process_images'):
        # Call the process_images function
        study_path = str(test_data_dir["study_dir"])
        base_image = None  # This should be determined by your implementation
        
        # In a real test, you'd provide the base_image if needed
        # Here we just check if the function runs without errors
        _03b_resample.process_images(study_path, base_image)
        
        # Add assertions based on expected outputs

def test_main_function(test_data_dir):
    """Test the main function of _03b_resample.py"""
    # Mock the open function for ONcontrol.txt
    original_open = open
    
    def mock_open_file(*args, **kwargs):
        if args[0] == os.path.join(str(test_data_dir["tmp_path"]), 'ONcontrol.txt') and args[1] == 'r':
            return mock_open(read_data=str(test_data_dir["study_dir"]))(*args, **kwargs)
        return original_open(*args, **kwargs)
    
    with patch('builtins.open', mock_open_file):
        # Run the main function
        _03b_resample.main(path=str(test_data_dir["tmp_path"]))
        
        # Check for resampled outputs for each subject
        for subject in test_data_dir["subjects"]:
            subj_dir = test_data_dir["proc_dir"] / subject
            
            # Check for resampled files
            for structure in ["on", "ot", "oc"]:
                for side in ["r", "l"]:
                    # Each structure should have a resampled version
                    resampled_file = subj_dir / f"{structure}_{side}_resampled.nii.gz"
                    assert resampled_file.exists(), f"Resampled file {resampled_file} was not created"
                    
                    # Load the normalized and resampled files
                    normalized = nib.load(subj_dir / f"{structure}_{side}_norm.nii.gz")
                    resampled = nib.load(resampled_file)
                    
                    # Check that resampling was applied
                    resampled_affine = resampled.affine
                    
                    # All resampled images should have the same resolution across subjects
                    # This test depends on your implementation details
                    x_resolution = abs(resampled_affine[0, 0])
                    y_resolution = abs(resampled_affine[1, 1])
                    z_resolution = abs(resampled_affine[2, 2])
                    
                    # Check for isotropic resolution (if that's what your implementation does)
                    assert abs(x_resolution - y_resolution) < 0.001
                    assert abs(y_resolution - z_resolution) < 0.001
                    assert abs(x_resolution - z_resolution) < 0.001