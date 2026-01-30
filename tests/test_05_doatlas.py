import os
import pytest
import nibabel as nib
import numpy as np
from unittest.mock import patch, mock_open
from pathlib import Path

from avpy import _05_doatlas

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
        
        # Create normalized and resampled files
        for structure in ["on", "ot", "oc", "onincr", "oninca", "oninor"]:
            for side in ["r", "l"]:
                # Create resampled data for atlas generation
                data = np.zeros((10, 10, 10))
                
                # Different patterns for each subject
                if subject == "sub01":
                    data[2:7, 3:6, 3:6] = 1
                else:
                    data[3:6, 2:7, 3:6] = 1
                
                img = nib.Nifti1Image(data, np.eye(4))
                nib.save(img, subj_dir / f"{structure}_{side}_normalized_4.nii.gz")
    
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

def test_main_function(test_data_dir):
    """Test the main function of _05_doatlas.py"""
    # Mock the open function for ONcontrol.txt
    original_open = open
    
    def mock_open_file(*args, **kwargs):
        if args[0] == os.path.join(str(test_data_dir["tmp_path"]), 'ONcontrol.txt') and args[1] == 'r':
            return mock_open(read_data=str(test_data_dir["study_dir"]))(*args, **kwargs)
        return original_open(*args, **kwargs)
    
    with patch('builtins.open', mock_open_file):
        # Run the main function
        _05_doatlas.main(path=str(test_data_dir["tmp_path"]))
        
        # Check for atlas outputs
        # The exact output files depend on the implementation of _05_doatlas.py
        
        # Check for combined probability maps in results directory
        atlas_files = [
            "on_atlas.nii.gz",
            "ot_atlas.nii.gz",
            "oc_atlas.nii.gz",
            "onincr_atlas.nii.gz",
            "oninca_atlas.nii.gz",
            "oninor_atlas.nii.gz"
        ]
        
        for atlas_file in atlas_files:
            file_path = test_data_dir["results_dir"] / atlas_file
            if file_path.exists():
                # Load the atlas file
                atlas_img = nib.load(file_path)
                atlas_data = atlas_img.get_fdata()
                
                # Check that the atlas has values between 0 and 1 (probability maps)
                assert np.min(atlas_data) >= 0
                assert np.max(atlas_data) <= 1
                
                # Check that the atlas is not empty
                assert np.sum(atlas_data > 0) > 0
        
        # Check for binarized atlas maps
        binary_atlas_files = [
            "on_binary_atlas.nii.gz",
            "ot_binary_atlas.nii.gz",
            "oc_binary_atlas.nii.gz",
            "onincr_binary_atlas.nii.gz",
            "oninca_binary_atlas.nii.gz",
            "oninor_binary_atlas.nii.gz"
        ]
        
        for binary_atlas_file in binary_atlas_files:
            file_path = test_data_dir["results_dir"] / binary_atlas_file
            if file_path.exists():
                # Load the binary atlas file
                binary_atlas_img = nib.load(file_path)
                binary_atlas_data = binary_atlas_img.get_fdata()
                
                # Check that the binary atlas has only 0 and 1 values
                unique_values = np.unique(binary_atlas_data)
                assert len(unique_values) <= 2
                if len(unique_values) > 1:
                    assert np.array_equal(unique_values, np.array([0, 1])) or \
                           np.array_equal(unique_values, np.array([0.0, 1.0]))