import os
import pytest
import nibabel as nib
import numpy as np
from unittest.mock import patch, mock_open
from pathlib import Path

from avpy import _02_basics

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
        
        # Create test NIfTI files with different volumes
        # Create segmentation files that would be outputs of _01_prep
        for segment, value in [
            ("on_r", 1), ("on_l", 1),
            ("ot_r", 16), ("ot_l", 16),
            ("oc_r", 8), ("oc_l", 8),
            ("onincr_r", 4), ("onincr_l", 4),
            ("oninca_r", 2), ("oninca_l", 2),
            ("oninor_r", 1), ("oninor_l", 1)
        ]:
            # Create different sized ROIs for each subject for testing
            size = 4 if subject == "sub01" else 6
            data = np.zeros((10, 10, 10))
            data[2:2+size, 2:2+size, 2:2+size] = value
            img = nib.Nifti1Image(data, np.eye(4))
            nib.save(img, subj_dir / f"{segment}.nii.gz")
    
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
    """Test the main function of _02_basics.py"""
    # Mock the open function for ONcontrol.txt
    original_open = open
    
    def mock_open_file(*args, **kwargs):
        if args[0] == os.path.join(str(test_data_dir["tmp_path"]), 'ONcontrol.txt') and args[1] == 'r':
            return mock_open(read_data=str(test_data_dir["study_dir"]))(*args, **kwargs)
        return original_open(*args, **kwargs)
    
    with patch('builtins.open', mock_open_file):
        # Run the main function
        _02_basics.main(path=str(test_data_dir["tmp_path"]))
        
        # Check output file exists
        volume_file = test_data_dir["results_dir"] / "volume_py_version.csv"
        assert volume_file.exists(), "Volume output file was not created"
        
        # Check content of the volume file
        with open(volume_file, "r") as f:
            content = f.read()
            
            # Verify header exists
            assert "subject,side,structure,volume_mm3" in content
            
            # Verify all subjects are included
            for subject in test_data_dir["subjects"]:
                assert subject in content
            
            # Verify all structures are included
            for structure in ["on", "ot", "oc", "onincr", "oninca", "oninor"]:
                assert structure in content
            
            # Verify both sides are included
            assert ",r," in content
            assert ",l," in content
            
            # Additional checks could verify the actual volume calculations
            # but this depends on the implementation details of _02_basics.py