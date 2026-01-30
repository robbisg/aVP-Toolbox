import os
import shutil
import pytest
import nibabel as nib
import numpy as np
from unittest.mock import patch, mock_open
from pathlib import Path

from avpy import _01_prep

@pytest.fixture
def test_data_dir(tmp_path):
    """Setup test directory structure with test data"""
    # Create directory structure
    study_dir = tmp_path / "study"
    orig_dir = study_dir / "data" / "orig"
    orig_dir.mkdir(parents=True, exist_ok=True)
    proc_dir = study_dir / "data" / "proc"
    proc_dir.mkdir(parents=True, exist_ok=True)
    results_dir = study_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test subjects
    subjects = ["sub01", "sub02"]
    for subject in subjects:
        subj_dir = orig_dir / subject
        subj_dir.mkdir(parents=True)
        
        # Create test NIfTI files with specific values
        # Create on{r,l} files with different values for different segmentations
        for side in ['r', 'l']:
            on_data = np.zeros((10, 10, 10))
            on_data[1:3, 1:3, 1:3] = 2  # Value for oninor
            on_data[3:5, 3:5, 3:5] = 4  # Value for oninca
            on_data[5:7, 5:7, 5:7] = 6  # Value for onincr
            
            on_img = nib.Nifti1Image(on_data, np.eye(4))
            nib.save(on_img, subj_dir / f"on{side}.nii.gz")
            
            # Create ot files
            ot_data = np.zeros((10, 10, 10))
            ot_data[7:9, 7:9, 7:9] = 10  # Value for ot
            ot_img = nib.Nifti1Image(ot_data, np.eye(4))
            nib.save(ot_img, subj_dir / f"ot{side}.nii.gz")
        
        # Create onc file with values 8 for right and 9 for left
        onc_data = np.zeros((10, 10, 10))
        onc_data[2:4, 2:4, 7:9] = 8  # Right side (value 8)
        onc_data[6:8, 6:8, 7:9] = 9  # Left side (value 9)
        onc_img = nib.Nifti1Image(onc_data, np.eye(4))
        nib.save(onc_img, subj_dir / "onc.nii.gz")
    
    # Write study path to control file
    control_file = tmp_path / "ONcontrol.txt"
    with open(control_file, "w") as f:
        f.write(str(study_dir))
    
    yield {
        "tmp_path": tmp_path,
        "study_dir": study_dir,
        "orig_dir": orig_dir,
        "proc_dir": proc_dir,
        "results_dir": results_dir,
        "subjects": subjects,
        "control_file": control_file
    }

def test_apply_threshold():
    """Test the apply_threshold function"""
    # Create a test array with multiple values
    test_data = np.zeros((10, 10, 10))
    test_data[2:4, 2:4, 2:4] = 3
    test_data[5:7, 5:7, 5:7] = 5
    test_data[7:9, 7:9, 7:9] = 10
    
    test_img = nib.Nifti1Image(test_data, np.eye(4))
    
    # Save to a temporary file
    with pytest.raises(Exception):
        temp_file = os.path.join(pytest.TempDir(), "test_img.nii.gz")
    
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "test_img.nii.gz")
        nib.save(test_img, temp_file)
        
        # Test thresholding with exact values
        result1 = _01_prep.apply_threshold(temp_file, 3, 3, binary=True, multiplier=1)
        result_data1 = result1.get_fdata()
        expected1 = np.zeros((10, 10, 10))
        expected1[2:4, 2:4, 2:4] = 1
        np.testing.assert_array_equal(result_data1, expected1)
        
        # Test with a multiplier
        result2 = _01_prep.apply_threshold(temp_file, 5, 5, binary=True, multiplier=4)
        result_data2 = result2.get_fdata()
        expected2 = np.zeros((10, 10, 10))
        expected2[5:7, 5:7, 5:7] = 4
        np.testing.assert_array_equal(result_data2, expected2)
        
        # Test with a range of values
        result3 = _01_prep.apply_threshold(temp_file, 3, 5, binary=True, multiplier=2)
        result_data3 = result3.get_fdata()
        expected3 = np.zeros((10, 10, 10))
        expected3[2:4, 2:4, 2:4] = 2
        expected3[5:7, 5:7, 5:7] = 2
        np.testing.assert_array_equal(result_data3, expected3)

def test_main_function(test_data_dir):
    """Test the main function of _01_prep.py"""
    # Mock the open function for ONcontrol.txt
    with patch('builtins.open', mock_open(read_data=str(test_data_dir["study_dir"]))):
        # Run the main function with the path to the control file
        _01_prep.main(path=str(test_data_dir["tmp_path"]))
        
        # Check that the subject list was created
        sbj_list_path = test_data_dir["study_dir"] / "data" / "sbj.list"
        assert sbj_list_path.exists(), "Subject list file was not created"
        
        # Read the subject list
        with open(sbj_list_path, "r") as f:
            subjects = [line.strip() for line in f.readlines()]
        
        # Check all subjects are in the list
        for subject in test_data_dir["subjects"]:
            assert subject in subjects, f"Subject {subject} is missing from the list"
        
        # Check that output files were created for each subject
        for subject in subjects:
            subj_out_dir = test_data_dir["proc_dir"] / subject
            assert subj_out_dir.exists(), f"Output directory for {subject} was not created"
            
            # Check specific files
            expected_files = [
                # OT files
                "ot_r.nii.gz", "ot_l.nii.gz", 
                # OC files
                "oc_r.nii.gz", "oc_l.nii.gz", 
                # ON component files
                "oninor_r.nii.gz", "oninor_l.nii.gz",
                "oninca_r.nii.gz", "oninca_l.nii.gz",
                "onincr_r.nii.gz", "onincr_l.nii.gz",
                # Combined ON files
                "on_r.nii.gz", "on_l.nii.gz"
            ]
            
            for file in expected_files:
                file_path = subj_out_dir / file
                assert file_path.exists(), f"Expected file {file} was not created for {subject}"
            
            # Verify the content of combined files
            for side in ['r', 'l']:
                # Load component files
                ot = nib.load(subj_out_dir / f"ot_{side}.nii.gz").get_fdata()
                onincr = nib.load(subj_out_dir / f"onincr_{side}.nii.gz").get_fdata()
                oninca = nib.load(subj_out_dir / f"oninca_{side}.nii.gz").get_fdata()
                oninor = nib.load(subj_out_dir / f"oninor_{side}.nii.gz").get_fdata()
                oc = nib.load(subj_out_dir / f"oc_{side}.nii.gz").get_fdata()
                
                # Load combined file
                combined = nib.load(subj_out_dir / f"on_{side}.nii.gz").get_fdata()
                
                # Check that combined file is the sum of components
                expected_combined = ot + onincr + oninca + oninor + oc
                np.testing.assert_array_equal(combined, expected_combined)
                
                # Check specific values
                # OT should have value 16 where original had value 10
                assert np.max(ot) == 16
                # ONINCR should have value 4 where original had value 6
                assert np.max(onincr) == 4
                # ONINCA should have value 2 where original had value 4
                assert np.max(oninca) == 2
                # ONINOR should have value 1 where original had value 2
                assert np.max(oninor) == 1
                # OC should have value 8 
                assert np.max(oc) == 8