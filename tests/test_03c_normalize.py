import os
import pytest
import nibabel as nib
import numpy as np
import pandas as pd
from unittest.mock import patch, mock_open
from pathlib import Path

from avpy import _03c_normalize

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
        
        # Create resampled files for statistics calculation
        for structure in ["on", "ot", "oc", "onincr", "oninca", "oninor"]:
            for side in ["r", "l"]:
                # Create data with different patterns for linear/normalize analysis
                data = np.zeros((10, 10, 10))
                
                # Different shapes for each subject
                start = 1 if subject == "sub01" else 3
                length = 6 if subject == "sub01" else 4
                
                # Create a more complex tubular structure along z-axis
                # with varying radius to test cross-sectional area calculations
                for z in range(start, start + length):
                    radius = 2 if z < start + length//2 else 1
                    for x in range(4-radius, 7+radius):
                        for y in range(4-radius, 7+radius):
                            # Make a rough circle
                            dist = np.sqrt((x-5)**2 + (y-5)**2)
                            if dist <= radius:
                                data[x, y, z] = 1
                
                # Save both resampled and linearized versions
                img = nib.Nifti1Image(data, np.eye(4))
                nib.save(img, subj_dir / f"{structure}_{side}_resampled.nii.gz")
                
                # Create linearized version with the same data but different file name
                nib.save(img, subj_dir / f"{structure}_{side}_linearize.nii.gz")
    
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
    """Test the main function of _03c_normalize.py"""
    # Mock the open function for ONcontrol.txt
    original_open = open
    
    def mock_open_file(*args, **kwargs):
        if args[0] == os.path.join(str(test_data_dir["tmp_path"]), 'ONcontrol.txt') and args[1] == 'r':
            return mock_open(read_data=str(test_data_dir["study_dir"]))(*args, **kwargs)
        return original_open(*args, **kwargs)
    
    with patch('builtins.open', mock_open_file):
        # Run the main function
        _03c_normalize.main(path=str(test_data_dir["tmp_path"]))
        
        # Check for output Excel files
        excel_files = [
            "aVP_slice_data_iso.xlsx",
            "aVP_section_CSA_length_iso.xlsx",
            "py_aVP_section_CSA_length.xlsx"
        ]
        
        for excel_file in excel_files:
            file_path = test_data_dir["results_dir"] / excel_file
            assert file_path.exists(), f"Excel output file {excel_file} was not created"
            
            # Check the Excel file content
            try:
                df = pd.read_excel(file_path)
                
                # Check that the expected data columns are present
                for subject in test_data_dir["subjects"]:
                    assert subject in df.values, f"Subject {subject} not found in {excel_file}"
                
                # Check for expected columns based on the specific file
                if "slice_data" in excel_file:
                    # This file should have slice-by-slice data
                    expected_columns = ["subject", "side", "slice", "area"]
                    for col in expected_columns:
                        assert any(col in column for column in df.columns), f"Column {col} not found in {excel_file}"
                
                elif "section_CSA_length" in excel_file:
                    # This file should have section-level data
                    expected_columns = ["subject", "side", "length", "CSA"]
                    for col in expected_columns:
                        assert any(col in column for column in df.columns), f"Column {col} not found in {excel_file}"
            
            except Exception as e:
                pytest.fail(f"Failed to read Excel file {excel_file}: {e}")