import os
import pytest
import nibabel as nib
import numpy as np
from unittest.mock import patch, mock_open
from pathlib import Path

from avpy import _03a_normalize

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
        
        # Create optic nerve segmentation files to normalize
        for structure in ["on", "ot", "oc", "onincr", "oninca", "oninor"]:
            for side in ["r", "l"]:
                # Create unique shapes for each structure-side pair to test normalization
                data = np.zeros((10, 10, 10))
                
                # Vary the shape based on subject and structure
                offset = 2 if subject == "sub01" else 4
                length = 5 if subject == "sub01" else 3
                
                # Make each structure have a different spatial position
                if structure == "on":
                    data[offset:offset+length, 3:6, 3:6] = 1
                elif structure == "ot":
                    data[3:6, offset:offset+length, 3:6] = 16
                elif structure == "oc":
                    data[3:6, 3:6, offset:offset+length] = 8
                elif structure == "onincr":
                    data[offset:offset+length, 2:5, 2:5] = 4
                elif structure == "oninca":
                    data[2:5, offset:offset+length, 2:5] = 2
                elif structure == "oninor":
                    data[2:5, 2:5, offset:offset+length] = 1
                
                img = nib.Nifti1Image(data, np.eye(4))
                nib.save(img, subj_dir / f"{structure}_{side}.nii.gz")
    
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
    """Test the main function of _03a_normalize.py"""
    # Mock the open function for ONcontrol.txt
    original_open = open
    
    def mock_open_file(*args, **kwargs):
        if args[0] == os.path.join(str(test_data_dir["tmp_path"]), 'ONcontrol.txt') and args[1] == 'r':
            return mock_open(read_data=str(test_data_dir["study_dir"]))(*args, **kwargs)
        return original_open(*args, **kwargs)
    
    with patch('builtins.open', mock_open_file):
        # Run the main function
        _03a_normalize.main(path=str(test_data_dir["tmp_path"]))
        
        # Check for normalized outputs for each subject
        for subject in test_data_dir["subjects"]:
            subj_dir = test_data_dir["proc_dir"] / subject
            
            # Check for normalized files
            for structure in ["on", "ot", "oc"]:
                for side in ["r", "l"]:
                    # Each structure should have a normalized version
                    normalized_file = subj_dir / f"{structure}_{side}_norm.nii.gz"
                    assert normalized_file.exists(), f"Normalized file {normalized_file} was not created"
                    
                    # Load the original and normalized files to check normalization was applied
                    original = nib.load(subj_dir / f"{structure}_{side}.nii.gz")
                    normalized = nib.load(normalized_file)
                    
                    # Normalized image should maintain the same overall shape
                    assert original.shape == normalized.shape
                    
                    # Check that the data has been properly normalized
                    orig_data = original.get_fdata()
                    norm_data = normalized.get_fdata()
                    
                    if np.sum(orig_data) > 0:  # Only check if there's actual data
                        # Normalization should maintain non-zero values
                        assert np.sum(norm_data > 0) > 0
                        
                        # If normalization preserves spatial location:
                        nonzero_orig = np.where(orig_data > 0)
                        nonzero_norm = np.where(norm_data > 0)
                        # The non-zero voxels should be in the same locations
                        for i in range(3):  # Check x, y, z dimensions
                            np.testing.assert_array_equal(nonzero_orig[i], nonzero_norm[i])