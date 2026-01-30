import os
import pytest
import importlib
import argparse
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

from avpy import main

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
        orig_subj_dir = orig_dir / subject
        orig_subj_dir.mkdir(parents=True)
        proc_subj_dir = proc_dir / subject
        proc_subj_dir.mkdir(parents=True)
    
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
        "orig_dir": orig_dir,
        "proc_dir": proc_dir,
        "results_dir": results_dir,
        "subjects": subjects,
        "control_file": control_file
    }

def test_argument_parsing():
    """Test argument parsing in main.py"""
    # Save original sys.argv
    original_argv = sys.argv.copy()
    
    try:
        # Test with different command line arguments
        test_args = ["avpy", "--root-dir", "/test/root/dir", "--steps", "all"]
        sys.argv = test_args
        
        # Reset the ArgumentParser to avoid previous state
        importlib.reload(main)
        
        # Patch the main's argparse.ArgumentParser to capture arguments
        with patch('argparse.ArgumentParser.parse_args', 
                  return_value=argparse.Namespace(
                      config=None,
                      config_switch=None,
                      steps="all",
                      root_dir="/test/root/dir",
                      deriv_root="./",
                      test=None,
                      dataset_a=None,
                      dataset_b=None
                  )):
            
            # Patch all processing modules
            with patch('avpy._01_prep.main') as mock_prep, \
                 patch('avpy._02_basics.main') as mock_basics, \
                 patch('avpy._03a_normalize.main') as mock_normalize_a, \
                 patch('avpy._03b_resample.main') as mock_resample, \
                 patch('avpy._03c_normalize.main') as mock_normalize_c, \
                 patch('avpy._05_doatlas.main') as mock_doatlas, \
                 patch('avpy._06_stats.main') as mock_stats:
                
                # Run main function
                main.main()
                
                # Check that all module mains were called
                mock_prep.assert_called_once()
                mock_basics.assert_called_once()
                mock_normalize_a.assert_called_once()
                mock_resample.assert_called_once()
                mock_normalize_c.assert_called_once()
                mock_doatlas.assert_called_once()
                
                # Stats shouldn't be called since dataset_a and dataset_b are None
                mock_stats.assert_not_called()
    
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

def test_main_with_datasets():
    """Test main function when datasets are provided"""
    # Save original sys.argv
    original_argv = sys.argv.copy()
    
    try:
        # Test with dataset arguments
        test_args = ["avpy", "--dataset-A", "/path/to/datasetA", "--dataset-B", "/path/to/datasetB"]
        sys.argv = test_args
        
        # Reset the ArgumentParser to avoid previous state
        importlib.reload(main)
        
        # Patch the main's argparse.ArgumentParser to capture arguments
        with patch('argparse.ArgumentParser.parse_args', 
                  return_value=argparse.Namespace(
                      config=None,
                      config_switch=None,
                      steps="all",
                      root_dir="./",
                      deriv_root="./",
                      test=None,
                      dataset_a="/path/to/datasetA",
                      dataset_b="/path/to/datasetB"
                  )):
            
            # Patch _06_stats.main specifically
            with patch('avpy._06_stats.main') as mock_stats:
                
                # Run main function
                main.main()
                
                # Check that stats was called with the correct arguments
                mock_stats.assert_called_once_with("./", "/path/to/datasetA", "/path/to/datasetB")
    
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

def test_pipeline_integration(test_data_dir):
    """Test full pipeline integration"""
    # Mock all module functions
    module_mocks = {}
    for module_name in ['_01_prep', '_02_basics', '_03a_normalize', '_03b_resample', '_03c_normalize', '_05_doatlas']:
        mock = MagicMock()
        module_mocks[module_name] = mock
        
        # Add the module name attribute
        mock.NAME = module_name.replace('_', '')
    
    # Patch all modules
    with patch.dict('sys.modules', {
        'avpy._01_prep': module_mocks['_01_prep'],
        'avpy._02_basics': module_mocks['_02_basics'],
        'avpy._03a_normalize': module_mocks['_03a_normalize'],
        'avpy._03b_resample': module_mocks['_03b_resample'],
        'avpy._03c_normalize': module_mocks['_03c_normalize'],
        'avpy._05_doatlas': module_mocks['_05_doatlas'],
        'avpy._06_stats': MagicMock()
    }):
        # Reimport main with mocked modules
        importlib.reload(main)
        
        # Patch argument parsing
        with patch('argparse.ArgumentParser.parse_args', 
                  return_value=argparse.Namespace(
                      config=None,
                      config_switch=None,
                      steps="all",
                      root_dir=str(test_data_dir["study_dir"]),
                      deriv_root=str(test_data_dir["proc_dir"]),
                      test=None,
                      dataset_a=None,
                      dataset_b=None
                  )):
            
            # Run main function
            main.main()
            
            # Check that each module's main function was called
            for module_name, mock_module in module_mocks.items():
                mock_module.main.assert_called_once()