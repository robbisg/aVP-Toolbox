#!/usr/bin/env python3
"""
Test runner script for aVP-Toolbox.

This script provides an easy way to run the test suite with different configurations.
"""

import subprocess
import sys
import os


def run_tests(test_type="all", verbose=True, coverage=False):
    """
    Run the test suite.
    
    Args:
        test_type (str): Type of tests to run ("all", "unit", "integration", "fast")
        verbose (bool): Enable verbose output
        coverage (bool): Enable coverage reporting
    """
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=avpy", "--cov-report=html", "--cov-report=term"])
    
    # Select tests based on type
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration": 
        cmd.extend(["-m", "integration"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    # "all" runs everything by default
    
    # Add test directory
    cmd.append("tests/")
    
    print(f"Running command: {' '.join(cmd)}")
    return subprocess.call(cmd)


def main():
    """Main function to handle command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run aVP-Toolbox tests")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "fast"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true", 
        help="Enable coverage reporting"
    )
    
    args = parser.parse_args()
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("Error: pytest is not installed. Install it with: pip install pytest")
        sys.exit(1)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run tests
    exit_code = run_tests(
        test_type=args.type,
        verbose=not args.quiet,
        coverage=args.coverage
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()