import argparse
from textwrap import dedent
import time
import numpy as np
import logging
from pathlib import Path
import sentry_sdk

from avpy import _01_prep, _02_basics, _03a_normalize, \
    _03b_resample, _03c_normalize, _05_doatlas, _06_stats

logger = logging.getLogger(__name__)

# Define all available step modules with their names for better management
STEP_MODULES = {
    'prep': _01_prep,
    'basics': _02_basics,
    'normalize': _03a_normalize,
    'resample': _03b_resample,
    'normalize_stats': _03c_normalize,
    'atlas': _05_doatlas,
    'stats': _06_stats
}

def parse_steps_argument(steps_arg):
    """Parse the steps argument and return the list of step names to execute.
    
    Supports formats:
    - "all" - All steps
    - "step_name" - Single step
    - "step_name-end" - From step_name to the end
    - "step1-step2" - Range of steps from step1 to step2
    """
    available_steps = list(STEP_MODULES.keys())
    
    if steps_arg == "all":
        return available_steps
    
    # Check if it's a single step
    if steps_arg in available_steps:
        return [steps_arg]
    
    # Check for range formats
    if "-" in steps_arg:
        parts = steps_arg.split("-")
        
        # Handle "step-end" format
        if len(parts) == 2 and parts[0] in available_steps and parts[1] == "end":
            start_idx = available_steps.index(parts[0])
            return available_steps[start_idx:]
        
        # Handle "step1-step2" format
        if len(parts) == 2 and parts[0] in available_steps and parts[1] in available_steps:
            start_idx = available_steps.index(parts[0])
            end_idx = available_steps.index(parts[1])
            if start_idx <= end_idx:  # Ensure valid range
                return available_steps[start_idx:end_idx+1]
    
    # Invalid format, return all steps as default and log a warning
    logger.warning(f"Invalid steps format: '{steps_arg}'. Using 'all' instead.")
    return available_steps

def main():
    """Main entry point for aVP-toolbox."""
    sentry_sdk.init(
        dsn="https://f2866916959e41bc81abdfaf580f3d26@o252224.ingest.us.sentry.io/1439199",
        # Add request headers and IP for users
        send_default_pii=True,
    )

    parser = argparse.ArgumentParser(description="aVP-toolbox: Analysis tools for optic nerve processing")
    parser.add_argument("config", nargs="?", default=None)
    parser.add_argument(
        "--root-dir",
        dest="root_dir",
        default="./",
        help="Directory of the data to process.",
    )
    parser.add_argument(
        "--steps",
        dest="steps",
        default="all",
        help=dedent(
            """\
        The processing steps to run:
        - 'all': Run all processing steps
        - 'step_name': Run only a specific step (e.g., 'prep', 'basics', etc.)
        - 'step_name-end': Run from specified step to the end
        - 'step1-step2': Run a range of steps from step1 to step2
        
        Available steps: prep, basics, normalize, resample, normalize_stats, atlas, stats"""
        ),
    )
    parser.add_argument(
        "--deriv_root",
        dest="deriv_root",
        default="./",
        help=dedent(
            """\
        The root of the derivatives directory
        in which the pipeline will store the processing results.
        If unspecified, this will be derivatives/mne-bids-pipeline
        inside the BIDS root."""
        ),
    )
    parser.add_argument(
        "--test", dest="test", default=None, help="The subject to process. \
        Currently not used in this script."
    )
    parser.add_argument(
        "--dataset-A", dest="dataset_a", default=None, help="The folder in which \
            the processed data of first dataset is."
    )
    parser.add_argument(
        "--dataset-B", dest="dataset_b", default=None, help="The folder in which \
            the processed data of second dataset is."
    )

    options = parser.parse_args()
    
    # Convert paths to Path objects for better handling
    root_dir = Path(options.root_dir)
    deriv_root = Path(options.deriv_root)
    
    # Parse the steps argument to determine which steps to run
    steps_to_run = parse_steps_argument(options.steps)
    logger.info(f"Will run steps: {', '.join(steps_to_run)}")
    
    try:
        # Special handling for stats which requires datasets
        if 'stats' in steps_to_run and (options.dataset_a is None or options.dataset_b is None):
            if len(steps_to_run) == 1:
                # Only stats was requested but datasets are missing
                raise ValueError("--dataset-A and --dataset-B must be provided to run the stats step.")
            else:
                # Remove stats from steps if datasets are missing
                logger.warning("Skipping 'stats' step because --dataset-A or --dataset-B is missing.")
                steps_to_run.remove('stats')
        
        # Run the processing steps
        for step_name in steps_to_run:
            if step_name == 'stats' and options.dataset_a and options.dataset_b:
                # Special case for stats which needs dataset arguments
                start = time.time()
                logger.info(f"Running {step_name}...")
                STEP_MODULES[step_name].main(
                    path=options.root_dir, 
                    dataset_a=options.dataset_a, 
                    dataset_b=options.dataset_b
                )
            else:
                # Standard processing steps
                step_module = STEP_MODULES[step_name]
                
                # Skip stats if datasets aren't provided
                if step_name == 'stats' and (not options.dataset_a or not options.dataset_b):
                    logger.warning(f"Skipping {step_name} because datasets are not provided.")
                    continue
                
                start = time.time()
                logger.info(f"Running {step_name}...")
                
                # Check if the module supports the new interface with main_folder and output_folder
                import inspect
                sig = inspect.signature(step_module.main)
                
                if 'main_folder' in sig.parameters and 'output_folder' in sig.parameters:
                    # New interface
                    step_module.main(main_folder=root_dir, output_folder=deriv_root)
                else:
                    # Legacy interface
                    step_module.main(options.root_dir)
                
            # Log execution time
            elapsed = time.time() - start
            hours, remainder = divmod(elapsed, 3600)
            hours = int(hours)
            minutes, seconds = divmod(remainder, 60)
            minutes = int(minutes)
            seconds = int(np.ceil(seconds))  # always take full seconds
            elapsed = f"{seconds}s"
            if minutes:
                elapsed = f"{minutes}m {elapsed}"
            if hours:
                elapsed = f"{hours}h {elapsed}"
            logger.info(f"Completed {step_name} ({elapsed})")
    
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"Error occurred: {e}")
        raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    main()

