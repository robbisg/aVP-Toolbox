import argparse
from textwrap import dedent
import time
import numpy as np
import logging
from . import _01_prep, _02_basics, _03a_normalize, \
    _03b_resample, _03c_normalize, _05_doatlas, _06_stats

logger = logging.getLogger(__name__)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=None)
    parser.add_argument(
        "--config",
        dest="config_switch",
        default=None,
        metavar="FILE",
        help="The path of the pipeline configuration file to use.",
    )
    parser.add_argument(
        "--steps",
        dest="steps",
        default="all",
        help=dedent(
            """\
        The processing steps to run.
        Can either be one of the processing groups 'preprocessing', sensor',
        'source', 'report',  or 'all',  or the name of a processing group plus
        the desired step sans the step number and
        filename extension, separated by a '/'. For example, to run ICA, you
        would pass 'sensor/run_ica`. If unspecified, will run all processing
        steps. Can also be a tuple of steps."""
        ),
    )
    parser.add_argument(
        "--root-dir",
        dest="root_dir",
        default="./",
        help="BIDS root directory of the data to process.",
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
    ),
    parser.add_argument(
        "--test", dest="test", default=None, help="The subject to process. \
        Currently not used in this script."
    )
    
    parser.add_argument(
        "--dataset-A", dest="dataset_a",  default=None, help="The folder in which \
            the processed data of first dataset is."
    )
    
    parser.add_argument(
        "--dataset-B", dest="dataset_b",  default=None, help="The folder in which \
            the processed data of first dataset is."
    )

    options = parser.parse_args()
    
    if options.dataset_b == options.dataset_a == None:
    
        step_modules = [
            _01_prep,
            _02_basics,
            _03a_normalize,
            _03b_resample,
            _03c_normalize,
            _05_doatlas
        ]
        
        for step_module in step_modules:
            start = time.time()
            logger.info(f"Running {step_module.NAME}...")
            step_module.main(options.root_dir)
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
            logger.info(f"done ({elapsed})")
            
    else:
        _06_stats.main(options.root_dir, 
                        options.dataset_a, 
                        options.dataset_b)