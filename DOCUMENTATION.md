# aVP-Toolbox Documentation

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Processing Pipeline](#processing-pipeline)
3. [API Reference](#api-reference)
4. [File Formats](#file-formats)
5. [Configuration](#configuration)
6. [Error Handling](#error-handling)
7. [Performance Optimization](#performance-optimization)

## Architecture Overview

The aVP-Toolbox follows a modular architecture with distinct processing steps:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Input    │───▶│   Processing    │───▶│   Results       │
│                 │    │   Pipeline      │    │                 │
│ • NIfTI files   │    │ • Step modules  │    │ • Statistics    │
│ • Subject lists │    │ • FSL tools     │    │ • Templates     │
│ • Config files  │    │ • MATLAB tools  │    │ • Visualizations│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

#### 1. Main Entry Point (`main.py`)
- Command-line interface
- Step execution orchestration
- Error handling and logging
- Configuration management

#### 2. Processing Modules
Each step is implemented as a separate module:

- `_01_prep.py`: Segmentation preprocessing
- `_02_basics.py`: Basic morphometric calculations  
- `_03a_normalize.py`: Image normalization
- `_03b_resample.py`: Image resampling
- `_03c_normalize.py`: Statistical normalization
- `_05_doatlas.py`: Atlas generation
- `stats.py`: Statistical comparisons

#### 3. Utilities (`utils.py`)
- Memory profiling
- Performance monitoring
- Common helper functions

## Processing Pipeline

### Step 1: Preparation (`prep`)

**Purpose**: Break down manual segmentations into anatomical components

**Input**: 
- Raw segmentation files (onc.nii.gz, onl.nii.gz, onr.nii.gz, otl.nii.gz, otr.nii.gz)

**Process**:
```python
def apply_threshold(img_path, threshold_min, threshold_max, binary=True, multiplier=1):
    """Apply intensity thresholds to extract specific anatomical regions"""
    # Load image and apply spatial corrections
    # Extract regions based on intensity values:
    # iOrb: 2, iCan: 4, iCran: 6 (for nerves)
    # Hemichiasm: 8-9 (for chiasm)
```

**Output**:
- Individual component masks (oc_l.nii.gz, oc_r.nii.gz, etc.)
- Anatomical subdivision masks (oninca_l.nii.gz, oninor_l.nii.gz, etc.)

### Step 2: Basic Statistics (`basics`)

**Purpose**: Calculate fundamental morphometric measurements

**Process**:
1. Count non-zero voxels in each segmentation
2. Calculate volumes using voxel dimensions
3. Generate summary statistics across subjects

**Output**:
- `raw_stats.xlsx`: Voxel counts and volumes per anatomical region

### Step 3: Normalization (`normalize`)

**Purpose**: Standardize optic nerve representations along their longitudinal axis

**Key Algorithm - Linearization**:
```python
def linearize_optic_nerve(nerve_data):
    """
    Transform curved optic nerve into straight representation
    while preserving cross-sectional area
    """
    # 1. Find nerve centerline using skeletonization
    # 2. Calculate perpendicular cross-sections
    # 3. Preserve CSA while straightening
    # 4. Output linearized representation
```

**Process Flow**:
1. Load nerve segmentations
2. Apply 4-connectivity morphological operations
3. Calculate centerline and cross-sections
4. Generate linearized representations
5. Create isotropic resampled versions

**Output**:
- `*_linearize_4bc.nii.gz`: Linearized nerve representations
- `*_normalized_4bc.nii.gz`: Normalized versions
- `*_iso06.nii.gz`: 0.6mm isotropic versions

### Step 4: Resampling (`resample`)

**Purpose**: Standardize spatial resolution across subjects

**Process**:
- Resample all images to 0.6mm³ isotropic resolution
- Maintain spatial alignment
- Preserve anatomical relationships

### Step 5: Statistical Analysis (`normalize_stats`)

**Purpose**: Extract comprehensive morphometric measurements

**Measurements Extracted**:
- **Volume**: Total volume per anatomical region
- **Length**: Longitudinal extent of nerve segments  
- **Cross-Sectional Area (CSA)**: Area measurements along nerve length
- **Major/Minor Axis**: Principal component analysis of cross-sections
- **Shape descriptors**: Roundness, aspect ratio

**Output Files**:
- `aVP_slice_data.xlsx`: Slice-by-slice measurements
- `aVP_section_CSA_length.xlsx`: Sectional analysis
- `log_check_*.xlsx`: Quality control metrics

### Step 6: Atlas Generation (`atlas`)

**Purpose**: Create probabilistic templates from multiple subjects

**Algorithm**:
```python
def generate_probabilistic_atlas(subject_images):
    """
    Create probability maps from normalized subject data
    """
    # 1. Register all subjects to common space
    # 2. Calculate voxel-wise probabilities
    # 3. Apply smoothing and thresholding
    # 4. Generate separate L/R templates
```

**Output**:
- `aVP_prob.nii.gz`: Combined probability template
- `aVP_prob_l.nii.gz`: Left hemisphere template
- `aVP_prob_r.nii.gz`: Right hemisphere template  
- Anatomical subdivision templates (iOrb_prob.nii.gz, etc.)

## API Reference

### Main Interface

```python
def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="aVP-toolbox")
    parser.add_argument("--root-dir", help="Data directory")
    parser.add_argument("--steps", help="Processing steps to run")
    parser.add_argument("--debug", action="store_true")
```

### Step Modules

Each processing step follows a consistent interface:

```python
def main(main_folder: str, output_folder: str = None, debug: bool = False):
    """
    Args:
        main_folder: Path to input data directory
        output_folder: Path to output directory (optional)
        debug: Enable verbose logging
    
    Returns:
        None (writes output files)
    """
```

### Utility Functions

```python
# Memory profiling
@memory_report
def process_function():
    """Decorator for monitoring memory usage"""

# Performance monitoring  
def get_memory_usage() -> float:
    """Return current memory usage in MB"""

def log_memory_usage(label: str):
    """Log memory usage with timestamp"""
```

## File Formats

### Input Data Structure
```
study_folder/
├── data/
│   ├── orig/
│   │   └── subject_001/
│   │       ├── onc.nii.gz    # Optic chiasm segmentation
│   │       ├── onl.nii.gz    # Left optic nerve
│   │       ├── onr.nii.gz    # Right optic nerve  
│   │       ├── otl.nii.gz    # Left optic tract
│   │       └── otr.nii.gz    # Right optic tract
│   ├── proc/                 # Processing intermediates
│   └── sbj.list             # Subject ID list
├── results/                 # Output directory
├── templates/              # Atlas outputs
└── logs/                   # Processing logs
```

### Segmentation File Specifications

**File Format**: NIfTI (.nii.gz)
**Voxel Size**: 0.6 × 0.6 × 0.6 mm
**Data Type**: Integer intensity values

**Intensity Coding**:
```
Optic Nerves (onl.nii.gz, onr.nii.gz):
├── Intraorbital (iOrb): 2
├── Intracanalicular (iCan): 4  
└── Intracranial (iCran): 6

Optic Chiasm (onc.nii.gz):
├── Right hemichiasm: 8
└── Left hemichiasm: 9

Optic Tracts (otl.nii.gz, otr.nii.gz):
└── Full tract: 1
```

### Output Files

**Excel Spreadsheets**:
- Subject-wise measurements with statistical summaries
- Quality control metrics
- Cross-sectional analyses

**NIfTI Templates**:
- Probabilistic atlases with floating-point probability values (0-1)
- Binary masks at different probability thresholds

## Configuration

### Subject List Format (`sbj.list`)
```
001
002
003
...
```

### Environment Variables
```bash
export FSLDIR=/usr/local/fsl    # FSL installation directory
export MATLABPATH=/path/to/matlab/toolboxes  # MATLAB toolbox paths
```

### Processing Parameters

Default parameters can be modified in respective modules:

```python
# Resampling resolution
TARGET_RESOLUTION = 0.6  # mm

# Atlas probability thresholds  
PROB_THRESHOLDS = [0.5, 1.0]

# Morphological operations
CONNECTIVITY = 4  # 4-connectivity for operations
```

## Error Handling

### Common Error Types

1. **FileNotFoundError**: Missing input files
   ```python
   if not os.path.exists(input_file):
       logger.error(f"Input file not found: {input_file}")
       raise FileNotFoundError(f"Required file {input_file} does not exist")
   ```

2. **MemoryError**: Insufficient RAM for large datasets
   ```python
   try:
       data = load_large_dataset()
   except MemoryError:
       logger.warning("Memory exhausted, processing in chunks")
       data = load_dataset_chunked()
   ```

3. **ValueError**: Invalid segmentation intensities
   ```python
   unique_vals = np.unique(segmentation_data)
   expected_vals = [2, 4, 6]
   if not set(unique_vals).issubset(expected_vals):
       raise ValueError(f"Invalid intensities: {unique_vals}")
   ```

### Logging Configuration

```python
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Enable debug logging
if debug:
    logging.basicConfig(level=logging.DEBUG)
```

### Error Recovery

```python
def robust_processing(func):
    """Decorator for error recovery in processing steps"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            # Attempt recovery or graceful degradation
            return None
    return wrapper
```

## Performance Optimization

### Memory Management

1. **Chunked Processing**:
   ```python
   def process_large_dataset(data_path, chunk_size=100):
       """Process data in chunks to manage memory"""
       for i in range(0, len(subjects), chunk_size):
           chunk = subjects[i:i+chunk_size]
           process_chunk(chunk)
           gc.collect()  # Force garbage collection
   ```

2. **Memory Monitoring**:
   ```python
   @memory_report
   def memory_intensive_function():
       # Processing code
       pass
   ```

### Parallel Processing

```python
from joblib import Parallel, delayed

def parallel_subject_processing(subjects, n_jobs=-1):
    """Process subjects in parallel"""
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_subject)(subject) for subject in subjects
    )
    return results
```

### I/O Optimization

1. **Lazy Loading**:
   ```python
   def lazy_load_images(file_paths):
       """Load images only when needed"""
       for path in file_paths:
           yield nib.load(path)
   ```

2. **Efficient File Formats**:
   - Use compressed NIfTI (.nii.gz) for storage
   - Consider HDF5 for large numerical arrays
   - Cache intermediate results

### Profiling Tools

```python
import cProfile
import pstats

def profile_function(func):
    """Profile function execution"""
    profiler = cProfile.Profile()
    profiler.enable()
    result = func()
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats()
    return result
```

## Troubleshooting Guide

### Common Issues and Solutions

1. **"FSL command not found"**
   - Ensure FSL is installed and in PATH
   - Source FSL configuration: `source $FSLDIR/etc/fslconf/fsl.sh`

2. **"MATLAB toolbox missing"**
   - Install required toolboxes: xlwrite, nifti_tools
   - Add toolbox paths to MATLAB path

3. **"Memory allocation failed"**
   - Reduce batch size or process subjects individually
   - Close other memory-intensive applications
   - Consider processing on higher-memory system

4. **"Invalid segmentation values"**
   - Check segmentation intensity values match expected ranges
   - Verify file format is correct (NIfTI .nii.gz)
   - Ensure no floating-point values in integer segmentations

5. **"Atlas generation fails"**
   - Verify all subjects have completed normalization step
   - Check that normalized images exist in expected locations
   - Ensure sufficient subjects for meaningful atlas (minimum 10-15)

For additional support, consult the original research publication and example datasets provided in the repository.