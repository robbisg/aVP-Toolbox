# aVP-Toolbox: Anterior Visual Pathway Analysis Toolkit

![aVP-Toolbox Logo](aVP_Toolbox-logo.png)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A comprehensive toolkit for morphometric MRI analysis of the anterior visual pathway (aVP), including the optic nerve, chiasm, and optic tract.

## Overview

The aVP-Toolbox is an innovative methodology for conducting standardized morphometric analysis of the anterior visual pathway using high-resolution MRI. It provides tools for extracting quantitative biomarkers from the entire anterior optic pathway and enables interindividual comparisons through probabilistic templates.

### Key Features

- **Comprehensive Analysis**: Extract volume, length, cross-sectional area (CSA), and axis measurements from optic nerve subdivisions
- **Normalization**: Normalize segmentations along longitudinal axis while preserving nerve CSA
- **Atlas Generation**: Create probabilistic templates of the entire aVP and anatomical subdivisions
- **Multi-platform**: Supports Linux/Mac systems with Python and optional MATLAB components
- **Standardized Pipeline**: Step-by-step processing workflow for reproducible results

## Scientific Background

This project advances in-vivo knowledge of anterior visual pathway structure using dedicated high-resolution MRI sequences. The toolkit builds on the STIR-ZOOMit MRI sequence development, which provides sub-millimeter resolution imaging of the anterior visual pathway.

### Publication

If you use aVP-Toolbox in your research, please cite:

> Pravatà, E., Diociasi, A., Navarra, R. et al. Biometry extraction and probabilistic anatomical atlas of the anterior Visual Pathway using dedicated high-resolution 3-D MRI. Sci Rep 14, 453 (2024). https://doi.org/10.1038/s41598-023-50980-x

## Installation

### Prerequisites

- Python 3.9+
- FSL (FMRIB Software Library) for some processing steps
- Optional: MATLAB R2015b+ with Image Processing Toolbox

### Installing from Source

```bash
git clone https://github.com/username/aVP-toolbox.git
cd aVP-toolbox
pip install -e .
```

### Dependencies

The toolkit automatically installs required Python packages including:
- NumPy, SciPy, matplotlib
- nibabel (neuroimaging data)
- scikit-learn, scikit-image
- pandas, seaborn (data analysis)
- joblib (parallel processing)

## Quick Start

### 1. Prepare Your Data

Organize your data in the following structure:
```
study_folder/
├── data/
│   ├── orig/
│   │   └── subject_id/
│   │       ├── onc.nii.gz    # optic chiasm
│   │       ├── onl.nii.gz    # left optic nerve
│   │       ├── onr.nii.gz    # right optic nerve
│   │       ├── otl.nii.gz    # left optic tract
│   │       └── otr.nii.gz    # right optic tract
│   └── sbj.list              # list of subject IDs
└── results/                  # output directory
```

### 2. Run the Complete Pipeline

```bash
# Run all processing steps
avp_all --root-dir /path/to/study_folder

# Run specific steps
avp_all --root-dir /path/to/study_folder --steps prep-normalize

# Run with debug output
avp_all --root-dir /path/to/study_folder --debug
```

### 3. Available Processing Steps

- **prep**: Break down segmentations into anatomical components
- **basics**: Calculate basic volume and voxel statistics
- **normalize**: Generate linearized and normalized images
- **resample**: Resample images to standard resolution
- **normalize_stats**: Extract morphometric measurements
- **atlas**: Generate probabilistic templates
- **stats**: Compare datasets (requires two datasets)

## Data Requirements

### Image Specifications
- **Format**: NIfTI (.nii.gz)
- **Resolution**: 0.6mm³ isotropic voxels
- **Orientation**: Axial-oriented images
- **Consistency**: All subjects must have identical image dimensions

### Segmentation Labels
Segmentations must use specific intensity values:

**Optic Nerves (onl.nii.gz, onr.nii.gz)**:
- Intraorbital (iOrb): 2
- Intracanalicular (iCan): 4  
- Intracranial (iCran): 6

**Optic Chiasm (onc.nii.gz)**:
- Right hemichiasm: 8
- Left hemichiasm: 9

## Processing Pipeline

### Step-by-Step Workflow

1. **Data Preparation** (`prep`):
   - Break down segmentations into anatomical components
   - Create unified segmentation masks

2. **Basic Statistics** (`basics`):
   - Calculate voxel counts and volumes
   - Generate initial morphometric data

3. **Normalization** (`normalize`, `resample`):
   - Linearize optic nerve structures
   - Normalize along longitudinal axis
   - Preserve cross-sectional area

4. **Atlas Generation** (`atlas`):
   - Create probabilistic templates
   - Generate subdivision probability masks
   - Output left, right, and combined templates

### Example Commands

```bash
# Process single subject through preparation
avp_all --root-dir ./study --steps prep

# Run normalization and resampling
avp_all --root-dir ./study --steps normalize-resample

# Generate probabilistic atlas
avp_all --root-dir ./study --steps atlas

# Compare two datasets
avp_all --root-dir ./study --steps stats --dataset-A ./dataset1 --dataset-B ./dataset2
```

## Output Files

### Individual Subject Results
- **Normalized images**: `*_normalized_4bc_iso06.nii.gz`
- **Linearized images**: `*_linearize_4bc_iso06.nii.gz`
- **Morphometric data**: Excel files with CSA, length, and axis measurements

### Group-Level Results
- **Probabilistic atlas**: `aVP_prob.nii.gz`
- **Anatomical subdivisions**: `iOrb_prob.nii.gz`, `iCan_prob.nii.gz`, etc.
- **Summary statistics**: Aggregated morphometric measurements

## Advanced Usage

### Custom Processing

```python
from avpy import _01_prep, _02_basics, _03a_normalize

# Run individual steps programmatically
_01_prep.main(main_folder="./data", output_folder="./results")
_02_basics.main("./study_folder", debug=True)
```

### Memory Management

For large datasets, the toolkit includes memory monitoring:

```python
from avpy.utils import memory_report

@memory_report
def process_large_dataset():
    # Your processing code here
    pass
```

## Atlas Information

The repository includes the **aVP-24** probabilistic atlas, generated from 24 healthy volunteers using 3T MRI. This atlas provides:

- Template images at 50% and 100% probability thresholds
- Anatomical subdivision probability masks
- Normalized individual subject data used for atlas construction

## Troubleshooting

### Common Issues

1. **Memory errors**: Use smaller batch sizes or process subjects individually
2. **FSL not found**: Ensure FSL is installed and in your PATH
3. **MATLAB errors**: Check that required toolboxes are installed
4. **File permissions**: Ensure scripts are executable: `chmod +x *.sh`

### Getting Help

- Check the [instructions document](code/v0.11/aVP_instructions_v0.11.txt) for detailed setup
- Review example data structure in the `data/test` directory
- Consult the original publication for methodological details

## Development

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=avpy
```

## License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). You are free to:
- **Share**: copy and redistribute the material
- **Adapt**: remix, transform, and build upon the material

See the [LICENSE](LICENSE) file for full details.

## Credits

**Development Team:**
- **Emanuele Pravatà** - Principal Investigator, Neurocenter of Southern Switzerland
- **Roberto Guidotti** - Python implementation
- **Riccardo Navarra** - Methodological development
- **Luca Roccatagliata** - Collaboration, University of Genoa
- **Andrea Diociasi** - Collaboration, University of Genoa
- **Paul Summers** - Collaboration, European Institute of Oncology

**Funding:**
- Velux Stiftung
- Swiss Society of Multiple Sclerosis
- Advisory Board for Research of the Ente Ospedaliero Cantonale del Ticino (ABREOC)

## Version History

- **v0.11** (2023-12): Current version with Python implementation
- **v0.1** (2023-12): Initial release with MATLAB/Bash implementation

---

For detailed usage instructions, see [aVP_instructions_v0.11.txt](code/v0.11/aVP_instructions_v0.11.txt)