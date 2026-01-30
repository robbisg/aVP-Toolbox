# aVP-Toolbox Examples

This document provides practical examples for using the aVP-Toolbox in various scenarios.

## Table of Contents

1. [Basic Usage Examples](#basic-usage-examples)
2. [Single Subject Processing](#single-subject-processing)
3. [Batch Processing](#batch-processing)
4. [Atlas Generation](#atlas-generation)
5. [Statistical Analysis](#statistical-analysis)
6. [Custom Scripts](#custom-scripts)
7. [Visualization Examples](#visualization-examples)
8. [Troubleshooting Examples](#troubleshooting-examples)

## Basic Usage Examples

### Example 1: Complete Pipeline Processing

Process a full study from raw segmentations to final atlas:

```bash
# Set up your study directory
export STUDY_DIR="/path/to/your/study"
cd $STUDY_DIR

# Ensure proper directory structure exists
mkdir -p data/orig data/proc results templates logs

# Create subject list
echo -e "001\n002\n003\n004\n005" > data/sbj.list

# Run complete pipeline
avp_all --root-dir $STUDY_DIR --steps all --debug
```

### Example 2: Step-by-Step Processing

Process data step by step for better control:

```bash
# Step 1: Prepare segmentations
avp_all --root-dir $STUDY_DIR --steps prep

# Step 2: Calculate basic statistics  
avp_all --root-dir $STUDY_DIR --steps basics

# Step 3: Normalize images
avp_all --root-dir $STUDY_DIR --steps normalize

# Step 4: Resample to standard resolution
avp_all --root-dir $STUDY_DIR --steps resample

# Step 5: Extract morphometric statistics
avp_all --root-dir $STUDY_DIR --steps normalize_stats

# Step 6: Generate probabilistic atlas
avp_all --root-dir $STUDY_DIR --steps atlas
```

### Example 3: Processing Specific Step Ranges

```bash
# Run from prep to normalize
avp_all --root-dir $STUDY_DIR --steps prep-normalize

# Run from resample to end
avp_all --root-dir $STUDY_DIR --steps resample-end

# Run only atlas generation
avp_all --root-dir $STUDY_DIR --steps atlas
```

## Single Subject Processing

### Example 4: Process Single Subject with Python API

```python
#!/usr/bin/env python3
import os
from pathlib import Path
from avpy import _01_prep, _02_basics, _03a_normalize

def process_single_subject(study_dir, subject_id):
    """Process a single subject through the complete pipeline"""
    
    # Set up paths
    study_path = Path(study_dir)
    subject_path = study_path / "data" / "orig" / subject_id
    
    # Check if subject data exists
    required_files = ['onc.nii.gz', 'onl.nii.gz', 'onr.nii.gz', 'otl.nii.gz', 'otr.nii.gz']
    for file in required_files:
        if not (subject_path / file).exists():
            raise FileNotFoundError(f"Missing {file} for subject {subject_id}")
    
    print(f"Processing subject: {subject_id}")
    
    # Create temporary subject list for single subject
    temp_sbj_list = study_path / "data" / "temp_sbj.list"
    with open(temp_sbj_list, 'w') as f:
        f.write(f"{subject_id}\n")
    
    try:
        # Step 1: Preparation
        print("- Running preparation step...")
        _01_prep.main(main_folder=str(study_path), output_folder=str(study_path))
        
        # Step 2: Basic statistics
        print("- Running basic statistics...")
        _02_basics.main(str(study_path), debug=True)
        
        # Step 3: Normalization
        print("- Running normalization...")
        _03a_normalize.main(str(study_path), debug=True)
        
        print(f"✓ Successfully processed subject {subject_id}")
        
    finally:
        # Clean up temporary file
        if temp_sbj_list.exists():
            temp_sbj_list.unlink()

# Usage
if __name__ == "__main__":
    study_directory = "/path/to/study"
    subject_to_process = "001"
    
    process_single_subject(study_directory, subject_to_process)
```

### Example 5: Quality Control for Single Subject

```python
#!/usr/bin/env python3
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def quality_check_subject(study_dir, subject_id):
    """Perform quality control checks on processed subject data"""
    
    study_path = Path(study_dir)
    proc_path = study_path / "data" / "proc" / subject_id
    
    # Check if processing outputs exist
    qc_results = {
        'subject_id': subject_id,
        'files_exist': {},
        'volume_stats': {},
        'intensity_check': {}
    }
    
    # Expected output files
    expected_files = [
        'onl_normalized_4bc_iso06.nii.gz',
        'onr_normalized_4bc_iso06.nii.gz',
        'oc_l.nii.gz', 'oc_r.nii.gz',
        'on_l.nii.gz', 'on_r.nii.gz'
    ]
    
    for file in expected_files:
        file_path = proc_path / file
        qc_results['files_exist'][file] = file_path.exists()
        
        if file_path.exists():
            # Load and check image
            img = nib.load(file_path)
            data = img.get_fdata()
            
            # Basic statistics
            qc_results['volume_stats'][file] = {
                'shape': data.shape,
                'voxel_count': np.sum(data > 0),
                'volume_mm3': np.sum(data > 0) * np.prod(img.header.get_zooms()),
                'max_intensity': np.max(data),
                'min_intensity': np.min(data[data > 0]) if np.any(data > 0) else 0
            }
            
            # Check for expected intensity values
            unique_values = np.unique(data[data > 0])
            qc_results['intensity_check'][file] = unique_values.tolist()
    
    # Generate QC report
    print(f"\n=== Quality Control Report for Subject {subject_id} ===")
    print(f"Files found: {sum(qc_results['files_exist'].values())}/{len(expected_files)}")
    
    for file, stats in qc_results['volume_stats'].items():
        print(f"\n{file}:")
        print(f"  Shape: {stats['shape']}")
        print(f"  Volume: {stats['volume_mm3']:.2f} mm³")
        print(f"  Voxel count: {stats['voxel_count']}")
        print(f"  Intensity range: {stats['min_intensity']:.1f} - {stats['max_intensity']:.1f}")
    
    return qc_results

# Usage
qc_results = quality_check_subject("/path/to/study", "001")
```

## Batch Processing

### Example 6: Parallel Processing Multiple Subjects

```python
#!/usr/bin/env python3
import multiprocessing as mp
from pathlib import Path
import subprocess
import logging

def process_subject_batch(study_dir, subject_list, n_jobs=4):
    """Process multiple subjects in parallel"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def process_single(subject_id):
        """Process single subject wrapper for multiprocessing"""
        try:
            # Create temporary subject list
            temp_dir = Path(study_dir) / "temp" / subject_id
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_sbj_file = temp_dir / "sbj.list"
            
            with open(temp_sbj_file, 'w') as f:
                f.write(f"{subject_id}\n")
            
            # Run processing pipeline
            cmd = [
                "avp_all",
                "--root-dir", str(temp_dir.parent.parent),
                "--steps", "prep-normalize_stats",
                "--debug"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✓ Subject {subject_id} processed successfully")
                return subject_id, True, ""
            else:
                logger.error(f"✗ Subject {subject_id} failed: {result.stderr}")
                return subject_id, False, result.stderr
                
        except Exception as e:
            logger.error(f"✗ Subject {subject_id} exception: {str(e)}")
            return subject_id, False, str(e)
        finally:
            # Cleanup
            if temp_sbj_file.exists():
                temp_sbj_file.unlink()
            if temp_dir.exists():
                temp_dir.rmdir()
    
    # Process in parallel
    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(process_single, subject_list)
    
    # Summary
    successful = [r[0] for r in results if r[1]]
    failed = [(r[0], r[2]) for r in results if not r[1]]
    
    print(f"\n=== Batch Processing Summary ===")
    print(f"Successful: {len(successful)} subjects")
    print(f"Failed: {len(failed)} subjects")
    
    if failed:
        print("\nFailed subjects:")
        for subject, error in failed:
            print(f"  {subject}: {error[:100]}...")
    
    return successful, failed

# Usage
if __name__ == "__main__":
    study_directory = "/path/to/study"
    subjects = ["001", "002", "003", "004", "005"]
    
    successful, failed = process_subject_batch(
        study_directory, 
        subjects, 
        n_jobs=4
    )
```

### Example 7: Resume Failed Processing

```python
#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

def find_incomplete_subjects(study_dir):
    """Find subjects that haven't completed all processing steps"""
    
    study_path = Path(study_dir)
    
    # Read subject list
    with open(study_path / "data" / "sbj.list", 'r') as f:
        all_subjects = [line.strip() for line in f.readlines()]
    
    incomplete_subjects = []
    
    for subject in all_subjects:
        proc_path = study_path / "data" / "proc" / subject
        
        # Check for key output files
        required_files = [
            'onl_normalized_4bc_iso06.nii.gz',
            'onr_normalized_4bc_iso06.nii.gz'
        ]
        
        missing_files = []
        for file in required_files:
            if not (proc_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            incomplete_subjects.append({
                'subject': subject,
                'missing_files': missing_files
            })
    
    return incomplete_subjects

def resume_processing(study_dir):
    """Resume processing for incomplete subjects"""
    
    incomplete = find_incomplete_subjects(study_dir)
    
    if not incomplete:
        print("✓ All subjects completed processing")
        return
    
    print(f"Found {len(incomplete)} incomplete subjects")
    
    for item in incomplete:
        subject = item['subject']
        print(f"\nResuming processing for subject {subject}")
        print(f"Missing files: {item['missing_files']}")
        
        # Create temporary subject list and reprocess
        temp_path = Path(study_dir) / f"temp_{subject}"
        temp_path.mkdir(exist_ok=True)
        
        with open(temp_path / "sbj.list", 'w') as f:
            f.write(f"{subject}\n")
        
        # Run processing
        import subprocess
        cmd = ["avp_all", "--root-dir", study_dir, "--steps", "normalize-normalize_stats"]
        subprocess.run(cmd)
        
        # Cleanup
        if temp_path.exists():
            import shutil
            shutil.rmtree(temp_path)

# Usage
resume_processing("/path/to/study")
```

## Atlas Generation

### Example 8: Custom Atlas Generation

```python
#!/usr/bin/env python3
import nibabel as nib
import numpy as np
from pathlib import Path
from scipy import ndimage
import matplotlib.pyplot as plt

def generate_custom_atlas(study_dir, probability_threshold=0.5):
    """Generate custom probabilistic atlas with specific parameters"""
    
    study_path = Path(study_dir)
    results_path = study_path / "results"
    
    # Collect normalized images
    left_images = list((results_path / "normalized_iso_L").glob("*_norm_iso06.nii.gz"))
    right_images = list((results_path / "normalized_iso_R").glob("*_norm_iso06.nii.gz"))
    
    print(f"Found {len(left_images)} left and {len(right_images)} right images")
    
    if not left_images or not right_images:
        raise FileNotFoundError("No normalized images found for atlas generation")
    
    # Load template space from first image
    template_img = nib.load(left_images[0])
    template_shape = template_img.shape
    template_affine = template_img.affine
    
    # Initialize probability arrays
    left_prob = np.zeros(template_shape, dtype=np.float32)
    right_prob = np.zeros(template_shape, dtype=np.float32)
    
    # Accumulate left hemisphere
    print("Processing left hemisphere images...")
    for img_path in left_images:
        img = nib.load(img_path)
        data = img.get_fdata()
        # Binary mask where data > 0
        binary_mask = (data > 0).astype(np.float32)
        left_prob += binary_mask
    
    # Accumulate right hemisphere
    print("Processing right hemisphere images...")
    for img_path in right_images:
        img = nib.load(img_path)
        data = img.get_fdata()
        binary_mask = (data > 0).astype(np.float32)
        right_prob += binary_mask
    
    # Convert to probabilities
    left_prob = left_prob / len(left_images)
    right_prob = right_prob / len(right_images)
    
    # Create combined atlas
    combined_prob = np.maximum(left_prob, right_prob)
    
    # Apply smoothing
    left_prob_smooth = ndimage.gaussian_filter(left_prob, sigma=1.0)
    right_prob_smooth = ndimage.gaussian_filter(right_prob, sigma=1.0)
    combined_prob_smooth = ndimage.gaussian_filter(combined_prob, sigma=1.0)
    
    # Save probability maps
    templates_path = study_path / "templates"
    templates_path.mkdir(exist_ok=True)
    
    # Save unthresholded versions
    nib.save(nib.Nifti1Image(left_prob_smooth, template_affine), 
             templates_path / "custom_aVP_prob_l.nii.gz")
    nib.save(nib.Nifti1Image(right_prob_smooth, template_affine), 
             templates_path / "custom_aVP_prob_r.nii.gz")
    nib.save(nib.Nifti1Image(combined_prob_smooth, template_affine), 
             templates_path / "custom_aVP_prob.nii.gz")
    
    # Save thresholded versions
    left_thresh = (left_prob_smooth >= probability_threshold).astype(np.float32)
    right_thresh = (right_prob_smooth >= probability_threshold).astype(np.float32)
    combined_thresh = (combined_prob_smooth >= probability_threshold).astype(np.float32)
    
    nib.save(nib.Nifti1Image(left_thresh, template_affine), 
             templates_path / f"custom_aVP_prob{int(probability_threshold*100)}_l.nii.gz")
    nib.save(nib.Nifti1Image(right_thresh, template_affine), 
             templates_path / f"custom_aVP_prob{int(probability_threshold*100)}_r.nii.gz")
    nib.save(nib.Nifti1Image(combined_thresh, template_affine), 
             templates_path / f"custom_aVP_prob{int(probability_threshold*100)}.nii.gz")
    
    # Generate summary statistics
    stats = {
        'n_subjects': len(left_images),
        'template_shape': template_shape,
        'left_volume_mm3': np.sum(left_thresh) * np.prod(template_img.header.get_zooms()),
        'right_volume_mm3': np.sum(right_thresh) * np.prod(template_img.header.get_zooms()),
        'combined_volume_mm3': np.sum(combined_thresh) * np.prod(template_img.header.get_zooms()),
        'max_probability': float(np.max(combined_prob_smooth)),
        'mean_probability': float(np.mean(combined_prob_smooth[combined_prob_smooth > 0]))
    }
    
    print(f"\n=== Custom Atlas Generated ===")
    print(f"Subjects included: {stats['n_subjects']}")
    print(f"Template dimensions: {stats['template_shape']}")
    print(f"Combined volume at {probability_threshold} threshold: {stats['combined_volume_mm3']:.2f} mm³")
    print(f"Maximum probability: {stats['max_probability']:.3f}")
    
    return stats

# Usage
atlas_stats = generate_custom_atlas("/path/to/study", probability_threshold=0.6)
```

## Statistical Analysis

### Example 9: Compare Two Datasets

```bash
# Compare healthy controls vs patients
avp_all --root-dir /path/to/main/study \
        --steps stats \
        --dataset-A /path/to/healthy/controls \
        --dataset-B /path/to/patient/data \
        --debug
```

### Example 10: Custom Statistical Analysis

```python
#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def compare_morphometrics(dataset_a_path, dataset_b_path, output_path):
    """Compare morphometric measurements between two datasets"""
    
    # Load data
    data_a = pd.read_excel(Path(dataset_a_path) / "results" / "aVP_section_CSA_length_iso.xlsx")
    data_b = pd.read_excel(Path(dataset_b_path) / "results" / "aVP_section_CSA_length_iso.xlsx")
    
    # Add group labels
    data_a['Group'] = 'Dataset A'
    data_b['Group'] = 'Dataset B'
    
    # Combine datasets
    combined_data = pd.concat([data_a, data_b], ignore_index=True)
    
    # Morphometric measures to compare
    measures = [
        'ON_L_volume', 'ON_R_volume',
        'ON_L_length', 'ON_R_length',
        'ON_L_mean_CSA', 'ON_R_mean_CSA',
        'OC_volume', 'OT_L_volume', 'OT_R_volume'
    ]
    
    results = []
    
    print("=== Statistical Comparison Results ===\n")
    
    for measure in measures:
        if measure in combined_data.columns:
            group_a_values = data_a[measure].dropna()
            group_b_values = data_b[measure].dropna()
            
            # Descriptive statistics
            desc_a = group_a_values.describe()
            desc_b = group_b_values.describe()
            
            # Statistical test (Mann-Whitney U test for non-parametric data)
            statistic, p_value = stats.mannwhitneyu(
                group_a_values, group_b_values, 
                alternative='two-sided'
            )
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(group_a_values) - 1) * group_a_values.std() ** 2 + 
                                (len(group_b_values) - 1) * group_b_values.std() ** 2) / 
                               (len(group_a_values) + len(group_b_values) - 2))
            cohens_d = (group_a_values.mean() - group_b_values.mean()) / pooled_std
            
            result = {
                'measure': measure,
                'dataset_a_mean': desc_a['mean'],
                'dataset_a_std': desc_a['std'],
                'dataset_a_n': desc_a['count'],
                'dataset_b_mean': desc_b['mean'],
                'dataset_b_std': desc_b['std'], 
                'dataset_b_n': desc_b['count'],
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05
            }
            
            results.append(result)
            
            print(f"{measure}:")
            print(f"  Dataset A: {desc_a['mean']:.3f} ± {desc_a['std']:.3f} (n={int(desc_a['count'])})")
            print(f"  Dataset B: {desc_b['mean']:.3f} ± {desc_b['std']:.3f} (n={int(desc_b['count'])})")
            print(f"  p-value: {p_value:.6f} {'*' if p_value < 0.05 else ''}")
            print(f"  Cohen's d: {cohens_d:.3f}")
            print()
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = Path(output_path) / "statistical_comparison.xlsx"
    results_df.to_excel(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Create visualizations
    create_comparison_plots(combined_data, measures, output_path)
    
    return results_df

def create_comparison_plots(data, measures, output_path):
    """Create comparison plots for morphometric measures"""
    
    output_path = Path(output_path)
    plots_dir = output_path / "comparison_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    for measure in measures[:4]:  # Plot first 4 measures as example
        if measure in data.columns:
            plt.figure(figsize=(10, 6))
            
            # Box plot
            plt.subplot(1, 2, 1)
            sns.boxplot(data=data, x='Group', y=measure)
            plt.title(f'Box Plot: {measure}')
            plt.ylabel(measure.replace('_', ' ').title())
            
            # Violin plot  
            plt.subplot(1, 2, 2)
            sns.violinplot(data=data, x='Group', y=measure)
            plt.title(f'Violin Plot: {measure}')
            plt.ylabel(measure.replace('_', ' ').title())
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"{measure}_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Comparison plots saved to: {plots_dir}")

# Usage
results = compare_morphometrics(
    "/path/to/dataset_a",
    "/path/to/dataset_b", 
    "/path/to/output"
)
```

## Custom Scripts

### Example 11: Custom Preprocessing Script

```python
#!/usr/bin/env python3
"""
Custom preprocessing script for aVP-Toolbox
Handles special cases and quality control
"""

import nibabel as nib
import numpy as np
from pathlib import Path
import logging

def custom_preprocess_subject(subject_dir):
    """Custom preprocessing with additional quality checks"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    subject_path = Path(subject_dir)
    subject_id = subject_path.name
    
    logger.info(f"Preprocessing subject: {subject_id}")
    
    # Define input files
    input_files = {
        'onc': subject_path / 'onc.nii.gz',
        'onl': subject_path / 'onl.nii.gz', 
        'onr': subject_path / 'onr.nii.gz',
        'otl': subject_path / 'otl.nii.gz',
        'otr': subject_path / 'otr.nii.gz'
    }
    
    # Check files exist
    missing_files = [name for name, path in input_files.items() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing files for {subject_id}: {missing_files}")
    
    # Quality control checks
    qc_results = perform_quality_control(input_files, logger)
    
    # Apply corrections if needed
    if qc_results['needs_correction']:
        apply_corrections(input_files, qc_results, logger)
    
    # Create output directory
    output_dir = subject_path.parent / 'preprocessed' / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy processed files
    for name, input_path in input_files.items():
        output_path = output_dir / input_path.name
        import shutil
        shutil.copy2(input_path, output_path)
        logger.info(f"Processed {name} -> {output_path}")
    
    return qc_results

def perform_quality_control(input_files, logger):
    """Perform quality control checks on input segmentations"""
    
    qc_results = {
        'needs_correction': False,
        'issues_found': [],
        'file_stats': {}
    }
    
    expected_intensities = {
        'onl': [2, 4, 6],     # iOrb, iCan, iCran
        'onr': [2, 4, 6],     # iOrb, iCan, iCran  
        'onc': [8, 9],        # left/right hemichiasm
        'otl': [1],           # optic tract
        'otr': [1]            # optic tract
    }
    
    for name, file_path in input_files.items():
        logger.info(f"Checking {name}...")
        
        # Load image
        img = nib.load(file_path)
        data = img.get_fdata()
        
        # Basic statistics
        stats = {
            'shape': data.shape,
            'voxel_size': img.header.get_zooms(),
            'unique_values': np.unique(data).tolist(),
            'total_voxels': np.sum(data > 0),
            'volume_mm3': np.sum(data > 0) * np.prod(img.header.get_zooms())
        }
        
        qc_results['file_stats'][name] = stats
        
        # Check voxel size (should be 0.6mm isotropic)
        voxel_sizes = np.array(stats['voxel_size'][:3])
        if not np.allclose(voxel_sizes, 0.6, atol=0.01):
            issue = f"{name}: Non-standard voxel size {voxel_sizes}"
            qc_results['issues_found'].append(issue)
            logger.warning(issue)
        
        # Check expected intensity values
        actual_values = [v for v in stats['unique_values'] if v > 0]
        expected_values = expected_intensities[name]
        
        unexpected_values = set(actual_values) - set(expected_values)
        missing_values = set(expected_values) - set(actual_values)
        
        if unexpected_values:
            issue = f"{name}: Unexpected intensity values {list(unexpected_values)}"
            qc_results['issues_found'].append(issue)
            logger.warning(issue)
            qc_results['needs_correction'] = True
        
        if missing_values and name in ['onl', 'onr']:  # Some subdivisions might be missing
            issue = f"{name}: Missing expected subdivisions {list(missing_values)}"
            qc_results['issues_found'].append(issue)
            logger.info(issue)  # Info level as this might be normal
        
        # Check for reasonable volume
        if stats['volume_mm3'] < 10:  # Very small volume
            issue = f"{name}: Suspiciously small volume {stats['volume_mm3']:.2f} mm³"
            qc_results['issues_found'].append(issue)
            logger.warning(issue)
        
        logger.info(f"  Shape: {stats['shape']}, Volume: {stats['volume_mm3']:.2f} mm³")
    
    return qc_results

def apply_corrections(input_files, qc_results, logger):
    """Apply automatic corrections to segmentation issues"""
    
    logger.info("Applying automatic corrections...")
    
    for name, file_path in input_files.items():
        img = nib.load(file_path)
        data = img.get_fdata().copy()
        
        # Round to integers (common issue)
        data = np.round(data).astype(int)
        
        # Remove small isolated voxels (< 5 voxels)
        from scipy import ndimage
        for intensity in np.unique(data)[1:]:  # Skip 0
            mask = (data == intensity)
            labeled, num_features = ndimage.label(mask)
            
            for i in range(1, num_features + 1):
                component = (labeled == i)
                if np.sum(component) < 5:  # Remove small components
                    data[component] = 0
                    logger.info(f"Removed small component in {name} (intensity {intensity})")
        
        # Save corrected file
        corrected_img = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(corrected_img, file_path)
        logger.info(f"Applied corrections to {name}")

# Usage as standalone script
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python custom_preprocess.py /path/to/subject/dir")
        sys.exit(1)
    
    subject_directory = sys.argv[1]
    qc_results = custom_preprocess_subject(subject_directory)
    
    if qc_results['issues_found']:
        print("\nIssues found:")
        for issue in qc_results['issues_found']:
            print(f"  - {issue}")
    else:
        print("\n✓ No issues found")
```

### Example 12: Batch Quality Control

```python
#!/usr/bin/env python3
"""
Batch quality control for entire study
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def batch_quality_control(study_dir):
    """Run quality control on all subjects in study"""
    
    study_path = Path(study_dir)
    
    # Read subject list
    with open(study_path / "data" / "sbj.list", 'r') as f:
        subjects = [line.strip() for line in f.readlines()]
    
    qc_summary = []
    
    for subject in subjects:
        subject_dir = study_path / "data" / "orig" / subject
        
        try:
            qc_result = custom_preprocess_subject(subject_dir)
            
            summary = {
                'subject_id': subject,
                'status': 'PASS' if not qc_result['issues_found'] else 'ISSUES',
                'num_issues': len(qc_result['issues_found']),
                'issues': '; '.join(qc_result['issues_found'])
            }
            
            # Add volume statistics
            for structure in ['onl', 'onr', 'onc', 'otl', 'otr']:
                if structure in qc_result['file_stats']:
                    volume_key = f'{structure}_volume_mm3'
                    summary[volume_key] = qc_result['file_stats'][structure]['volume_mm3']
            
            qc_summary.append(summary)
            
        except Exception as e:
            qc_summary.append({
                'subject_id': subject,
                'status': 'ERROR',
                'num_issues': 1,
                'issues': str(e)
            })
    
    # Create summary DataFrame
    qc_df = pd.DataFrame(qc_summary)
    
    # Save results
    output_path = study_path / "quality_control"
    output_path.mkdir(exist_ok=True)
    
    qc_df.to_excel(output_path / "qc_summary.xlsx", index=False)
    
    # Generate QC plots
    generate_qc_plots(qc_df, output_path)
    
    # Print summary
    print(f"\n=== Quality Control Summary ===")
    print(f"Total subjects: {len(subjects)}")
    print(f"Passed: {len(qc_df[qc_df['status'] == 'PASS'])}")
    print(f"Issues found: {len(qc_df[qc_df['status'] == 'ISSUES'])}")
    print(f"Errors: {len(qc_df[qc_df['status'] == 'ERROR'])}")
    
    return qc_df

def generate_qc_plots(qc_df, output_path):
    """Generate quality control visualization plots"""
    
    # Volume distribution plots
    volume_columns = [col for col in qc_df.columns if 'volume_mm3' in col]
    
    if volume_columns:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(volume_columns[:6]):  # Plot first 6 structures
            if i < len(axes):
                qc_df[col].hist(bins=20, ax=axes[i])
                axes[i].set_title(f'{col.replace("_volume_mm3", "").upper()} Volume Distribution')
                axes[i].set_xlabel('Volume (mm³)')
                axes[i].set_ylabel('Frequency')
        
        # Remove empty subplots
        for i in range(len(volume_columns), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(output_path / "volume_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Status summary plot
    status_counts = qc_df['status'].value_counts()
    
    plt.figure(figsize=(8, 6))
    status_counts.plot(kind='bar', color=['green', 'orange', 'red'])
    plt.title('Quality Control Status Summary')
    plt.xlabel('Status')
    plt.ylabel('Number of Subjects')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path / "qc_status_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

# Usage
qc_results = batch_quality_control("/path/to/study")
```

## Visualization Examples

### Example 13: Create Atlas Visualization

```python
#!/usr/bin/env python3
"""
Create visualization of probabilistic atlas
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting
from pathlib import Path

def visualize_atlas(atlas_path, output_dir):
    """Create comprehensive atlas visualizations"""
    
    atlas_path = Path(atlas_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load atlas
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()
    
    print(f"Atlas shape: {atlas_data.shape}")
    print(f"Probability range: {np.min(atlas_data):.3f} - {np.max(atlas_data):.3f}")
    
    # 1. Glass brain view
    plt.figure(figsize=(15, 5))
    plotting.plot_glass_brain(
        atlas_img,
        threshold=0.1,
        display_mode='lyrz',
        colorbar=True,
        title=f'Atlas: {atlas_path.stem}'
    )
    plt.savefig(output_dir / f"{atlas_path.stem}_glass_brain.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Statistical map view
    plt.figure(figsize=(12, 8))
    plotting.plot_stat_map(
        atlas_img,
        threshold=0.1,
        display_mode='z',
        cut_coords=5,
        colorbar=True,
        title=f'Atlas Probability Map: {atlas_path.stem}'
    )
    plt.savefig(output_dir / f"{atlas_path.stem}_stat_map.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Slice-by-slice view
    create_slice_montage(atlas_data, atlas_img.affine, output_dir, atlas_path.stem)
    
    # 4. 3D surface rendering (if available)
    try:
        fig = plotting.plot_img(
            atlas_img,
            threshold=0.3,
            title=f'3D View: {atlas_path.stem}'
        )
        fig.savefig(output_dir / f"{atlas_path.stem}_3d.png", dpi=300, bbox_inches='tight')
        plt.close()
    except:
        print("3D rendering not available")
    
    print(f"Atlas visualizations saved to: {output_dir}")

def create_slice_montage(data, affine, output_dir, name):
    """Create montage of atlas slices"""
    
    # Find slices with significant probability
    slice_sums = np.sum(data > 0.1, axis=(0, 1))
    significant_slices = np.where(slice_sums > 10)[0]  # At least 10 voxels
    
    if len(significant_slices) == 0:
        return
    
    # Select representative slices
    n_slices = min(12, len(significant_slices))
    selected_slices = significant_slices[::len(significant_slices)//n_slices][:n_slices]
    
    # Create montage
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, slice_idx in enumerate(selected_slices):
        if i < len(axes):
            slice_data = data[:, :, slice_idx]
            
            im = axes[i].imshow(slice_data.T, cmap='hot', origin='lower', 
                               vmin=0, vmax=np.max(data))
            axes[i].set_title(f'Slice {slice_idx}')
            axes[i].axis('off')
    
    # Remove empty subplots
    for i in range(len(selected_slices), len(axes)):
        fig.delaxes(axes[i])
    
    # Add colorbar
    plt.colorbar(im, ax=axes, shrink=0.6, aspect=20)
    plt.suptitle(f'Atlas Slice Montage: {name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_slice_montage.png", dpi=300, bbox_inches='tight')
    plt.close()

# Usage
visualize_atlas(
    "/path/to/study/templates/aVP_prob.nii.gz",
    "/path/to/output/visualizations"
)
```

## Troubleshooting Examples

### Example 14: Debug Processing Issues

```python
#!/usr/bin/env python3
"""
Debug common aVP-Toolbox processing issues
"""

import subprocess
import sys
from pathlib import Path
import nibabel as nib
import numpy as np

def debug_processing_environment():
    """Check processing environment and dependencies"""
    
    print("=== Environment Debug Check ===\n")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check required packages
    required_packages = [
        'nibabel', 'numpy', 'scipy', 'matplotlib', 
        'pandas', 'scikit-learn', 'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nInstall missing packages with: pip install {' '.join(missing_packages)}")
    
    # Check FSL installation
    try:
        result = subprocess.run(['which', 'fsl'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ FSL found at: {result.stdout.strip()}")
        else:
            print("✗ FSL not found in PATH")
    except:
        print("✗ FSL check failed")
    
    # Check MATLAB (optional)
    try:
        result = subprocess.run(['which', 'matlab'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ MATLAB found at: {result.stdout.strip()}")
        else:
            print("⚠ MATLAB not found (optional)")
    except:
        print("⚠ MATLAB check failed (optional)")

def debug_subject_data(subject_dir):
    """Debug issues with subject data"""
    
    subject_path = Path(subject_dir)
    subject_id = subject_path.name
    
    print(f"\n=== Debug Subject: {subject_id} ===\n")
    
    # Check file structure
    required_files = ['onc.nii.gz', 'onl.nii.gz', 'onr.nii.gz', 'otl.nii.gz', 'otr.nii.gz']
    
    file_issues = []
    for filename in required_files:
        filepath = subject_path / filename
        
        if not filepath.exists():
            file_issues.append(f"Missing: {filename}")
            continue
        
        try:
            # Load and check image
            img = nib.load(filepath)
            data = img.get_fdata()
            header = img.header
            
            print(f"{filename}:")
            print(f"  Shape: {data.shape}")
            print(f"  Voxel size: {header.get_zooms()[:3]}")
            print(f"  Data type: {data.dtype}")
            print(f"  Value range: {np.min(data):.1f} - {np.max(data):.1f}")
            print(f"  Unique values: {sorted(np.unique(data).astype(int))}")
            print(f"  Non-zero voxels: {np.sum(data > 0)}")
            
            # Check for common issues
            if not np.allclose(header.get_zooms()[:3], 0.6, atol=0.01):
                file_issues.append(f"{filename}: Non-standard voxel size")
            
            if data.dtype not in [np.int16, np.int32, np.float32, np.float64]:
                file_issues.append(f"{filename}: Unusual data type {data.dtype}")
            
            unique_vals = np.unique(data[data > 0]).astype(int)
            expected_ranges = {
                'onl': [2, 4, 6], 'onr': [2, 4, 6], 
                'onc': [8, 9], 'otl': [1], 'otr': [1]
            }
            
            structure = filename.replace('.nii.gz', '')
            if structure in expected_ranges:
                expected = expected_ranges[structure]
                unexpected = [v for v in unique_vals if v not in expected]
                if unexpected:
                    file_issues.append(f"{filename}: Unexpected intensities {unexpected}")
            
        except Exception as e:
            file_issues.append(f"{filename}: Load error - {str(e)}")
        
        print()
    
    if file_issues:
        print("Issues found:")
        for issue in file_issues:
            print(f"  ✗ {issue}")
    else:
        print("✓ No issues detected")
    
    return file_issues

def debug_processing_step(study_dir, step_name, subject_id=None):
    """Debug specific processing step"""
    
    print(f"\n=== Debug Step: {step_name} ===\n")
    
    study_path = Path(study_dir)
    
    # Check input requirements for step
    step_requirements = {
        'prep': ['data/orig/{subject}/onc.nii.gz', 'data/sbj.list'],
        'basics': ['data/proc/{subject}/oc_l.nii.gz'],
        'normalize': ['data/proc/{subject}/on_l.nii.gz'],
        'resample': ['data/proc/{subject}/onl_normalized_4bc.nii.gz'],
        'atlas': ['results/normalized_iso_L/']
    }
    
    if step_name in step_requirements:
        missing_inputs = []
        
        for req_path in step_requirements[step_name]:
            if subject_id and '{subject}' in req_path:
                req_path = req_path.format(subject=subject_id)
            
            full_path = study_path / req_path
            
            if not full_path.exists():
                missing_inputs.append(str(full_path))
        
        if missing_inputs:
            print("Missing required inputs:")
            for path in missing_inputs:
                print(f"  ✗ {path}")
        else:
            print("✓ All required inputs present")
    
    # Run step with debug output
    try:
        cmd = [
            'avp_all', 
            '--root-dir', str(study_path),
            '--steps', step_name,
            '--debug'
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Step completed successfully")
        else:
            print(f"✗ Step failed with return code: {result.returncode}")
            print(f"STDERR: {result.stderr}")
        
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
            
    except Exception as e:
        print(f"✗ Failed to run step: {str(e)}")

# Usage examples
if __name__ == "__main__":
    # Debug environment
    debug_processing_environment()
    
    # Debug specific subject
    # debug_subject_data("/path/to/study/data/orig/001")
    
    # Debug processing step
    # debug_processing_step("/path/to/study", "prep", "001")
```

These examples cover the most common use cases for the aVP-Toolbox. Each example is self-contained and can be adapted to your specific needs. For more complex scenarios, you can combine multiple examples or modify them according to your requirements.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Explore repository structure and understand the project", "status": "completed", "activeForm": "Exploring repository structure and understanding the project"}, {"content": "Analyze main code components and functionality", "status": "completed", "activeForm": "Analyzing main code components and functionality"}, {"content": "Create comprehensive README.md", "status": "completed", "activeForm": "Creating comprehensive README.md"}, {"content": "Create detailed documentation", "status": "completed", "activeForm": "Creating detailed documentation"}, {"content": "Create set of examples", "status": "completed", "activeForm": "Creating set of examples"}]