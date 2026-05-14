#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        AVP_SEG_2026  —  Portable Edition                   ║
║          Anterior Visual Pathway Segmentation Tool  (V1.0)                 ║
║                                                                            ║
║  Multi-label optic nerve segmentation from 3D CISS MRI using ensemble      ║
║  VNet + Refinement UNet + Majority Voting + Post-processing pipeline.      ║
║                                                                            ║
║  Author:  Andrea Diociasi                                                  ║
║  Version: V1.0 (2026) — self-contained portable package                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

=== HOW TO USE THIS TOOL ===

1) INSTALL REQUIREMENTS
       pip install torch nibabel numpy scipy scikit-image tqdm pydicom

2) FOLDER STRUCTURE  (already included in this package)
   This script is designed to run from the AVP_SEG_V1.0/ folder:

       AVP_SEG_V1.0/
       ├── AVP_SEG_2026.py            <-- this script
       ├── input/                      <-- put your CISS images HERE (.nii.gz or DICOM)
       ├── output/                     <-- results will be saved here
       └── models/
           ├── base_models/
           │   ├── RS1/val/best_epoch.pth
           │   ├── RS2/val/best_epoch.pth
           │   └── ...  (RS1 through RS11)
           └── refinement_models/
               ├── RS1/checkpoints/epoch_204.pth
               ├── RS2/checkpoints/epoch_204.pth
               └── ...  (RS1 through RS11)

3) COPY YOUR IMAGES
   Place your 3D CISS images into the  input/  folder.
   Supported: NIfTI (.nii.gz) or DICOM (any folder structure).
   DICOM files are auto-detected and converted to NIfTI before processing.

4) RUN
       cd AVP_SEG_V1.0
       python AVP_SEG_2026.py

   That's it! The script auto-detects all paths relative to itself.
   Results will appear in  output/PREDICTIONS/ .
   The cropped CISS image is saved alongside segmentations for overlay.

=== ADVANCED: CUSTOM PATHS ===
   If you prefer to keep models or images elsewhere, you can override
   the three paths in the USER CONFIGURATION section below, or set them
   as environment variables:
       export AVP_PROJECT_ROOT="/path/to/AVP_SEG_V1.0"
       export AVP_INPUT_FOLDER="/path/to/my_images"
       export AVP_OUTPUT_FOLDER="/path/to/my_results"

=== NOTES ===
- Works on Windows, macOS, and Linux (paths handled with os.path / pathlib).
- If a model checkpoint is not found, it is skipped with a warning.
- GPU is used automatically if available; otherwise CPU is used.
- The refinement checkpoint filename defaults to "epoch_204.pth".
  Change REFINEMENT_EPOCH below if your checkpoints have a different name.
"""

import os
import sys
import glob
import shutil
import re
import struct
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from scipy.ndimage import label as cc_label, center_of_mass, binary_fill_holes
from skimage.morphology import remove_small_objects
from tqdm import tqdm

logger = logging.getLogger(__name__)
NAME = "ai"

# pydicom is optional — only needed if DICOM files are present in input/
try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         USER CONFIGURATION                                 ║
# ║  Set the three paths below to match YOUR system.                           ║
# ║  Use raw strings on Windows: r"C:\Users\..."                               ║
# ║  Or forward slashes everywhere: "C:/Users/..."                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# --- Path 1: PROJECT ROOT ---
# The root folder of this package (the folder containing this script,
# plus the models/, input/, and output/ subfolders).
#
# If set to None (recommended), it is auto-detected as the folder
# where this script is located.
#
# Override examples (only if you moved models elsewhere):
#   Windows:  r"C:\Users\YourName\Desktop\AVP_SEG_V1.0"
#   macOS:    "/Users/yourname/Desktop/AVP_SEG_V1.0"
#   Linux:    "/home/yourname/AVP_SEG_V1.0"
#
# You can also use the environment variable AVP_PROJECT_ROOT.
PROJECT_ROOT = None

# --- Path 2: INPUT FOLDER ---
# Folder containing the CISS 3D NIfTI images (.nii.gz) to segment.
#
# If set to None (recommended), defaults to the  input/  subfolder
# inside PROJECT_ROOT (i.e.  AVP_SEG_V1.0/input/ ).
# Just copy your .nii.gz images there and run.
#
# Override examples (only if your images are elsewhere):
#   r"D:\MRI_data\CISS_images"
#   "/data/study/CISS_nifti"
#
# You can also use the environment variable AVP_INPUT_FOLDER.
INPUT_FOLDER = None

# --- Path 3: OUTPUT FOLDER ---
# Where all predictions, refinements, merged results, and post-processed
# segmentations will be saved. Created automatically if it does not exist.
#
# If set to None (recommended), defaults to the  output/  subfolder
# inside PROJECT_ROOT (i.e.  AVP_SEG_V1.0/output/ ).
#
# You can also use the environment variable AVP_OUTPUT_FOLDER.
OUTPUT_FOLDER = None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      ADVANCED CONFIGURATION                                ║
# ║  You should not need to change these unless your setup is different.        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# RS model IDs used in the ensemble (RS1 through RS11).
# Base model checkpoints are at:   models/base_models/RS{n}/val/best_epoch.pth
# Refinement checkpoints are at:   models/refinement_models/RS{n}/checkpoints/{REFINEMENT_EPOCH}
RS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Refinement model checkpoint filename
REFINEMENT_EPOCH = "epoch_204.pth"

# Target shape for cropping/padding input volumes
TARGET_SHAPE = (256, 256, 64)

# Majority voting threshold (fraction of models that must agree)
VOTING_THRESHOLD = 0.5


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         PATH AUTO-DETECTION                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _find_project_root():
    """Auto-detect PROJECT_ROOT using multiple strategies."""
    script_dir = Path(__file__).resolve().parent

    # Strategy 1: script sits directly in AVP_SEG_V1.0/ (portable package)
    if _is_valid_project_root(script_dir):
        return str(script_dir)

    # Strategy 2: script is one level deeper (e.g. code/ subfolder)
    if _is_valid_project_root(script_dir.parent):
        return str(script_dir.parent)

    # Strategy 3: script is two levels deeper (e.g. MODELLO/CODE/)
    if _is_valid_project_root(script_dir.parent.parent):
        return str(script_dir.parent.parent)

    # Strategy 4: look in common locations
    for search_root in [Path.home() / "Desktop", Path.home() / "Documents",
                        Path.home() / "Downloads", Path.home()]:
        if search_root.exists():
            for d in search_root.iterdir():
                if d.is_dir() and d.name.startswith("AVP_SEG"):
                    if _is_valid_project_root(d):
                        return str(d)

    return None


def _is_valid_project_root(path):
    """Check if a path looks like a valid project root (portable or legacy)."""
    p = Path(path)
    if not p.exists():
        return False
    # Portable layout: models/base_models/ exists
    if (p / "models" / "base_models").is_dir():
        return True
    # Legacy layout: RESULTSATTENTION folders or RefineSegModel
    has_results = any(d.name.startswith("RESULTSATTENTION2025V2_RS") for d in p.iterdir() if d.is_dir())
    has_refine = (p / "RefineSegModel").is_dir()
    return has_results or has_refine


def _detect_layout(root):
    """Detect whether the project uses portable or legacy folder layout.
    Returns 'portable' or 'legacy'."""
    if (Path(root) / "models" / "base_models").is_dir():
        return "portable"
    return "legacy"


def _get_base_model_path(rs_id):
    """Return path to the base model checkpoint for a given RS id."""
    if _detect_layout(PROJECT_ROOT) == "portable":
        return os.path.join(PROJECT_ROOT, "models", "base_models", f"RS{rs_id}", "val", "best_epoch.pth")
    else:
        return os.path.join(PROJECT_ROOT, f"RESULTSATTENTION2025V2_RS{rs_id}NEW", "val", "best_epoch.pth")


def _get_refine_model_path(rs_id):
    """Return path to the refinement model checkpoint for a given RS id."""
    if _detect_layout(PROJECT_ROOT) == "portable":
        return os.path.join(PROJECT_ROOT, "models", "refinement_models", f"RS{rs_id}", "checkpoints", REFINEMENT_EPOCH)
    else:
        return os.path.join(PROJECT_ROOT, "RefineSegModel", f"RS{rs_id}", "checkpoints", REFINEMENT_EPOCH)


def resolve_paths():
    """Resolve and validate all paths, auto-detecting or prompting as needed."""
    global PROJECT_ROOT, INPUT_FOLDER, OUTPUT_FOLDER

    # --- PROJECT_ROOT (check env var, then auto-detect, then prompt) ---
    if PROJECT_ROOT is None:
        PROJECT_ROOT = os.environ.get("AVP_PROJECT_ROOT")
    if PROJECT_ROOT is None:
        detected = _find_project_root()
        if detected:
            logger.info(f"Auto-detected PROJECT_ROOT: {detected}")
            PROJECT_ROOT = detected
        else:
            PROJECT_ROOT = input(
                "\nCould not auto-detect PROJECT_ROOT.\n"
                "Please enter the full path to your AVP_SEG_V1.0 folder:\n> "
            ).strip().strip('"').strip("'")

    if not os.path.isdir(PROJECT_ROOT):
        logger.error(f"PROJECT_ROOT does not exist: {PROJECT_ROOT}")
        sys.exit(1)

    layout = _detect_layout(PROJECT_ROOT)
    logger.info(f"Detected layout: {layout}")

    # --- INPUT_FOLDER (env var, then default to PROJECT_ROOT/input/, then prompt) ---
    if INPUT_FOLDER is None:
        INPUT_FOLDER = os.environ.get("AVP_INPUT_FOLDER")
    if INPUT_FOLDER is None:
        default_input = os.path.join(PROJECT_ROOT, "input")
        if not os.path.isdir(default_input):
            # Auto-create input/ folder if missing (e.g. .gitkeep was deleted)
            os.makedirs(default_input, exist_ok=True)
        INPUT_FOLDER = default_input
        logger.info(f"Input folder: {INPUT_FOLDER}")

    if not os.path.isdir(INPUT_FOLDER):
        logger.error(f"INPUT_FOLDER does not exist: {INPUT_FOLDER}")
        sys.exit(1)

    # Check for any inputs (NIfTI or DICOM)
    nifti_count = len(glob.glob(os.path.join(INPUT_FOLDER, "*.nii.gz")))
    has_other_files = any(
        not f.startswith('.') and not f.endswith('.nii.gz')
        for f in os.listdir(INPUT_FOLDER)
        if os.path.isfile(os.path.join(INPUT_FOLDER, f)) or os.path.isdir(os.path.join(INPUT_FOLDER, f))
    )
    if nifti_count == 0 and not has_other_files:
        logger.warning(f"No images found in {INPUT_FOLDER}")
        logger.warning("Place your CISS images (.nii.gz or DICOM) there and re-run.")

    # --- OUTPUT_FOLDER (env var, then default to PROJECT_ROOT/output/) ---
    if OUTPUT_FOLDER is None:
        OUTPUT_FOLDER = os.environ.get("AVP_OUTPUT_FOLDER")
    if OUTPUT_FOLDER is None:
        OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "output")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # --- Check model availability ---
    models_found = sum(1 for rs in RS_IDS if os.path.exists(_get_base_model_path(rs)))
    refine_found = sum(1 for rs in RS_IDS if os.path.exists(_get_refine_model_path(rs)))

    # --- Log summary ---
    logger.info("AVP_SEG_2026 V1.0 — Configuration Summary")
    logger.info(f"  PROJECT_ROOT     : {PROJECT_ROOT}")
    logger.info(f"  INPUT_FOLDER     : {INPUT_FOLDER}")
    logger.info(f"  OUTPUT_FOLDER    : {OUTPUT_FOLDER}")
    logger.info(f"  Layout           : {layout}")
    logger.info(f"  NIfTI images     : {nifti_count}")
    logger.info(f"  Base models      : {models_found}/{len(RS_IDS)}")
    logger.info(f"  Refinement models: {refine_found}/{len(RS_IDS)}")
    logger.info(f"  Device           : {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          MODEL COMPONENTS                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ==== MODEL COMPONENTS ====
class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionGate, self).__init__()
        self.Wx = nn.Conv3d(in_channels, inter_channels, kernel_size=1)
        self.Wg = nn.Conv3d(gating_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv3d(inter_channels, 1, kernel_size=1)

    def forward(self, x, g):
        wx = self.Wx(x)
        wg = self.Wg(g)
        psi = torch.sigmoid(self.psi(F.relu(wx + wg)))
        return x * psi

class Res2Block3D(nn.Module):
    def __init__(self, in_channels, scale=4):
        super().__init__()
        assert in_channels % scale == 0, "Channels must be divisible by scale"
        self.scale = scale
        self.width = in_channels // scale
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(self.width, self.width, kernel_size=3, padding=1),
                nn.BatchNorm3d(self.width),
                nn.ReLU(inplace=True)
            ) for _ in range(scale - 1)
        ])
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        splits = torch.chunk(x, self.scale, dim=1)
        out = [splits[0]]
        for i in range(1, self.scale):
            res = splits[i] + out[-1]
            out.append(self.blocks[i - 1](res))
        x_out = torch.cat(out, dim=1)
        return self.relu(self.bn(x_out))

class VNet3D(nn.Module):
    def __init__(self, in_channels, out_channels=10):
        super(VNet3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=2, stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = Res2Block3D(64, scale=4)
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.attention1 = AttentionGate(32, 32, 16)
        self.upconv1 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.aux_output1 = nn.Conv3d(32, out_channels, kernel_size=1)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.attention2 = AttentionGate(16, 16, 8)
        self.upconv2 = nn.Sequential(
            nn.Conv3d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),
            nn.Conv3d(48, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout3d(p=0.5),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, out_channels, kernel_size=1)
        )

    def forward(self, x):
        enc = self.encoder(x)
        down1 = self.downsample1(enc)
        conv1 = self.conv1(down1)
        down2 = self.downsample2(conv1)
        conv2 = self.conv2(down2)
        up1 = self.upsample1(conv2)
        att1 = self.attention1(conv1, up1)
        concat1 = torch.cat([up1, att1], dim=1)
        upconv1 = self.upconv1(concat1)
        aux_out = self.aux_output1(upconv1)
        up2 = self.upsample2(upconv1)
        att2 = self.attention2(enc, up2)
        concat2 = torch.cat([up2, att2], dim=1)
        upconv2 = self.upconv2(concat2)
        main_out = self.classifier(upconv2)
        return main_out, aux_out


# ==== REFINEMENT MODEL ====
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()
        self.Wx = nn.Conv3d(in_channels, inter_channels, 1)
        self.Wg = nn.Conv3d(gating_channels, inter_channels, 1)
        self.psi = nn.Conv3d(inter_channels, 1, 1)

    def forward(self, x, g):
        psi = torch.sigmoid(self.psi(F.relu(self.Wx(x) + self.Wg(g))))
        return x * psi

class UNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=6):
        super().__init__()
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.pool = nn.MaxPool3d(2)

        self.up2 = nn.ConvTranspose3d(128, 64, 2, 2)
        self.att2 = AttentionBlock(64, 64, 32)
        self.dec2 = self.conv_block(128, 64)

        self.up1 = nn.ConvTranspose3d(64, 32, 2, 2)
        self.att1 = AttentionBlock(32, 32, 16)
        self.dec1 = self.conv_block(64, 32)

        self.out = nn.Conv3d(32, out_channels, 1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.up2(e3)
        a2 = self.att2(e2, d2)
        d2 = self.dec2(torch.cat([d2, a2], dim=1))

        d1 = self.up1(d2)
        a1 = self.att1(e1, d1)
        d1 = self.dec1(torch.cat([d1, a1], dim=1))

        return self.out(d1)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              UTILITIES                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def normalize(volume):
    mean = np.mean(volume)
    std = np.std(volume)
    return (volume - mean) / std if std != 0 else volume

def crop_or_pad(volume, target_shape):
    current_shape = volume.shape
    pad_width = []
    slices = []

    for i in range(3):
        diff = target_shape[i] - current_shape[i]
        if diff > 0:
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_width.append((pad_before, pad_after))
            slices.append(slice(0, current_shape[i]))
        else:
            crop_before = (-diff) // 2
            crop_after = crop_before + target_shape[i]
            pad_width.append((0, 0))
            slices.append(slice(crop_before, crop_after))

    volume = volume[slices[0], slices[1], slices[2]]
    volume = np.pad(volume, pad_width, mode='constant')
    return volume


def reorient_to_canonical(nii_img):
    """Reorient a NIfTI image to RAS+ canonical orientation with identity affine,
    matching the training data format.

    This works for ANY input orientation (axial, coronal, sagittal, oblique)
    as long as the affine correctly encodes the voxel-to-patient mapping.
    For DICOM conversions, make sure the affine is in RAS convention
    (i.e. LPS→RAS conversion has been applied).

    Steps:
      1) Use nibabel to determine the current orientation from the affine
      2) Compute the transform needed to reach RAS+ orientation
      3) Apply the axis reorder/flip to the data array
      4) Save with identity affine (matching training data)
    """
    # Determine current orientation from the affine
    orig_ornt = nib.io_orientation(nii_img.affine)
    # Target: RAS+ orientation
    ras_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
    # Compute the transform (axis reorder + flips)
    transform = nib.orientations.ornt_transform(orig_ornt, ras_ornt)

    # Apply the orientation transform to the data
    data = nib.orientations.apply_orientation(nii_img.get_fdata(), transform)

    # Use identity affine — the model was trained with identity affine
    # and expects RAS-ordered voxel data in this format
    return nib.Nifti1Image(data.astype(np.float32), np.eye(4))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       POST-PROCESSING FUNCTIONS                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

CHIASM_LABEL = 4
POSTERIOR_LABEL = 5  # Optic tracts
ANTERIOR_LABEL = 3   # Optic nerves

# 1) Remove small components per label
def remove_small_clusters(segmentation, min_size=20):
    output = np.zeros_like(segmentation)
    for label_id in np.unique(segmentation):
        if label_id == 0:
            continue
        mask = segmentation == label_id
        if label_id == 1:
            labeled, num = cc_label(mask)
            mid_x = segmentation.shape[0] // 2
            sizes = [(i, np.sum(labeled == i)) for i in range(1, num + 1)]
            left = [s for s in sizes if np.mean(np.where(labeled == s[0])[0]) < mid_x]
            right = [s for s in sizes if np.mean(np.where(labeled == s[0])[0]) >= mid_x]
            keep = []
            if left: keep.append(max(left, key=lambda x: x[1])[0])
            if right: keep.append(max(right, key=lambda x: x[1])[0])
            for k in keep:
                output[labeled == k] = label_id
        else:
            cleaned = remove_small_objects(mask, min_size=min_size)
            output[cleaned] = label_id
    return output

# 2) Chiasm labeling rule enforcement
def enforce_chiasm_sandwich(seg):
    seg_out = seg.copy()
    y_dim = seg.shape[1]

    for y in range(y_dim):
        slice_ = seg[:, y, :]
        labels = np.unique(slice_)

        if CHIASM_LABEL in labels:
            # Chiasm and Tract: keep only tract
            if POSTERIOR_LABEL in labels:
                seg_out[:, y, :][slice_ == CHIASM_LABEL] = POSTERIOR_LABEL

            # Chiasm and ONR: keep only ONR
            elif ANTERIOR_LABEL in labels:
                seg_out[:, y, :][slice_ == CHIASM_LABEL] = ANTERIOR_LABEL

    return seg_out

# 3) Per-slice, side-aware cleanup based on dominant label
def per_slice_side_cleanup(seg):
    seg_out = seg.copy()
    y_dim, x_dim = seg.shape[1], seg.shape[0]
    mid_x = x_dim // 2

    for y in range(y_dim):
        slice_ = seg[:, y, :]

        for side, x_start, x_end in [("left", 0, mid_x), ("right", mid_x, x_dim)]:
            region = slice_[x_start:x_end, :]
            flat = region[region > 0]
            if len(flat) == 0:
                continue

            labels, counts = np.unique(flat, return_counts=True)
            if len(labels) <= 1:
                continue

            dominant = labels[np.argmax(counts)]
            for label in labels:
                if label != dominant:
                    mask = (seg_out[x_start:x_end, y, :] == label)
                    seg_out[x_start:x_end, y, :][mask] = dominant

    return seg_out

def fix_slicewise_label_flips(seg, max_gap=3):
    """
    Replace short label segments along Y-axis (coronal slices) if they are sandwiched between same surrounding label.
    """
    seg_out = seg.copy()
    x_dim, y_dim, z_dim = seg.shape

    for x in range(x_dim):
        for z in range(z_dim):
            col = seg[x, :, z]
            start = 0

            while start < y_dim:
                current_label = col[start]
                end = start + 1
                while end < y_dim and col[end] == current_label:
                    end += 1

                segment_len = end - start
                if 0 < segment_len <= max_gap:
                    # Try to get surrounding labels
                    prev_label = col[start - 1] if start > 0 else None
                    next_label = col[end] if end < y_dim else None

                    # If both neighbors exist and match, replace segment
                    if prev_label is not None and next_label is not None and prev_label == next_label:
                        seg_out[x, start:end, z] = prev_label

                start = end

    return seg_out



def fill_label_slice_gaps(seg, max_gap=2):
    """
    Fill label gaps along the Y-axis if slices are missing the label between consistent slices.
    Works per label and slice.
    """
    seg_out = seg.copy()
    y_dim = seg.shape[1]

    for label in [3, 4, 5]:  # Only anatomical labels (ONR, chiasm, tract)
        y = 1
        while y < y_dim - 1:
            prev = seg[:, y - 1, :] == label
            curr = seg[:, y, :] == label
            next_ = seg[:, y + 1, :] == label

            if not np.any(curr) and np.any(prev) and np.any(next_):
                # Fill slice y with the intersection (more conservative) or union (more aggressive)
                # Conservative version: only voxels that exist in both slices before and after
                fill_mask = np.logical_and(prev, next_)

                # Assign to seg_out
                seg_out[:, y, :][fill_mask] = label

                # Optional: Fill full union instead
                # fill_mask = np.logical_or(prev, next_)
                # seg_out[:, y, :][fill_mask] = label

                y += 1  # Skip ahead to avoid chaining errors
            else:
                y += 1

    return seg_out



# 6) (NEW) Side-aware dominance enforcement (final refinement)
def enforce_dominant_label_per_slice_side(seg):
    seg_out = seg.copy()
    x_dim, y_dim, z_dim = seg.shape
    mid_x = x_dim // 2
    for y in range(y_dim):
        for x_start, x_end in [(0, mid_x), (mid_x, x_dim)]:
            region = seg[x_start:x_end, y, :]
            flat = region[region > 0]
            if len(flat) == 0:
                continue
            labels, counts = np.unique(flat, return_counts=True)
            if len(labels) <= 1:
                continue
            dominant = labels[np.argmax(counts)]
            for label in labels:
                if label != dominant:
                    mask = (seg_out[x_start:x_end, y, :] == label)
                    seg_out[x_start:x_end, y, :][mask] = dominant
    return seg_out




def fill_all_label_gaps_with_interpolation(seg, label_order=[1,2,3,4,5], max_gap=6, min_voxels=10):
    seg_out = seg.copy()
    mid_x = seg.shape[0] // 2

    def get_label_centroids_and_sizes(label_id, right_side):
        side_mask = np.zeros_like(seg, dtype=bool)
        if right_side:
            side_mask[mid_x:, :, :] = True
        else:
            side_mask[:mid_x, :, :] = True

        label_mask = (seg == label_id) & side_mask
        centroids = []
        sizes = {}

        for y in range(seg.shape[1]):
            slice_mask = label_mask[:, y, :]
            if np.sum(slice_mask) > min_voxels:
                centroid = center_of_mass(slice_mask)
                centroids.append((y, centroid))
                sizes[y] = np.sum(slice_mask)
        return centroids, sizes

    def interpolate_centroids(centroid1, centroid2, y1, y2):
        ys = np.arange(y1 + 1, y2)
        interpolated = []
        for y in ys:
            alpha = (y - y1) / (y2 - y1)
            interp = (
                (1 - alpha) * np.array(centroid1) + alpha * np.array(centroid2)
            )
            interpolated.append((y, interp))
        return interpolated

    for side, right in zip(["left", "right"], [False, True]):
        for i in range(len(label_order) - 1):
            label_a = label_order[i]
            label_b = label_order[i + 1]

            c_a, s_a = get_label_centroids_and_sizes(label_a, right)
            c_b, s_b = get_label_centroids_and_sizes(label_b, right)

            centroids = sorted(c_a + c_b, key=lambda x: x[0])
            sizes = {**s_a, **s_b}

            for j in range(len(centroids) - 1):
                y1, c1 = centroids[j]
                y2, c2 = centroids[j + 1]
                gap = y2 - y1 - 1
                if gap <= 0 or gap > max_gap:
                    continue

                interpolated = interpolate_centroids(c1, c2, y1, y2)

                a1 = sizes.get(y1, 10)
                a2 = sizes.get(y2, 10)

                for y_interp, (x, z) in interpolated:
                    distance_to_a1 = abs(y_interp - y1)
                    distance_to_a2 = abs(y_interp - y2)
                    total_dist = distance_to_a1 + distance_to_a2 + 1e-6

                    w1 = distance_to_a2 / total_dist
                    w2 = distance_to_a1 / total_dist
                    blended_area = w1 * a1 + w2 * a2
                    radius = max(1, int(np.sqrt(blended_area / np.pi)))

                    x, z = int(round(x)), int(round(z))
                    for dx in range(-radius, radius + 1):
                        for dz in range(-radius, radius + 1):
                            if dx**2 + dz**2 <= radius**2:
                                xi, zi = x + dx, z + dz
                                if 0 <= xi < seg.shape[0] and 0 <= zi < seg.shape[2]:
                                    seg_out[xi, y_interp, zi] = label_a

    return seg_out


def fill_holes_coronal(segmentation):
    filled = segmentation.copy()
    for y in range(segmentation.shape[1]):
        for label_id in np.unique(segmentation):
            if label_id == 0:
                continue
            mask = segmentation[:, y, :] == label_id
            filled_mask = binary_fill_holes(mask)
            filled[:, y, :][filled_mask] = label_id
    return filled



def expand_last_slice_of_label3_before_chiasm(seg, min_voxels=10, num_slices=3):
    seg_out = seg.copy()
    y_dim = seg.shape[1]
    x_dim = seg.shape[0]
    mid_x = x_dim // 2

    for side, x_range in [('left', (0, mid_x)), ('right', (mid_x, x_dim))]:
        # Get all Y slices where label 3 is present on this side
        label3_slices = [y for y in range(y_dim) if np.sum(seg[x_range[0]:x_range[1], y, :] == 3) > min_voxels]
        label4_slices = [y for y in range(y_dim) if np.sum(seg[x_range[0]:x_range[1], y, :] == 4) > min_voxels]

        if not label3_slices or not label4_slices:
            continue

        last_label3_y = max(label3_slices)
        first_label4_y = min(label4_slices)

        # Only expand if chiasm starts right after label 3
        if first_label4_y - last_label3_y <= 2:
            # Collect last few label 3 slices before chiasm
            shapes = []
            for offset in range(num_slices):
                y = last_label3_y - offset
                if y < 0:
                    continue
                mask = seg[x_range[0]:x_range[1], y, :] == 3
                if np.sum(mask) > 0:
                    shapes.append(mask)

            if not shapes:
                continue

            # Median shape
            median_mask = np.median(np.stack(shapes).astype(np.uint8), axis=0) > 0.5

            # Expand the last label 3 slice
            seg_out[x_range[0]:x_range[1], last_label3_y, :][median_mask] = 3

    return seg_out



def propagate_last_label3_slice_with_shape(seg, label_id=3, next_label_id=4, n_slices=2, min_voxels=5):
    """
    Propagate the shape of the last few slices of label 3 to improve anatomical plausibility
    at the transition to label 4 (chiasm), separately for left and right sides.
    """
    seg_out = seg.copy()
    mid_x = seg.shape[0] // 2
    y_dim = seg.shape[1]

    for side, x_range in zip(["left", "right"], [(0, mid_x), (mid_x, seg.shape[0])]):
        # Collect the last few valid slices of label 3 before label 4
        label3_slices = []
        for y in range(y_dim - 1):
            curr = (seg_out[x_range[0]:x_range[1], y, :] == label_id)
            next_ = (seg_out[x_range[0]:x_range[1], y + 1, :] == next_label_id)
            if np.any(curr) and np.any(next_):
                # Backtrack to get previous n_slices of label 3
                for k in range(n_slices):
                    y_back = y - k
                    if y_back >= 0:
                        mask = (seg_out[x_range[0]:x_range[1], y_back, :] == label_id)
                        if np.sum(mask) >= min_voxels:
                            label3_slices.append((y_back, mask.copy()))
                break  # only fill once at transition

        if len(label3_slices) < n_slices:
            continue  # not enough slices to propagate

        # Use the centroid direction to guide propagation
        centroids = []
        for y_val, mask in label3_slices:
            centroid = center_of_mass(mask)
            centroids.append((y_val, centroid))

        # Fit a line and extrapolate one more slice
        y_vals = np.array([y for y, _ in centroids])
        cz_vals = np.array([c[1] for _, c in centroids])
        cx_vals = np.array([c[0] for _, c in centroids])

        # Linear fit
        poly_z = np.polyfit(y_vals, cz_vals, 1)
        poly_x = np.polyfit(y_vals, cx_vals, 1)

        y_target = max(y_vals) + 1
        z_pred = np.polyval(poly_z, y_target)
        x_pred = np.polyval(poly_x, y_target)

        # Use most recent slice as shape to propagate
        shape_mask = label3_slices[0][1]
        radius = int(np.sqrt(np.sum(shape_mask) / np.pi))

        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx**2 + dz**2 <= radius**2:
                    xi = int(round(x_pred)) + dx
                    zi = int(round(z_pred)) + dz
                    if x_range[0] <= xi < x_range[1] and 0 <= zi < seg.shape[2] and 0 <= y_target < seg.shape[1]:
                        seg_out[xi, y_target, zi] = label_id

    return seg_out




# Processing pipeline
def postprocess(seg):
     seg = fill_all_label_gaps_with_interpolation(seg, label_order=[1,2,3,4,5], max_gap=6)
     seg = remove_small_clusters(seg, min_size=4)
     seg = enforce_chiasm_sandwich(seg)
     seg = per_slice_side_cleanup(seg)
     #seg = fix_slicewise_label_flips(seg, max_gap=3)
     seg = fill_label_slice_gaps(seg, max_gap=10)
     seg = enforce_dominant_label_per_slice_side(seg)
     seg = fill_holes_coronal(seg)
     #seg =  expand_last_slice_of_label3_before_chiasm(seg, min_voxels=25, num_slices=3)
     seg =  propagate_last_label3_slice_with_shape(seg, label_id=3, next_label_id=4, n_slices=2, min_voxels=5)
     return seg


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        SPLIT LABELS FOR AVP TOOLBOX                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Label definitions and remapping
LABEL_MAP = {1: 2, 2: 4, 3: 6, 4: 8, 5: 10}  # Map original labels to output labels
ON_SEG_LABELS = [1, 2, 3]  # For ONR and ONL
ONC_LABEL = 4
OT_LABEL = 5

# Labels for chiasm splitting
LABEL_CHIASMA = 8     # chiasma intero nell'input
LABEL_LEFT = 9        # emichiasma sinistro (output)
LABEL_RIGHT = 8       # emichiasma destro (output, rimane 8)

FILENAMES = ["onc.nii.gz", "onl.nii.gz", "onr.nii.gz", "otl.nii.gz", "otr.nii.gz"]


def get_midline_from_chiasm(data):
    chiasm_coords = np.argwhere(data == ONC_LABEL)
    if chiasm_coords.size == 0:
        return data.shape[0] // 2  # fallback
    return int(np.round(center_of_mass(data == ONC_LABEL)[0]))


def split_chiasm_by_bbox_midline(onc_path):
    """
    Carica onc.nii.gz, divide la label 8 in due usando la midline del bounding box
    lungo l'asse x. Restituisce (data_splittata, affine, header).
    """
    img = nib.load(onc_path)
    data = img.get_fdata().astype(np.uint8)
    affine = img.affine
    header = img.header

    mask8 = (data == LABEL_CHIASMA)
    if not np.any(mask8):
        # Nessun chiasma: ritorna com'è
        return data, affine, header

    # Bounding box lungo x per i voxel della label 8
    xs = np.where(mask8)[0]
    x_min, x_max = xs.min(), xs.max()
    mid_bbox = (x_min + x_max) // 2

    # Copia e split
    out = data.copy()
    # Tutto ciò che è label 8 e sta a sinistra/uguale alla midline diventa 9 (sinistra)
    left_mask = mask8 & (np.arange(data.shape[0])[:, None, None] <= mid_bbox)
    out[left_mask] = LABEL_LEFT
    # Tutto ciò che è label 8 e sta a destra resta 8 (destro)
    # (già 8, quindi non serve riassegnare)

    # Fallback se uno dei lati è vuoto: usa midline della matrice
    if not np.any(out == LABEL_LEFT) or not np.any(out == LABEL_RIGHT):
        mid_global = data.shape[0] // 2
        out = data.copy()
        left_mask = mask8 & (np.arange(data.shape[0])[:, None, None] <= mid_global)
        out[left_mask] = LABEL_LEFT
        # destra resta 8

    return out, affine, header


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         DICOM SUPPORT                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _is_dicom_file(filepath):
    """Check if a file is DICOM by reading the magic bytes (DICM at offset 128)."""
    try:
        with open(filepath, 'rb') as f:
            f.seek(128)
            magic = f.read(4)
            if magic == b'DICM':
                return True
        # Some old DICOM files lack the preamble — try reading as pydicom
        if HAS_PYDICOM:
            try:
                pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
                return True
            except Exception:
                return False
    except Exception:
        return False
    return False


def _find_dicom_files(input_folder):
    """Recursively find all DICOM files in input_folder.
    Returns a list of file paths."""
    dicom_files = []
    for root, dirs, files in os.walk(input_folder):
        for f in files:
            # Skip hidden files and already-converted NIfTI
            if f.startswith('.') or f.endswith('.nii.gz') or f.endswith('.nii'):
                continue
            fpath = os.path.join(root, f)
            if _is_dicom_file(fpath):
                dicom_files.append(fpath)
    return dicom_files


def _group_dicom_by_series(dicom_files):
    """Group DICOM files by SeriesInstanceUID.
    Returns dict: {series_uid: [list of file paths]}"""
    series_dict = defaultdict(list)
    for fpath in tqdm(dicom_files, desc="  Reading DICOM headers"):
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            series_uid = str(getattr(ds, 'SeriesInstanceUID', 'unknown'))
            series_dict[series_uid].append(fpath)
        except Exception as e:
            logger.warning(f"Could not read {fpath}: {e}")
    return dict(series_dict)


def _get_series_description(dicom_files_for_series):
    """Try to build a human-readable name for a DICOM series."""
    ds = pydicom.dcmread(dicom_files_for_series[0], stop_before_pixels=True, force=True)
    patient_id = str(getattr(ds, 'PatientID', 'unknown')).strip()
    series_desc = str(getattr(ds, 'SeriesDescription', '')).strip()
    series_num = str(getattr(ds, 'SeriesNumber', '')).strip()
    # Build a clean filename
    parts = [p for p in [patient_id, series_num, series_desc] if p]
    name = '_'.join(parts)
    # Sanitize for filesystem
    name = re.sub(r'[^\w\-.]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name if name else 'dicom_series'


def _sort_dicom_slices(dicom_files_for_series):
    """Sort DICOM slices by ImagePositionPatient[2] (slice position along Z),
    falling back to InstanceNumber."""
    slices_with_pos = []
    for fpath in dicom_files_for_series:
        ds = pydicom.dcmread(fpath, force=True)
        # Try ImagePositionPatient (most reliable)
        ipp = getattr(ds, 'ImagePositionPatient', None)
        if ipp is not None:
            sort_key = float(ipp[2])
        else:
            sort_key = float(getattr(ds, 'InstanceNumber', 0))
        slices_with_pos.append((sort_key, fpath, ds))
    slices_with_pos.sort(key=lambda x: x[0])
    return slices_with_pos


def _build_affine_from_multiframe(ds, n_slices):
    """Build a NIfTI-compatible 4x4 affine from a multi-frame (enhanced) DICOM.
    Enhanced DICOMs store orientation/position in functional group sequences
    rather than top-level attributes."""

    # --- Orientation (ImageOrientationPatient) ---
    iop = None
    ps = None
    sfgs = getattr(ds, 'SharedFunctionalGroupsSequence', None)
    if sfgs:
        sfg = sfgs[0]
        pos_seq = getattr(sfg, 'PlaneOrientationSequence', None)
        if pos_seq:
            iop = [float(x) for x in pos_seq[0].ImageOrientationPatient]
        pms = getattr(sfg, 'PixelMeasuresSequence', None)
        if pms:
            ps_attr = getattr(pms[0], 'PixelSpacing', None)
            if ps_attr:
                ps = [float(x) for x in ps_attr]
    # Fallback to top-level
    if iop is None:
        iop = getattr(ds, 'ImageOrientationPatient', [1, 0, 0, 0, 1, 0])
        iop = [float(x) for x in iop]
    if ps is None:
        ps_attr = getattr(ds, 'PixelSpacing', [1.0, 1.0])
        ps = [float(x) for x in ps_attr]

    row_cosine = np.array(iop[:3])
    col_cosine = np.array(iop[3:])
    row_spacing = float(ps[0])   # between rows → column direction
    col_spacing = float(ps[1])   # between cols → row direction

    # --- Slice positions from PerFrameFunctionalGroupsSequence ---
    ipp_first = None
    ipp_last = None
    pffgs = getattr(ds, 'PerFrameFunctionalGroupsSequence', None)
    if pffgs and len(pffgs) > 0:
        pps_first = getattr(pffgs[0], 'PlanePositionSequence', None)
        if pps_first:
            ipp_first = np.array([float(x) for x in pps_first[0].ImagePositionPatient])
        if len(pffgs) > 1:
            pps_last = getattr(pffgs[-1], 'PlanePositionSequence', None)
            if pps_last:
                ipp_last = np.array([float(x) for x in pps_last[0].ImagePositionPatient])
    # Fallback to top-level
    if ipp_first is None:
        ipp_attr = getattr(ds, 'ImagePositionPatient', [0, 0, 0])
        ipp_first = np.array([float(x) for x in ipp_attr])

    # --- Slice direction ---
    if ipp_last is not None and n_slices > 1:
        slice_dir = (ipp_last - ipp_first) / max(n_slices - 1, 1)
    else:
        st = None
        if sfgs:
            pms = getattr(sfgs[0], 'PixelMeasuresSequence', None)
            if pms:
                st = getattr(pms[0], 'SpacingBetweenSlices', None)
                if st is None:
                    st = getattr(pms[0], 'SliceThickness', None)
        if st is None:
            st = getattr(ds, 'SpacingBetweenSlices',
                         getattr(ds, 'SliceThickness', 1.0))
        st = float(st)
        slice_normal = np.cross(row_cosine, col_cosine)
        slice_dir = slice_normal * st

    # --- Build affine (same convention as single-frame) ---
    affine = np.eye(4)
    affine[:3, 0] = col_cosine * row_spacing   # dim0 (rows): column direction, ps[0]
    affine[:3, 1] = row_cosine * col_spacing   # dim1 (cols): row direction,    ps[1]
    affine[:3, 2] = slice_dir
    affine[:3, 3] = ipp_first

    # LPS → RAS
    affine[0, :] *= -1  # L → R
    affine[1, :] *= -1  # P → A

    return affine


def _build_affine_from_dicom(sorted_slices):
    """Build a NIfTI-compatible 4x4 affine matrix from sorted DICOM slices."""
    ds_first = sorted_slices[0][2]
    ds_last = sorted_slices[-1][2]
    n_slices = len(sorted_slices)

    # Image orientation
    iop = getattr(ds_first, 'ImageOrientationPatient', [1, 0, 0, 0, 1, 0])
    iop = [float(x) for x in iop]
    row_cosine = np.array(iop[:3])
    col_cosine = np.array(iop[3:])

    # Pixel spacing — DICOM convention:
    #   PixelSpacing[0] = row spacing   = distance between rows   = spacing in column direction
    #   PixelSpacing[1] = col spacing   = distance between cols   = spacing in row direction
    ps = getattr(ds_first, 'PixelSpacing', [1.0, 1.0])
    row_spacing = float(ps[0])  # between rows → step size along column direction
    col_spacing = float(ps[1])  # between cols → step size along row direction

    # Slice direction and spacing
    ipp_first = getattr(ds_first, 'ImagePositionPatient', [0, 0, 0])
    ipp_first = np.array([float(x) for x in ipp_first])

    if n_slices > 1:
        ipp_last = getattr(ds_last, 'ImagePositionPatient', None)
        if ipp_last is not None:
            ipp_last = np.array([float(x) for x in ipp_last])
            slice_dir = (ipp_last - ipp_first) / max(n_slices - 1, 1)
        else:
            st = float(getattr(ds_first, 'SpacingBetweenSlices',
                       getattr(ds_first, 'SliceThickness', 1.0)))
            slice_normal = np.cross(row_cosine, col_cosine)
            slice_dir = slice_normal * st
    else:
        st = float(getattr(ds_first, 'SliceThickness', 1.0))
        slice_normal = np.cross(row_cosine, col_cosine)
        slice_dir = slice_normal * st

    # Build affine in DICOM LPS patient coordinates.
    #
    # CRITICAL axis mapping for pixel_array shape (rows, cols):
    #   - NIfTI dim 0 = DICOM rows.  Row index increases along the COLUMN
    #     direction (col_cosine), with spacing = PixelSpacing[0] (row_spacing).
    #   - NIfTI dim 1 = DICOM cols.  Col index increases along the ROW
    #     direction (row_cosine), with spacing = PixelSpacing[1] (col_spacing).
    #   - NIfTI dim 2 = slices.
    #
    affine = np.eye(4)
    affine[:3, 0] = col_cosine * row_spacing   # dim0 (rows): column direction, ps[0]
    affine[:3, 1] = row_cosine * col_spacing   # dim1 (cols): row direction,    ps[1]
    affine[:3, 2] = slice_dir
    affine[:3, 3] = ipp_first

    # CRITICAL: DICOM uses LPS coordinates (Left-Posterior-Superior),
    # but NIfTI uses RAS (Right-Anterior-Superior).
    # Negate rows 0 and 1 to convert LPS → RAS so that nibabel
    # correctly interprets axis directions (just like dcm2niix does).
    affine[0, :] *= -1  # L → R
    affine[1, :] *= -1  # P → A

    return affine


def _dicom_series_to_nifti(sorted_slices, output_path):
    """Convert a sorted list of DICOM slices into a NIfTI file.
    Handles both classic single-frame DICOM series and enhanced multi-frame
    DICOMs (where a single .dcm contains the entire 3D volume).
    sorted_slices: list of (sort_key, filepath, pydicom_dataset)"""

    ds_first = sorted_slices[0][2]
    n_frames = int(getattr(ds_first, 'NumberOfFrames', 1))

    # ---- Multi-frame (enhanced) DICOM: one file holds all slices ----
    if n_frames > 1 or (len(sorted_slices) == 1 and ds_first.pixel_array.ndim == 3):
        volume = ds_first.pixel_array.astype(np.float32)  # (frames, rows, cols)
        # Apply rescale
        slope = float(getattr(ds_first, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds_first, 'RescaleIntercept', 0.0))
        if slope != 1.0 or intercept != 0.0:
            volume = volume * slope + intercept
        # Transpose to (rows, cols, slices) — NIfTI convention
        volume = np.transpose(volume, (1, 2, 0))
        affine = _build_affine_from_multiframe(ds_first, volume.shape[2])

    # ---- Classic single-frame series: one file per slice ----
    else:
        pixel_arrays = []
        for _, _, ds in sorted_slices:
            arr = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, 'RescaleSlope', 1.0))
            intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
            if slope != 1.0 or intercept != 0.0:
                arr = arr * slope + intercept
            pixel_arrays.append(arr)
        volume = np.stack(pixel_arrays, axis=-1)  # (rows, cols, slices)
        affine = _build_affine_from_dicom(sorted_slices)

    # Ensure 3D
    if volume.ndim > 3:
        volume = volume.squeeze()
    if volume.ndim == 2:
        volume = volume[:, :, np.newaxis]

    # Create NIfTI and reorient to RAS canonical orientation
    nii = nib.Nifti1Image(volume.astype(np.float32), affine)
    nii = reorient_to_canonical(nii)
    nib.save(nii, output_path)
    return output_path


def convert_dicom_in_input_folder(input_folder):
    """Scan input folder for DICOM files, convert each series to .nii.gz.
    Converted files are saved directly in input_folder.
    Returns the number of series converted."""

    logger.info("Scanning input folder for DICOM files...")
    dicom_files = _find_dicom_files(input_folder)

    if not dicom_files:
        logger.info("No DICOM files found — using NIfTI files only.")
        return 0

    if not HAS_PYDICOM:
        logger.warning("DICOM files detected but pydicom is not installed!")
        logger.warning("Install it with: pip install pydicom")
        logger.warning("Or convert your DICOM files to NIfTI (.nii.gz) externally.")
        return 0

    logger.info(f"Found {len(dicom_files)} DICOM files. Grouping by series...")
    series_dict = _group_dicom_by_series(dicom_files)
    logger.info(f"Found {len(series_dict)} DICOM series.")

    converted = 0
    for series_uid, files in series_dict.items():
        series_name = _get_series_description(files)
        output_path = os.path.join(input_folder, f"{series_name}.nii.gz")

        # Skip if already converted
        if os.path.exists(output_path):
            logger.info(f"Skipping {series_name}.nii.gz — already exists")
            continue

        # Avoid name collisions
        counter = 1
        while os.path.exists(output_path):
            output_path = os.path.join(input_folder, f"{series_name}_{counter}.nii.gz")
            counter += 1

        try:
            sorted_slices = _sort_dicom_slices(files)
            # Detect multi-frame DICOMs for accurate logging
            ds_check = sorted_slices[0][2]
            n_frames = int(getattr(ds_check, 'NumberOfFrames', 1))
            if n_frames > 1:
                logger.info(f"Converting series: {series_name} (multi-frame, {n_frames} frames)...")
            else:
                logger.info(f"Converting series: {series_name} ({len(files)} slices)...")
            _dicom_series_to_nifti(sorted_slices, output_path)
            logger.info(f"Saved: {os.path.basename(output_path)}")
            converted += 1
        except Exception as e:
            logger.error(f"Failed to convert series {series_name}: {e}")

    logger.info(f"DICOM conversion complete: {converted} series converted to NIfTI.")
    return converted


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                            MAIN PIPELINE                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_pipeline():
    """Execute the full AVP segmentation pipeline."""

    # --- Resolve all paths ---
    resolve_paths()

    # ================================================================
    # STEP 0: DICOM DETECTION & CONVERSION
    # ================================================================
    logger.info("STEP 0: Input Detection (NIfTI / DICOM)")
    convert_dicom_in_input_folder(INPUT_FOLDER)

    # Verify we have at least one .nii.gz to process
    nifti_inputs = glob.glob(os.path.join(INPUT_FOLDER, "*.nii.gz"))
    if not nifti_inputs:
        logger.error(f"No .nii.gz files found in input folder: {INPUT_FOLDER}")
        logger.error("Place NIfTI (.nii.gz) or DICOM files in the input/ folder.")
        sys.exit(1)
    logger.info(f"Ready to process {len(nifti_inputs)} NIfTI volume(s).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_shape = TARGET_SHAPE

    # ================================================================
    # STEP 1: BASE MODEL PREDICTIONS
    # ================================================================
    logger.info("STEP 1/6: Base Model Predictions (VNet3D ensemble)")

    for rs_id in RS_IDS:
        rs_label = f"RS{rs_id}"
        logger.info(f"Processing model: {rs_label}")
        model_path = _get_base_model_path(rs_id)
        output_folder = os.path.join(OUTPUT_FOLDER, "RAW", rs_label)
        os.makedirs(output_folder, exist_ok=True)

        if not os.path.exists(model_path):
            logger.warning(f"Skipping {rs_label}: checkpoint not found at {model_path}")
            continue

        # Load model
        model = VNet3D(in_channels=1, out_channels=10)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()

        # Predict on all test images
        # Use identity affine for all outputs (consistent with training data)
        identity_affine = np.eye(4)

        with torch.no_grad():
            for nifti_file in glob.glob(os.path.join(INPUT_FOLDER, "*.nii.gz")):
                img_nii = nib.load(nifti_file)
                img_nii = reorient_to_canonical(img_nii)  # Force RAS orientation
                img = img_nii.get_fdata().astype(np.float32)

                # Safety: ensure 3D (squeeze extra dims, skip 2D)
                if img.ndim > 3:
                    img = img.squeeze()
                if img.ndim != 3:
                    base_name = os.path.basename(nifti_file).replace(".nii.gz", "")
                    logger.warning(f"Skipping {base_name}: unexpected shape {img.shape} (need 3D)")
                    continue

                img = normalize(img)
                img_cropped = crop_or_pad(img, target_shape)
                img_tensor = torch.from_numpy(img_cropped).unsqueeze(0).unsqueeze(0).to(device)

                main_out, _ = model(img_tensor)
                pred = torch.argmax(main_out, dim=1).squeeze().cpu().numpy().astype(np.uint8)

                base_name = os.path.basename(nifti_file).replace(".nii.gz", "")
                pred_path = os.path.join(output_folder, f"{base_name}_pred.nii.gz")
                crop_path = os.path.join(output_folder, f"{base_name}_cropped.nii.gz")

                nib.save(nib.Nifti1Image(pred, identity_affine), pred_path)
                nib.save(nib.Nifti1Image(img_cropped.astype(np.float32), identity_affine), crop_path)

                logger.info(f"  {base_name}: saved prediction and cropped image")

    # ================================================================
    # STEP 2: REFINEMENT
    # ================================================================
    logger.info("STEP 2/6: Refinement (UNet3D)")

    for rs_id in RS_IDS:
        rs_label = f"RS{rs_id}"
        refine_ckpt = _get_refine_model_path(rs_id)
        output_folder = os.path.join(OUTPUT_FOLDER, "RAW", rs_label)

        if not os.path.exists(refine_ckpt):
            logger.warning(f"Skipping {rs_label}: refinement checkpoint not found at {refine_ckpt}")
            continue

        logger.info(f"Refining predictions for {rs_label} using refinement model...")

        # Load refinement model
        refine_model = UNet3D(in_channels=2, out_channels=6)
        refine_model.load_state_dict(torch.load(refine_ckpt, map_location=device, weights_only=True))
        refine_model.to(device)
        refine_model.eval()

        # Loop over base predictions
        for base_pred_path in glob.glob(os.path.join(output_folder, "*_pred.nii.gz")):
            base_name = os.path.basename(base_pred_path).replace("_pred.nii.gz", "")
            ciss_path = os.path.join(output_folder, f"{base_name}_cropped.nii.gz")
            refined_path = os.path.join(output_folder, f"{base_name}_refined_pred.nii.gz")

            if not os.path.exists(ciss_path):
                logger.warning(f"Skipping {base_name}: missing cropped CISS at {ciss_path}")
                continue

            # Load inputs (all cropped files already have identity affine)
            ciss = nib.load(ciss_path).get_fdata().astype(np.float32)
            pred = nib.load(base_pred_path).get_fdata().astype(np.float32)
            affine = np.eye(4)  # identity affine, consistent with training data

            ciss_norm = normalize(ciss)
            input_tensor = np.stack([ciss_norm, pred], axis=0)
            input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(device)

            # Run refinement
            with torch.no_grad():
                output = refine_model(input_tensor)
                refined_pred = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)

            nib.save(nib.Nifti1Image(refined_pred, affine), refined_path)
            logger.info(f"Refined: {base_name}")

    # ================================================================
    # STEP 3: MERGE SEGMENTATIONS (MAJORITY VOTING WITH THRESHOLD)
    # ================================================================
    logger.info(f"STEP 3/6: Majority Voting Fusion (threshold={VOTING_THRESHOLD})")

    patient_predictions = defaultdict(list)

    for rs_id in RS_IDS:
        rs_label = f"RS{rs_id}"
        raw_folder = os.path.join(OUTPUT_FOLDER, "RAW", rs_label)
        refined_preds = glob.glob(os.path.join(raw_folder, "*_refined_pred.nii.gz"))
        for p in refined_preds:
            base_name = os.path.basename(p).replace("_refined_pred.nii.gz", "")
            patient_predictions[base_name].append(p)

    preprocess_dir = os.path.join(OUTPUT_FOLDER, "PREDICTIONS", "PREPROCESS")
    os.makedirs(preprocess_dir, exist_ok=True)

    for pid, paths in sorted(patient_predictions.items()):
        logger.info(f"Merging patient {pid} using {len(paths)} segmentations...")

        seg_list = [nib.load(p).get_fdata().astype(np.uint8) for p in paths]
        seg_stack = np.stack(seg_list, axis=0)  # Shape: (N, X, Y, Z)
        N = seg_stack.shape[0]

        label_votes = np.zeros(seg_stack[0].shape + (6,), dtype=np.uint16)  # labels 0-5

        for label in range(6):
            label_votes[..., label] = np.sum(seg_stack == label, axis=0)

        # Find label with max votes
        max_votes = np.max(label_votes, axis=-1)
        best_label = np.argmax(label_votes, axis=-1)

        # Apply threshold
        threshold = int(np.ceil(VOTING_THRESHOLD * N))
        final_seg = np.where(max_votes >= threshold, best_label, 0).astype(np.uint8)

        # Save with identity affine (consistent with training data)
        out_path = os.path.join(preprocess_dir, f"{pid}_seg_preprocess.nii.gz")
        nib.save(nib.Nifti1Image(final_seg, np.eye(4)), out_path)

        logger.info(f"Saved: {out_path}")

    # Copy cropped CISS images to PREPROCESS folder so users can overlay results
    logger.info("Copying cropped CISS images to PREPROCESS folder...")
    for pid in patient_predictions.keys():
        # Find the cropped CISS from the first available RS folder
        cropped_src = None
        for rs_id in RS_IDS:
            candidate = os.path.join(OUTPUT_FOLDER, "RAW", f"RS{rs_id}", f"{pid}_cropped.nii.gz")
            if os.path.exists(candidate):
                cropped_src = candidate
                break
        if cropped_src:
            cropped_dst = os.path.join(preprocess_dir, f"{pid}_CISS_cropped.nii.gz")
            shutil.copy2(cropped_src, cropped_dst)
            logger.info(f"Copied cropped CISS for {pid}")
        else:
            logger.warning(f"Cropped CISS not found for {pid}")

    logger.info(f"All pre-process segmentations saved to: {preprocess_dir}")

    # ================================================================
    # STEP 4: POST-PROCESSING
    # ================================================================
    logger.info("STEP 4/6: Post-processing")

    postprocess_dir = os.path.join(OUTPUT_FOLDER, "PREDICTIONS", "POSTPROCESS")
    os.makedirs(postprocess_dir, exist_ok=True)

    # Only post-process segmentation files (not CISS images)
    seg_files = [f for f in os.listdir(preprocess_dir) if f.endswith("_seg_preprocess.nii.gz")]
    logger.info(f"Found {len(seg_files)} files to post-process.")

    for fname in tqdm(seg_files):
        input_path = os.path.join(preprocess_dir, fname)
        pid = fname.replace("_seg_preprocess.nii.gz", "")
        output_path = os.path.join(postprocess_dir, f"{pid}_seg_postprocess.nii.gz")

        seg = nib.load(input_path)
        data = seg.get_fdata().astype(np.uint8)

        post = postprocess(data)
        nib.save(nib.Nifti1Image(post.astype(np.uint8), np.eye(4)), output_path)

    # Copy cropped CISS images to POSTPROCESS folder
    logger.info("Copying cropped CISS images to POSTPROCESS folder...")
    for fname in seg_files:
        pid = fname.replace("_seg_preprocess.nii.gz", "")
        cropped_src = os.path.join(preprocess_dir, f"{pid}_CISS_cropped.nii.gz")
        if os.path.exists(cropped_src):
            cropped_dst = os.path.join(postprocess_dir, f"{pid}_CISS_cropped.nii.gz")
            shutil.copy2(cropped_src, cropped_dst)

    logger.info(f"Saved postprocessed segmentations to: {postprocess_dir}")

    # ================================================================
    # STEP 5: SPLIT LABELS FOR AVP TOOLBOX
    # ================================================================
    logger.info("STEP 5/6: Split Labels for AVP Toolbox")

    split_output_base = os.path.join(OUTPUT_FOLDER, "PREDICTIONS", "split_structures")
    os.makedirs(split_output_base, exist_ok=True)

    for fname in tqdm(os.listdir(postprocess_dir)):
        if not fname.endswith("_seg_postprocess.nii.gz"):
            continue

        subj_id = fname.replace("_seg_postprocess.nii.gz", "")
        subj_folder = os.path.join(split_output_base, subj_id)
        os.makedirs(subj_folder, exist_ok=True)

        path = os.path.join(postprocess_dir, fname)
        seg = nib.load(path)
        data = seg.get_fdata().astype(np.uint8)
        affine = np.eye(4)  # identity affine, consistent with training data

        midline = get_midline_from_chiasm(data)
        left_mask = np.zeros_like(data, dtype=bool)
        right_mask = np.zeros_like(data, dtype=bool)
        left_mask[:midline, :, :] = True
        right_mask[midline:, :, :] = True

        # Initialize outputs
        onr = np.zeros_like(data, dtype=np.uint8)
        onl = np.zeros_like(data, dtype=np.uint8)
        onc = np.zeros_like(data, dtype=np.uint8)
        otr = np.zeros_like(data, dtype=np.uint8)
        otl = np.zeros_like(data, dtype=np.uint8)

        # Populate with remapped labels
        for orig_label, new_label in LABEL_MAP.items():
            mask = data == orig_label
            if orig_label in ON_SEG_LABELS:
                onr[np.logical_and(mask, right_mask)] = new_label
                onl[np.logical_and(mask, left_mask)] = new_label
            elif orig_label == ONC_LABEL:
                onc[mask] = new_label
            elif orig_label == OT_LABEL:
                otr[np.logical_and(mask, right_mask)] = new_label
                otl[np.logical_and(mask, left_mask)] = new_label

        # Save
        nib.save(nib.Nifti1Image(onr, affine), os.path.join(subj_folder, "onr.nii.gz"))
        nib.save(nib.Nifti1Image(onl, affine), os.path.join(subj_folder, "onl.nii.gz"))
        nib.save(nib.Nifti1Image(onc, affine), os.path.join(subj_folder, "onc.nii.gz"))
        nib.save(nib.Nifti1Image(otr, affine), os.path.join(subj_folder, "otr.nii.gz"))
        nib.save(nib.Nifti1Image(otl, affine), os.path.join(subj_folder, "otl.nii.gz"))

    # Copy cropped CISS images into each subject's split folder
    logger.info("Copying cropped CISS images to split_structures subject folders...")
    for fname in os.listdir(postprocess_dir):
        if not fname.endswith("_seg_postprocess.nii.gz"):
            continue
        subj_id = fname.replace("_seg_postprocess.nii.gz", "")
        cropped_src = os.path.join(postprocess_dir, f"{subj_id}_CISS_cropped.nii.gz")
        if os.path.exists(cropped_src):
            subj_folder = os.path.join(split_output_base, subj_id)
            cropped_dst = os.path.join(subj_folder, f"{subj_id}_CISS_cropped.nii.gz")
            shutil.copy2(cropped_src, cropped_dst)
            logger.info(f"Copied cropped CISS to {subj_folder}")

    logger.info(f"All segmentations split and saved to: {split_output_base}")

    # ================================================================
    # STEP 6: SPLIT CHIASM FOR AVP TOOLBOX
    # ================================================================
    logger.info("STEP 6/6: Split Chiasm for AVP Toolbox")

    source_root = split_output_base
    dest_root = split_output_base  # overwrite in place

    if not os.path.isdir(source_root):
        logger.error(f"Source not found: {source_root}")
    else:
        subjects = sorted([d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))])
        logger.info(f"Processing {len(subjects)} subjects...")

        for sid in subjects:
            src_subject_dir = os.path.join(source_root, sid)
            dst_subject_dir = os.path.join(dest_root, sid)
            os.makedirs(dst_subject_dir, exist_ok=True)

            for fname in FILENAMES:
                src_path = os.path.join(src_subject_dir, fname)
                dst_path = os.path.join(dst_subject_dir, fname)

                if not os.path.exists(src_path):
                    logger.warning(f"Missing: {src_path}")
                    continue

                if fname == "onc.nii.gz":
                    try:
                        out_data, affine, header = split_chiasm_by_bbox_midline(src_path)
                        nib.save(nib.Nifti1Image(out_data.astype(np.uint8), affine, header), dst_path)
                    except Exception as e:
                        logger.error(f"Failed split for {src_path}: {e}. Copying original.")
                        shutil.copy2(src_path, dst_path)
                else:
                    # Copy 1:1 (in-place, so skip if same path)
                    if os.path.abspath(src_path) != os.path.abspath(dst_path):
                        shutil.copy2(src_path, dst_path)

    # ================================================================
    # DONE
    # ================================================================
    logger.info("PIPELINE COMPLETE!")
    logger.info(f"All outputs saved to: {OUTPUT_FOLDER}")
    logger.info(f"Final split structures: {split_output_base}")


def main(path=None, input_folder=None, output_folder=None, debug=False):
    """Entry point for the AVP segmentation pipeline."""
    global PROJECT_ROOT, INPUT_FOLDER, OUTPUT_FOLDER

    if debug:
        logger.setLevel(logging.DEBUG)

    if path is not None:
        PROJECT_ROOT = path
    if input_folder is not None:
        INPUT_FOLDER = input_folder
    if output_folder is not None:
        OUTPUT_FOLDER = output_folder

    run_pipeline()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVP_SEG_2026: Anterior Visual Pathway segmentation pipeline")
    parser.add_argument("--root-dir", dest="path", default=None,
                        help="Project root folder (contains models/, input/, output/).")
    parser.add_argument("--input", dest="input_folder", default=None,
                        help="Input folder with CISS images (.nii.gz or DICOM).")
    parser.add_argument("--output", dest="output_folder", default=None,
                        help="Output folder for predictions and results.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    main(path=args.path, input_folder=args.input_folder,
         output_folder=args.output_folder, debug=args.debug)
