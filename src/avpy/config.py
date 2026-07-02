from dataclasses import dataclass


# ── Anatomy ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SegmentDef:
    name: str
    output_label: int  # value written into proc/ NIfTI files
    slice_start: int   # first slice in normalised space
    slice_end: int     # last slice in normalised space

SEGMENTS = [
    SegmentDef("iOrb",  1,   0,  36),
    SegmentDef("iCan",  2,  37,  47),
    SegmentDef("iCran", 4,  48,  73),
    SegmentDef("OC",    8,  74,  84),
    SegmentDef("OT",   16,  85, 101),
]

# Convenient look-ups derived from SEGMENTS
LABEL_TO_NAME  = {s.output_label: s.name for s in SEGMENTS}
NAME_TO_LABEL  = {s.name: s.output_label for s in SEGMENTS}
SEGMENT_RANGES = [(s.name, s.slice_start, s.slice_end) for s in SEGMENTS]


# ── Image geometry ────────────────────────────────────────────────────────────

VOXEL_SIZE_MM     = 0.6
ORIG_TARGET_SHAPE = (256, 256, 72)  # used in _01b affine correction
NORM_TARGET_SHAPE = (250, 102, 72)  # used in _03b resampling of normalised images


# ── Normalisation parameters ──────────────────────────────────────────────────

RESOLUTION_INCREASE = 10
MAX_NERVE_SLICES    = 160                                  # in original space
MAX_SLICES          = MAX_NERVE_SLICES * RESOLUTION_INCREASE
NORMALIZED_N_SLICES = 104 * RESOLUTION_INCREASE + 1


# ── File naming ───────────────────────────────────────────────────────────────

LINEARIZE_SUFFIX     = "_linearize_4bc.nii.gz"
NORMALIZE_SUFFIX     = "_normalized_4bc.nii.gz"
ISO_LINEARIZE_SUFFIX = "_linearize_4bc_iso06.nii.gz"
ISO_NORMALIZE_SUFFIX = "_normalized_4bc_iso06.nii.gz"


# ── Directory layout ──────────────────────────────────────────────────────────

ORIG_SUBDIR      = "data/orig"
PROC_SUBDIR      = "data/proc"
SBJ_LIST_FILE    = "data/sbj.list"
RESULTS_SUBDIR   = "results"
TEMPLATES_SUBDIR = "templates"


# ── Sides / image types ───────────────────────────────────────────────────────

SIDES       = ["r", "l"]
IMAGE_TYPES = ["linearize", "normalized"]
ANAT_PREFIX = "on"


# ── Atlas ─────────────────────────────────────────────────────────────────────

ATLAS_NAME          = "aVP-24_prob50.nii.gz"
ATLAS_PCT_THRESHOLD = 50
