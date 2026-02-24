#!/usr/bin/env python3
import os, glob, re
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes
from nilearn.image import resample_to_img
import templateflow.api as tflow

# ================= CONFIG =================
IN_DIR  = "/foundcog/forrestgump/mask/memorability/test/mtl_mni2mm_simple"
OUT_DIR = "/foundcog/forrestgump/mask/memorability/test/prob_masks_mtl_bilateral_cerebrum""
THR = 0.20

CORTEX_ATLAS = "/foundcog/forrestgump/mask/Schaefer2018_100Parcels_17Networks_2mm.nii.gz"

os.makedirs(OUT_DIR, exist_ok=True)

# ================= HELPER =================
def tf_get_one(**kwargs):
    """TemplateFlow get() but always return a single path"""
    p = tflow.get(**kwargs)
    if isinstance(p, (list, tuple)):
        return p[0]
    return p

# ================= LOAD MNI REFERENCE =================
ref_path = tf_get_one(
    template="MNI152NLin6Asym",
    resolution=2,
    desc="brain",
    suffix="T1w"
)
ref = nib.load(str(ref_path))
RSH, RAF = ref.shape, ref.affine

# ================= LOAD BRAIN MASK =================
brain_mask_path = tf_get_one(
    template="MNI152NLin6Asym",
    resolution=2,
    desc="brain",
    suffix="mask"
)
brain_mask = nib.load(str(brain_mask_path)).get_fdata().astype(np.uint8)

# ================= CREATE CEREBRUM MASK FROM SCHAEFER BOUNDARY =================
atlas_img = nib.load(CORTEX_ATLAS)
atlas_data = atlas_img.get_fdata()

# Get cortex mask
cortex_mask = (atlas_data > 0).astype(np.uint8)

# Dilate to create boundary
cortex_boundary = binary_dilation(cortex_mask, iterations=3).astype(np.uint8)

# Fill holes to include all subcortical structures INSIDE the cortical boundary
cerebrum_mask = binary_fill_holes(cortex_boundary).astype(np.uint8)

# Constrain to brain
cerebrum_mask *= brain_mask

print(f"Schaefer cortex: {cortex_mask.sum()} voxels")
print(f"Cortex boundary (dilated): {cortex_boundary.sum()} voxels")
print(f"Cerebrum (filled): {cerebrum_mask.sum()} voxels")
print(f"Brain total: {brain_mask.sum()} voxels")

# Save masks for inspection
nib.save(
    nib.Nifti1Image(cortex_mask, RAF, ref.header),
    os.path.join(OUT_DIR, "cortex_mask_schaefer.nii.gz")
)
nib.save(
    nib.Nifti1Image(cerebrum_mask, RAF, ref.header),
    os.path.join(OUT_DIR, "cerebrum_mask_filled.nii.gz")
)

# ================= ROI DEFINITIONS =================
#TOKENS = {"EVC": ["evc"],"LOC": ["loc"],"PPA": ["ppa"],"OPA": ["opa"],"RSC": ["rsc"],"FFA": ["ffa"],"OFA": ["ofa"],}
TOKENS={
 "Hippotail":["hippotail","hipptail","tailhipp"],   
}
def norm(s):
    return s.lower().replace("_", "").replace("-", "")

def parse_roi(fname):
    s = norm(fname)
    s = re.sub(r"\.nii(\.gz)?$", "", s)
    for roi, toks in TOKENS.items():
        for t in toks:
            if t in s:
                hemi = None
                if s.endswith(t + "l"):
                    hemi = "L"
                elif s.endswith(t + "r"):
                    hemi = "R"
                else:
                    hemi = "BILAT"
                return roi, hemi
    return None, None

# ================= GROUP FILES =================
groups = {}
for f in glob.glob(os.path.join(IN_DIR, "*.nii.gz")):
    roi, hemi = parse_roi(os.path.basename(f))
    if roi and hemi:
        groups.setdefault(roi, {}).setdefault(hemi, []).append(f)

# ================= BUILD PROBABILISTIC MASKS =================
for roi, hemi_dict in sorted(groups.items()):
    stacks = []

    for hemi in ("L", "R", "BILAT"):
        files = hemi_dict.get(hemi, [])
        if not files:
            continue

        data = []
        for f in files:
            img = nib.load(f)
            img_resamp = resample_to_img(img, ref, interpolation='nearest')
            d = (img_resamp.get_fdata() > 0).astype(np.uint8)
            d *= cerebrum_mask  # apply cerebrum mask (includes subcortical)
            data.append(d)

        stacks.append(np.stack(data, axis=0))

    if not stacks:
        print(f"[{roi}] no usable files, skipping")
        continue

    combined = np.concatenate(stacks, axis=0)
    prob = combined.mean(axis=0).astype(np.float32)
    mask = (prob >= THR).astype(np.uint8)
    mask = binary_closing(mask, iterations=1).astype(np.uint8)
    
    # Final cerebrum constraint
    mask *= cerebrum_mask

    # Save outputs
    nib.save(
        nib.Nifti1Image(prob, RAF, ref.header),
        os.path.join(OUT_DIR, f"{roi}_prob_bilat_cerebrum.nii.gz")
    )
    nib.save(
        nib.Nifti1Image(mask, RAF, ref.header),
        os.path.join(OUT_DIR, f"{roi}_mask_thr{int(THR*100)}_bilat_cerebrum.nii.gz")
    )

    nL = len(hemi_dict.get("L", []))
    nR = len(hemi_dict.get("R", []))
    nB = len(hemi_dict.get("BILAT", []))
    vox_count = mask.sum()
    print(f"[{roi}] N={nL+nR+nB} (L={nL}, R={nR}, BILAT={nB}) → {vox_count} voxels (cortex + subcortical)")

print("✓ Done.")
