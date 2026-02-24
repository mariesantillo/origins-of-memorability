import os
import numpy as np
import nibabel as nib

OUT_DIR = "/foundcog/forrestgump/mask/memorability/test/prob_masks_mtl_bilateral_cerebrum/"

def load_mask(name):
    path = os.path.join(OUT_DIR, f"{name}_mask_thr20_bilat_cerebrum.nii.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing ROI mask: {path}")
    return nib.load(path)

def union_and_save(roi_names, out_name):
    imgs = [load_mask(r) for r in roi_names]
    # sanity: same grid/affine
    shapes = {img.shape for img in imgs}
    affs   = {img.affine.tobytes() for img in imgs}
    if len(shapes) != 1 or len(affs) != 1:
        raise ValueError("Masks have inconsistent shapes or affines; rebuild to a common template.")
    ref = imgs[0]
    acc = np.zeros(ref.shape, dtype=np.uint8)
    for img in imgs:
        acc |= (img.get_fdata() > 0).astype(np.uint8)
    out_path = os.path.join(OUT_DIR, out_name)
    nib.save(nib.Nifti1Image(acc, ref.affine, ref.header), out_path)
    print("wrote", out_path)

# MTL memorability-significant: 
union_and_save(["PRC", "ERC", "Amy", "HippHead", "PHC" ], "MTL_memorability_sig_mask.nii.gz")
print("done significant mask: with ")
#nonsignifcant mask
union_and_save(["Hippobody", "Hippotail"], "MTL_memorability_nonsig_mask.nii.gz")

