import os
import numpy as np
import nibabel as nib

OUT_DIR = "/foundcog/forrestgump/mask/memorability/test/masks/"

def load_mask(name):
    path = os.path.join(OUT_DIR, f"{name}mask_thr20_bilat_cerebrum.nii.gz")
    if not os.path.exists(path):
        path = os.path.join(OUT_DIR, f"{name}mask_thr10_bilat_cerebrum.nii.gz")
    if not os.path.exists(path):
        path = os.path.join(OUT_DIR, f"{name}_mask.nii.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing ROI mask: {path}")
    return nib.load(path)

from nilearn.image import resample_to_img

def union_and_save(roi_names, out_name, ref_img=None):
    imgs = [load_mask(r) for r in roi_names]

    # If no reference image is given, use the first one
    if ref_img is None:
        ref_img = imgs[0]

    # Resample all masks to reference
    imgs_resamp = [resample_to_img(img, ref_img, interpolation='nearest') for img in imgs]

    acc = np.zeros(ref_img.shape, dtype=np.uint8)
    for img in imgs_resamp:
        acc |= (img.get_fdata() > 0).astype(np.uint8)

    out_path = os.path.join(OUT_DIR, out_name)
    nib.save(nib.Nifti1Image(acc, ref_img.affine, ref_img.header), out_path)
    print("wrote", out_path)

# memorability-significant: 
union_and_save(["OFA_", "FFA_", "LOC_", "PRC_", "ERC_", "Amy_", "PHC_", "HippHead_"], "memorability_sig_mask.nii.gz")
#nonsignifcant mask
union_and_save(["EVC_", "OPA_", "PPA_", "RSC_", "Hippobody_", "Hippotail_", "LPFC"], "memorability_nonsig_mask.nii.gz")


