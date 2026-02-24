#!/usr/bin/env python3
import os, glob, re
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from nibabel.processing import resample_from_to

# ---------------- CONFIG ----------------
OUT_DIR     = "/foundcog/forrestgump/foundcog-adult-2/memorability/resmem_stranet_vca_unscoredvideotest/roi_analysis/outputs/sig_vs_nonsig/mtl/"
WARPED_DIR  = "/foundcog/forrestgump/foundcog-adult-2/memorability/resmem_stranet_vca_unscoredvideotest/glm/beta_maps/averaged_maps"
MASK_DIR    = "/foundcog/forrestgump/mask/memorability/mtl_sig_vs_non/"

WARPED_GLOB = os.path.join(WARPED_DIR, "*_resmem_averaged_beta_map.nii*")
MASK_GLOB   = os.path.join(MASK_DIR,   "*bin.nii*")

ALT         = "two-sided"  
MIN_ROI_VOX = 10
AFF_TOL     = 1e-5
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

def subject_id_from_filename(p):
    b = os.path.basename(p)
    b = re.sub(r"_MNI2mm\.nii(\.gz)?$", "", b, flags=re.IGNORECASE)
    b = re.sub(r"\.nii(\.gz)?$", "", b, flags=re.IGNORECASE)
    return b

def same_grid(img, ref, atol=AFF_TOL):
    return img.shape == ref.shape and np.allclose(img.affine, ref.affine, atol=atol)

# Collect data
subs = sorted(glob.glob(WARPED_GLOB))
mask_paths = sorted(glob.glob(MASK_GLOB))
if not subs:
    raise FileNotFoundError(f"No beta maps found: {WARPED_GLOB}")
if len(mask_paths) != 2:
    raise FileNotFoundError(f"Expected exactly 2 masks in {MASK_DIR}, found {len(mask_paths)}")

print(f"[INFO] Betas: {len(subs)}  Masks: {mask_paths}")

# Define reference
ref_img = nib.load(subs[0])

# Load & align masks
aligned_masks = {}
for p in mask_paths:
    name = os.path.basename(p).lower()
    key = "sig" if "sig" in name and "nonsig" not in name else "nonsig"

    img = nib.load(p)
    if not same_grid(img, ref_img):
        img = resample_from_to(img, ref_img, order=0)
        print(f"[INFO] Resampled mask: {os.path.basename(p)}")

    mask = img.get_fdata() > 0
    vox = int(mask.sum())
    if vox < MIN_ROI_VOX:
        raise SystemExit(f"{key} mask too small ({vox} vox) -> abort.")

    aligned_masks[key] = mask
    print(f"[INFO] {key.upper()} vox: {vox}")

# Extract subject means
rows = []
for f in subs:
    sid = subject_id_from_filename(f)
    img = nib.load(f)
    if not same_grid(img, ref_img):
        img = resample_from_to(img, ref_img, order=1)
    dat = img.get_fdata(dtype=np.float32)

    for label, m in aligned_masks.items():
        val = float(np.nanmean(np.where(m, dat, np.nan)))
        rows.append({"subject": sid, "mask": label, "value": val})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "subject_sig_nonsig_values_long2.csv"), index=False)
wide = df.pivot(index="subject", columns="mask", values="value").sort_index()
wide.to_csv(os.path.join(OUT_DIR, "subject_sig_nonsig_values2.csv"))

# Paired t-test
pair = wide[['sig', 'nonsig']].dropna()
if len(pair) < 2:
    raise SystemExit("Not enough paired subjects.")

t, p = ttest_rel(pair['sig'], pair['nonsig'], alternative=ALT) \
    if "alternative" in ttest_rel.__code__.co_varnames \
    else ttest_rel(pair['sig'], pair['nonsig'])

diff = pair['sig'] - pair['nonsig']
dz = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-9))

res = pd.DataFrame([{
    "n": len(pair),
    "t": float(t),
    "p": float(p),
    "alternative": ALT,
    "cohen_dz": dz,
    "mean_sig": float(pair['sig'].mean()),
    "mean_nonsig": float(pair['nonsig'].mean()),
    "mean_diff(sig_minus_nonsig)": float(diff.mean())
}])
res["p_fwer"] = multipletests(res["p"], method="holm")[1]
res.to_csv(os.path.join(OUT_DIR, "sig_vs_nonsig_paired_test2.csv"), index=False)

# Plot
fig = plt.figure(figsize=(6,4))
for i, col in enumerate(['sig','nonsig']):
    y = pair[col].values
    xj = np.random.normal(i, 0.05, size=len(y))
    plt.plot(xj, y, "o", alpha=0.6)
    plt.plot([i-0.2,i+0.2], [np.median(y)]*2, lw=3)
plt.xticks([0,1], ['sig','nonsig'])
plt.ylabel("Beta (subject mean)")
plt.title("SIG vs NONSIG betas across subjects")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"sig_nonsig_scatter2.png"), dpi=150)
plt.close()
print("[OK] Analysis complete. Results in:", OUT_DIR)
