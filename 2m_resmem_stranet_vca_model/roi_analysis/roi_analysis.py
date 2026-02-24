#!/usr/bin/env python3
import os, glob, re
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, ttest_rel
from statsmodels.stats.multitest import multipletests
from nibabel.processing import resample_from_to

# ---------------- CONFIG ----------------
OUT_DIR     = "/foundcog/forrestgump/foundcog-adult-2/memorability/resmem_stranet_vca_unscoredvideotest/roi_analysis/outputs/two-sided/"
WARPED_DIR  = "/foundcog/forrestgump/foundcog-adult-2/memorability/resmem_stranet_vca_unscoredvideotest/glm/beta_maps/averaged_maps"
MASK_DIR    = "/foundcog/forrestgump/mask/memorability/test/masks/"

WARPED_GLOB = os.path.join(WARPED_DIR, "*_resmem_averaged_beta_map.nii*")    # beta maps
MASK_GLOB   = os.path.join(MASK_DIR,   "*.nii*")    

ALT         = "two-sided"  
MIN_ROI_VOX = 10           
AFF_TOL     = 1e-5         
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

def load_img(p):
    img = nib.load(p)
    return img, img.get_fdata()

def subject_id_from_filename(p):
    """Edit this if needed. Uses basename without common suffixes."""
    b = os.path.basename(p)
    b = re.sub(r"_MNI2mm\.nii(\.gz)?$", "", b, flags=re.IGNORECASE)
    b = re.sub(r"\.nii(\.gz)?$", "", b, flags=re.IGNORECASE)
    return b

def roi_label_from_path(p):
    """ROI label = mask filename without extension."""
    b = os.path.basename(p)
    return re.sub(r"\.nii(\.gz)?$", "", b, flags=re.IGNORECASE)

def same_grid(img, ref, atol=AFF_TOL):
    return img.shape == ref.shape and np.allclose(img.affine, ref.affine, atol=atol)

# 1) collect files
subs = sorted(glob.glob(WARPED_GLOB))
if not subs:
    raise FileNotFoundError(f"No beta maps found at {WARPED_GLOB}")

mask_paths = sorted(glob.glob(MASK_GLOB))
if not mask_paths:
    raise FileNotFoundError(f"No mask files found at {MASK_GLOB}")

print(f"[INFO] Betas: {len(subs)}  Masks (ROIs): {len(mask_paths)}")

# 2) define reference grid from the first beta
ref_img = nib.load(subs[0])

# 3) load & align masks (each mask = one ROI)
aligned_masks = {}
for mp in mask_paths:
    label = roi_label_from_path(mp)
    mimg = nib.load(mp)
    if not same_grid(mimg, ref_img):
        mimg = resample_from_to(mimg, ref_img, order=0)  # NN for masks
        print(f"[INFO] Resampled ROI to beta grid: {os.path.basename(mp)}")
    mdat = (mimg.get_fdata() > 0)        # <-- boolean
    vox = int(mdat.sum())
    if vox < MIN_ROI_VOX:
        print(f"[WARN] ROI too small after resampling ({vox} vox) -> skip: {os.path.basename(mp)}")
        continue
    aligned_masks[label] = mdat          # boolean mask


if not aligned_masks:
    raise SystemExit("No usable ROIs after resampling/size filtering.")

all_rois = sorted(aligned_masks.keys())
print(f"[INFO] Usable ROIs: {len(all_rois)}")

# 4) extract subject means for each ROI
rows = []
for f in subs:
    sid = subject_id_from_filename(f)
    img = nib.load(f)
    if not same_grid(img, ref_img):
        img = resample_from_to(img, ref_img, order=1)  # linear for betas
    dat = img.get_fdata(dtype=np.float32)              # smaller dtype

    for label, m in aligned_masks.items():
        if not np.any(m):
            rows.append({"subject": sid, "roi": label, "value": np.nan})
            continue
        # EITHER boolean indexing (fine when m is bool)
        # val = float(np.nanmean(dat[m]))
        # OR the robust NaN-safe way:
        val = float(np.nanmean(np.where(m, dat, np.nan)))
        rows.append({"subject": sid, "roi": label, "value": val})


df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No ROI values extracted. Check paths and grids.")

# save per-subject long table
df.to_csv(os.path.join(OUT_DIR, "subject_roi_values_long2.csv"), index=False)

# wide table (subjects x ROIs)
wide = df.pivot(index="subject", columns="roi", values="value")
# keep columns in a stable order
wide = wide.reindex(columns=[r for r in all_rois if r in wide.columns]).sort_index()
wide.to_csv(os.path.join(OUT_DIR, "subject_roi_values2.csv"))

# 5) within-ROI tests vs 0 (one-sample t)
tests = []
for roi in wide.columns:
    vals = wide[roi].dropna().values
    if len(vals) < 2:
        print(f"[WARN] ROI {roi}: n={len(vals)} < 2 -> skip t-test")
        continue
    try:
        t, p = ttest_1samp(vals, 0.0, alternative=ALT)
    except TypeError:
        # older SciPy without 'alternative' param
        t, p = ttest_1samp(vals, 0.0)
        if ALT == "greater":
            # one-sided from two-sided (approx): halve p if t>0 else 1 - (p/2)
            p = p/2 if t > 0 else 1 - p/2
        elif ALT == "less":
            p = p/2 if t < 0 else 1 - p/2
    d = np.mean(vals) / (np.std(vals, ddof=1) + 1e-9)  # Cohen's d
    tests.append({"roi": roi, "n": len(vals), "t": float(t), "p": float(p), "cohen_d": float(d)})

res = pd.DataFrame(tests)
if not res.empty:
    res["p_fwer"] = multipletests(res["p"], method="holm")[1]
    res = res.sort_values("p")
    res.to_csv(os.path.join(OUT_DIR, "subject_withinROI_tests_fwer2.csv"), index=False)
    print("[OK] wrote within-ROI tests -> subject_withinROI_tests_fwer2.csv")
else:
    print("[WARN] no ROI had n>=2; no tests written.")

if comps:
    comp = pd.DataFrame(comps).sort_values("p")
    comp["p_fwer"] = multipletests(comp["p"], method="holm")[1]
    comp.to_csv(os.path.join(OUT_DIR, "subject_paired_tests_fwer2.csv"), index=False)
    print("[OK] wrote paired tests -> subject_paired_tests_fwer2.csv")

# 7) quick plot of all ROIs (scatter + median)
present = [c for c in wide.columns if c in all_rois]
fig = plt.figure(figsize=(max(6, 0.6*len(present)), 4))
X = np.arange(len(present))
for i, col in enumerate(present):
    y = wide[col].dropna().values
    if y.size == 0:
        plt.text(i, 0, "no data", ha="center", va="bottom", rotation=90, fontsize=8)
        continue
    xj = np.random.normal(i, 0.05, size=len(y))
    plt.plot(xj, y, "o", alpha=0.6)
    plt.plot([i-0.2, i+0.2], [np.median(y)]*2, lw=3)
plt.xticks(X, present, rotation=45, ha="right")
plt.ylabel("Beta (subject mean within ROI)")
plt.title("ROI betas across subjects")
plt.tight_layout()
figpath = os.path.join(OUT_DIR, "subject_roi_scatter_medians2.png")
fig.savefig(figpath, dpi=150); plt.close(fig)
print("Wrote:", figpath)
print("Tables in:", OUT_DIR)
