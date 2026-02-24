#!/usr/bin/env python3
"""
Convert BrainVoyager Talairach (.voi) files to NIfTI masks.

For each VOI in each .voi file, a binary NIfTI mask is created
on the grid of a reference Talairach-space NIfTI image.

Supports:
- Legacy text .voi format
- XML-like .voi format
- Optional manual BrainVoyager Talairach conversion
- Optional Talairach brain masking
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine


# =============================================================================
# Coordinate Conversion
# =============================================================================

def mm_to_ijk(coords_mm: np.ndarray, ref_img: nib.Nifti1Image) -> np.ndarray:
    """Convert millimeter (Talairach) coordinates to voxel indices using affine."""
    Ainv = np.linalg.inv(ref_img.affine)
    return apply_affine(Ainv, coords_mm.astype(float))


def mm_to_ijk_bv_manual(coords_mm: np.ndarray, ref_img: nib.Nifti1Image) -> np.ndarray:
    """
    Manual BrainVoyager Talairach conversion.

    Assumes:
    - Talairach (0,0,0) is at volume center.
    - Positive X = right
    - Positive Y = anterior
    - Positive Z = superior
    """
    shape = np.array(ref_img.shape[:3])
    center = shape / 2.0
    return coords_mm.astype(float) + center


def rasterize_mm_coords(
    coords_mm: List[Tuple[int, int, int]],
    ref_img: nib.Nifti1Image,
    use_manual_method: bool = False,
) -> np.ndarray:
    """Rasterize mm coordinates into a binary voxel mask."""
    coords_array = np.asarray(coords_mm, dtype=float)

    if coords_array.size == 0:
        return np.zeros(ref_img.shape[:3], dtype=np.uint8)

    converter = mm_to_ijk_bv_manual if use_manual_method else mm_to_ijk
    ijk = converter(coords_array, ref_img)

    ijk_round = np.rint(ijk).astype(int)
    shape = ref_img.shape[:3]

    in_bounds = (
        (ijk_round[:, 0] >= 0) & (ijk_round[:, 0] < shape[0]) &
        (ijk_round[:, 1] >= 0) & (ijk_round[:, 1] < shape[1]) &
        (ijk_round[:, 2] >= 0) & (ijk_round[:, 2] < shape[2])
    )

    mask = np.zeros(shape, dtype=np.uint8)
    valid_vox = ijk_round[in_bounds]

    if valid_vox.size:
        mask[valid_vox[:, 0], valid_vox[:, 1], valid_vox[:, 2]] = 1

    return mask


# =============================================================================
# VOI Parsing
# =============================================================================

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def is_xml_like(txt: str) -> bool:
    return txt.lstrip().startswith("<")


def parse_voi_legacy_text(txt: str) -> List[Dict]:
    """Parse legacy BrainVoyager text-based VOI format."""
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    vois: List[Dict] = []
    i = 0

    def parse_int_triplet(line: str):
        m = re.match(r"^([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)$", line)
        return tuple(map(int, m.groups())) if m else None

    while i < len(lines):
        if not lines[i].startswith("NameOfVOI"):
            i += 1
            continue

        name = lines[i].split(":", 1)[1].strip()
        i += 1

        color = (255, 0, 0)
        if i < len(lines) and lines[i].startswith("ColorOfVOI"):
            m = re.search(r"(\d+)\s+(\d+)\s+(\d+)", lines[i])
            if m:
                color = tuple(map(int, m.groups()))
            i += 1

        if i < len(lines) and lines[i].startswith("NrOfVoxels"):
            i += 1

        coords: List[Tuple[int, int, int]] = []
        while i < len(lines) and not lines[i].startswith("NameOfVOI"):
            triplet = parse_int_triplet(lines[i])
            if triplet:
                coords.append(triplet)
            i += 1

        vois.append({"name": name, "color": color, "coords": coords})

    if not vois:
        raise ValueError("No VOIs parsed from legacy text.")

    return vois


def parse_voi_xml_like(txt: str) -> List[Dict]:
    """Parse XML-like BrainVoyager VOI format."""
    vois: List[Dict] = []
    blocks = re.findall(r"<VOI>(.*?)</VOI>", txt, re.DOTALL | re.IGNORECASE)

    for block in blocks:
        name_match = re.search(r"<NameOfVOI>\s*(.*?)\s*</NameOfVOI>", block, re.DOTALL)
        voxels = re.findall(
            r"<Voxel>\s*([+-]?\d+)[^\d]+([+-]?\d+)[^\d]+([+-]?\d+)\s*</Voxel>",
            block,
            re.DOTALL,
        )

        name = name_match.group(1).strip() if name_match else "VOI"
        coords = [(int(a), int(b), int(c)) for a, b, c in voxels]

        vois.append({"name": name, "color": (255, 0, 0), "coords": coords})

    if not vois:
        raise ValueError("No VOIs parsed from XML format.")

    return vois


def parse_voi_file(path: Path) -> List[Dict]:
    txt = read_text(path)
    return parse_voi_xml_like(txt) if is_xml_like(txt) else parse_voi_legacy_text(txt)


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BrainVoyager Talairach .voi files to NIfTI masks."
    )
    parser.add_argument("--in_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--ref_nii", required=True, type=Path,
                        help="Reference Talairach NIfTI defining grid + affine")
    parser.add_argument("--tal_brain_mask", type=Path,
                        help="Optional Talairach brain mask (same grid as ref_nii)")
    parser.add_argument("--manual_method", action="store_true",
                        help="Use manual BrainVoyager center-based conversion")

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ref_img = nib.load(str(args.ref_nii))
    affine = ref_img.affine

    brain_mask = None
    if args.tal_brain_mask:
        bm_img = nib.load(str(args.tal_brain_mask))
        if bm_img.shape[:3] != ref_img.shape[:3]:
            raise ValueError("Brain mask shape does not match reference image.")
        brain_mask = (bm_img.get_fdata() > 0)

    voi_files = sorted(args.in_dir.glob("*.voi"))
    if not voi_files:
        raise SystemExit(f"No .voi files found in {args.in_dir}")

    for vf in voi_files:
        try:
            entries = parse_voi_file(vf)
        except Exception as e:
            print(f"[ERROR] {vf.name}: {e}")
            continue

        for idx, voi in enumerate(entries, start=1):
            name = sanitize(voi.get("name", f"VOI{idx}"))
            coords_mm = voi.get("coords", [])

            if not coords_mm:
                continue

            mask = rasterize_mm_coords(
                coords_mm,
                ref_img,
                use_manual_method=args.manual_method,
            )

            if brain_mask is not None:
                mask = mask * brain_mask.astype(np.uint8)

            out_path = args.out_dir / f"{vf.stem}__{name}.nii.gz"
            nib.save(nib.Nifti1Image(mask, affine), out_path)

            print(f"{vf.name} → {out_path.name} ({int(mask.sum())} voxels)")

    print("\nConversion complete.")


if __name__ == "__main__":
    main()
