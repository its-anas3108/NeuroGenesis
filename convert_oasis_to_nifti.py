"""
Convert OASIS-1 Analyze (.hdr/.img) MRI files to NIfTI (.nii.gz) format.
Processes all 39 subjects from disc1/ subdirectory.
"""
import sys
from pathlib import Path
import nibabel as nib
import numpy as np

OASIS_DIR = Path("dataset/OASIS")
DISC_DIR = OASIS_DIR / "disc1"

converted = 0
failed = 0

for subj_dir in sorted(DISC_DIR.iterdir()):
    if not subj_dir.is_dir() or not subj_dir.name.startswith("OAS1_"):
        continue

    subj_id = subj_dir.name
    t88_dir = subj_dir / "PROCESSED" / "MPRAGE" / "T88_111"
    if not t88_dir.exists():
        continue

    # Use the masked (skull-stripped) version if available, else unmasked
    masked_hdr = t88_dir / f"{subj_id}_mpr_n4_anon_111_t88_masked_gfc.hdr"
    unmasked_hdr = t88_dir / f"{subj_id}_mpr_n4_anon_111_t88_gfc.hdr"

    src_hdr = masked_hdr if masked_hdr.exists() else unmasked_hdr
    if not src_hdr.exists():
        print(f"  [SKIP] {subj_id} — no .hdr found")
        failed += 1
        continue

    out_nii = OASIS_DIR / f"{subj_id}.nii.gz"
    if out_nii.exists():
        print(f"  [SKIP] {subj_id} — already converted")
        converted += 1
        continue

    try:
        img = nib.load(str(src_hdr))
        data = img.get_fdata(dtype=np.float32)
        nii_img = nib.Nifti1Image(data, affine=img.affine, header=img.header)
        nib.save(nii_img, str(out_nii))
        print(f"  [OK]   {subj_id} → {out_nii.name}  shape={data.shape}")
        converted += 1
    except Exception as e:
        print(f"  [FAIL] {subj_id}: {e}")
        failed += 1

print(f"\nDone: {converted} converted, {failed} failed.")
