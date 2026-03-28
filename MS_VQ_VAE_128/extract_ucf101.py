"""
extract_ucf101.py — Extract UCF101.rar into UCF-101/

Usage:
    conda activate vae_env
    python extract_ucf101.py
"""

import os
import rarfile

RAR_PATH   = "../UCF101.rar"
EXTRACT_TO = "../"   # extracts to ../UCF-101/

rar_abs = os.path.abspath(RAR_PATH)
out_abs = os.path.abspath(EXTRACT_TO)

print(f"[INFO] Extracting {rar_abs}")
print(f"[INFO] Destination: {out_abs}")

with rarfile.RarFile(rar_abs) as rf:
    members = rf.infolist()
    print(f"[INFO] {len(members)} files in archive")
    for i, m in enumerate(members, 1):
        rf.extract(m, out_abs)
        if i % 500 == 0 or i == len(members):
            print(f"  {i}/{len(members)} extracted...")

print("[DONE] Extraction complete.")
print(f"       Videos at: {os.path.join(out_abs, 'UCF-101')}")
print("Next step: run preprocess_128.py")
