"""
check_progress.py — Training progress monitor
Run anytime to see current status:
    conda run -n vae_env python check_progress.py
"""
import csv, os, glob
from pathlib import Path

OUT = Path(__file__).parent / "outputs"

# ── CSV log ────────────────────────────────────────────────────────────────────
log_csv = OUT / "train_log.csv"
if log_csv.exists():
    with open(log_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if rows:
        print(f"\n{'='*60}")
        print(f"TRAINING LOG  ({len(rows)} epochs completed)")
        print(f"{'='*60}")
        current_prefix = None
        for r in rows:
            if r["prefix"] != current_prefix:
                current_prefix = r["prefix"]
                print(f"\n  K = {r['prefix'].rstrip('_').replace('K','')}")
                print(f"  {'Stage':<8} {'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14} {'Val BPP':<12} {'Time(s)':<8}")
                print(f"  {'-'*64}")
            loss_str = f"{float(r['train_loss']):.4f}" if r.get('train_loss') else "—"
            val_str  = f"{float(r['val_loss']):.4f}"   if r.get('val_loss')   else "—"
            bpp_str  = f"{float(r['val_bpp']):.4f}"    if r.get('val_bpp')    else "—"
            t_str    = f"{float(r['elapsed']):.0f}"    if r.get('elapsed')    else "—"
            print(f"  {r['stage']:<8} {r['epoch']:<8} {loss_str:<14} {val_str:<14} {bpp_str:<12} {t_str:<8}")
    else:
        print("\n  [LOG] No epochs completed yet (training still in first epoch).")
else:
    print("\n  [LOG] train_log.csv not found.")

# ── checkpoints ────────────────────────────────────────────────────────────────
ckpts = sorted(glob.glob(str(OUT / "*.pt")))
print(f"\n{'='*60}")
print(f"CHECKPOINTS  ({len(ckpts)} files)")
print(f"{'='*60}")
if ckpts:
    for c in ckpts:
        size_mb = os.path.getsize(c) / 1e6
        print(f"  {Path(c).name:<45} {size_mb:.1f} MB")
else:
    print("  None yet.")

# ── summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("WHAT TO DO NEXT")
print(f"{'='*60}")
if not ckpts:
    print("  Training is still running (first epoch in progress).")
    print("  Come back in ~1-2 hours for first results.")
else:
    ae_ks  = set(Path(c).name.split("_")[1] for c in ckpts if "ae_K" in c)
    pr_ks  = set(Path(c).name.split("_")[1] for c in ckpts if "priors_K" in c)
    print(f"  AE checkpoints for K values:     {sorted(ae_ks)}")
    print(f"  Prior checkpoints for K values:  {sorted(pr_ks)}")
    if len(ae_ks) >= 4 and len(pr_ks) >= 4:
        print("\n  Training COMPLETE. Run eval:")
        print("    conda run -n vae_env python eval_motion_codec.py")
    else:
        print(f"\n  Training in progress. Completed: {sorted(ae_ks)}")
