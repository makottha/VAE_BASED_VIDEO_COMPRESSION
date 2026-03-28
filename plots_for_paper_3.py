#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, cv2, torch, numpy as np, pandas as pd
from glob import glob
from torchvision.transforms import ToTensor
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# ============================================================
# CONFIG
# ============================================================
TEST_CLIPS_DIR   = "./dataset_human_readable_64/test/clips"
OUTPUT_CSV       = "rd_eval_metrics.csv"
IMAGE_DIR        = "Archive/paper_images"  # each method's PNG per clip goes here
EXPECTED_FRAMES  = 32
FRAME_SIZE       = (64, 64)                # (W, H)
FRAME_SIZE       = (64, 64)                # (W, H)
UPSCALE          = 3                       # 64->192 for paper visibility; set 1 to keep 64
MID_FRAME_IDX    = EXPECTED_FRAMES // 2    # representative frame to save
MAX_CLIPS        = None                    # e.g., 5 to limit, or None for all
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(IMAGE_DIR, exist_ok=True)

# ============================================================
# MODEL REGISTRY (fill in your classes + ckpts)
# ============================================================
# from your_vqvae_module import VQVAE3D
# from your_vqvae2_module import VQVAE2_3D
from MS_VQ_VAE_2.deeper_vqvae2_entropy_model import MultiScaleVQVAE2  # you already have this

# --- helpers for robust checkpoint loading ---
def _torch_load_safe(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)  # newer PyTorch
    except TypeError:
        return torch.load(path, map_location=DEVICE)  # older PyTorch

def _unwrap_state_dict(blob):
    if isinstance(blob, dict):
        for k in ("state_dict", "model", "net"):
            if k in blob and isinstance(blob[k], dict):
                return blob[k]
    return blob if isinstance(blob, dict) else blob

def _strip_module(sd):
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

# --- VQ-VAE (single level) ---
def load_vqvae(ckpt_path):
    # import your class from the file you just sent
    from VQ_VAE.deeper_vqvae_architecture_64x64 import DeeperVideoVQVAE
    model = DeeperVideoVQVAE().to(DEVICE).eval()

    # load weights robustly
    blob = _torch_load_safe(ckpt_path)
    sd = _strip_module(_unwrap_state_dict(blob))
    model.load_state_dict(sd, strict=False)
    return model

# --- VQ-VAE-2 (two level) ---
def load_vqvae2(ckpt_path):
    # reuse your existing 2-level class for inference; the checkpoint differentiates configs
    from VQ_VAE_2.deeper_vqvae2_architecture_64x64 import MultiScaleVQVAE2
    model = MultiScaleVQVAE2().to(DEVICE).eval()

    blob = _torch_load_safe(ckpt_path)
    sd = _strip_module(_unwrap_state_dict(blob))
    model.load_state_dict(sd, strict=False)
    print("   ↳ VQ-VAE-2 loaded via MultiScaleVQVAE2 class (different checkpoint)")
    return model


def load_msvqvae(ckpt_path):
    model = MultiScaleVQVAE2().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model

MODEL_ZOO = {
    "VQ-VAE":   {"loader": load_vqvae,  "ckpt": "checkpoints/VQ_VAE/deeper_vqvae_epoch050.pt",  "slug": "vq-vae"},
    "VQ-VAE-2": {"loader": load_vqvae2, "ckpt": "checkpoints/VQ_VAE_2/vqvae2_epoch050.pt",     "slug": "vq-vae-2"},
    "MS-VQ-VAE":{"loader": load_msvqvae,"ckpt": "checkpoints/MS_VQVAE/vqvae2_epoch050.pt",      "slug": "ms-vq-vae"},
}

# ============================================================
# I/O HELPERS
# ============================================================
to_tensor = ToTensor()

def log(msg):  # simple unified logger
    print(msg, flush=True)

def load_clip_as_tensor(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < EXPECTED_FRAMES:
        ok, f = cap.read()
        if not ok: break
        f = cv2.resize(f, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames.append(to_tensor(f))  # (3,H,W) in [0,1]
    cap.release()
    if len(frames) != EXPECTED_FRAMES:
        log(f"⚠️  Skipping {os.path.basename(video_path)}: got {len(frames)} frames (expected {EXPECTED_FRAMES})")
        return None
    clip = torch.stack(frames, dim=1).unsqueeze(0).to(DEVICE)  # (1,C,T,H,W)
    return clip

def tensor_to_uint8_frames(x):
    """
    x: (1,C,T,H,W) in [0,1] torch -> np (T,H,W,3) uint8
    """
    x = x.squeeze(0).permute(1, 2, 3, 0).detach().cpu().numpy()  # (T,H,W,3)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)

def forward_reconstruct(model, x):
    """
    Return reconstruction tensor (1,C,T,H,W) in [0,1].
    Works if model returns either 'recon' or tuple (recon, ...).
    """
    with torch.no_grad():
        out = model(x)
    recon = out[0] if isinstance(out, (tuple, list)) else out
    return recon.clamp(0, 1)

def clip_metrics(ref_uint8, dec_uint8):
    ps, ss = [], []
    for r, d in zip(ref_uint8, dec_uint8):
        ps.append(psnr_metric(r, d, data_range=255))
        ss.append(ssim_metric(r, d, channel_axis=2, data_range=255))
    return float(np.mean(ps)), float(np.mean(ss))

def upscale(img, scale=UPSCALE):
    if scale == 1:
        return img
    return cv2.resize(img, (img.shape[1]*scale, img.shape[0]*scale), interpolation=cv2.INTER_NEAREST)

def save_png(img_rgb_uint8, out_path, scale=UPSCALE):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    up = upscale(img_rgb_uint8, scale=scale)
    ok = cv2.imwrite(out_path, cv2.cvtColor(up, cv2.COLOR_RGB2BGR))
    if ok:
        log(f"💾 Saved: {out_path}")
    else:
        log(f"❌ Failed to write image: {out_path}")

def clip_id_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]

# ============================================================
# SANITY / DRY-RUN
# ============================================================
def dry_run_check(name, model):
    """
    Run a synthetic zero clip through the model to verify shapes and ranges.
    """
    dummy = torch.zeros((1, 3, EXPECTED_FRAMES, FRAME_SIZE[1], FRAME_SIZE[0]), device=DEVICE)
    try:
        with torch.no_grad():
            out = model(dummy)
        recon = out[0] if isinstance(out, (tuple, list)) else out
        assert isinstance(recon, torch.Tensor), "forward() did not return a tensor or tuple(list) with tensor first"
        assert recon.shape == dummy.shape, f"unexpected recon shape {tuple(recon.shape)} != {tuple(dummy.shape)}"
        rmin, rmax = float(recon.min()), float(recon.max())
        log(f"   ↳ Dry-run OK for {name}: recon shape {tuple(recon.shape)} in [{rmin:.3f},{rmax:.3f}]")
        return True
    except Exception as e:
        log(f"   ↳ Dry-run FAILED for {name}: {e}")
        return False

# ============================================================
# MAIN
# ============================================================
def main():
    # Load models with strong logging
    models = {}
    log("🔎 Checking checkpoints:")
    for name, cfg in MODEL_ZOO.items():
        ck = cfg["ckpt"]
        exists = os.path.isfile(ck)
        log(f" • {name:10s} ckpt: {ck}  ({'found' if exists else 'MISSING'})")
    log("")

    for name, cfg in MODEL_ZOO.items():
        ckpt = cfg["ckpt"]
        if not os.path.isfile(ckpt):
            log(f"⏭️  Skipping {name}: checkpoint missing at {ckpt}")
            continue
        try:
            log(f"📦 Loading {name} on {DEVICE} …")
            model = cfg["loader"](ckpt)
            log(f"✅ Loaded {name}: {model.__class__.__name__}")
            ok = dry_run_check(name, model)
            if ok:
                models[name] = {"model": model, "slug": cfg["slug"]}
            else:
                log(f"⏭️  Skipping {name}: dry-run failed")
        except NotImplementedError as e:
            log(f"⏭️  Skipping {name}: {e}")
        except Exception as e:
            log(f"❌ Failed to load {name}: {e}")

    if not models:
        raise RuntimeError("No models loaded. Implement loaders or fix checkpoints.")

    video_paths = sorted(glob(os.path.join(TEST_CLIPS_DIR, "*.avi")))
    if MAX_CLIPS is not None:
        video_paths = video_paths[:MAX_CLIPS]
    log(f"\n🧪 Evaluating {len(video_paths)} clips…\n")

    rows = []

    for vp in video_paths:
        cid = clip_id_from_path(vp)
        log(f"▶️  Clip: {cid}")

        clip = load_clip_as_tensor(vp)
        if clip is None:
            continue

        ref_frames = tensor_to_uint8_frames(clip)
        mid_ref = ref_frames[MID_FRAME_IDX]

        # Save ORIGINAL
        save_png(mid_ref, os.path.join(IMAGE_DIR, f"{cid}__original.png"), scale=UPSCALE)

        # Each model
        for name, entry in models.items():
            model = entry["model"]
            slug  = entry["slug"]
            try:
                recon = forward_reconstruct(model, clip)       # (1,C,T,H,W)
                recon_frames = tensor_to_uint8_frames(recon)   # (T,H,W,3)
                p, s = clip_metrics(ref_frames, recon_frames)
                save_png(recon_frames[MID_FRAME_IDX], os.path.join(IMAGE_DIR, f"{cid}__{slug}.png"), scale=UPSCALE)

                rows.append({
                    "clip": cid,
                    "method": name,
                    "psnr": round(p, 2),
                    "ssim": round(s, 4),
                })
                log(f"   {name:10s} → PSNR {p:5.2f} dB | SSIM {s:.4f}")
            except Exception as e:
                log(f"   ⚠️  {name} failed on {cid}: {e}")

    # Save metrics CSV
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_CSV, index=False)
        summary = df.groupby("method").agg(psnr=("psnr","mean"), ssim=("ssim","mean")).reset_index()
        log("\n📊 Mean metrics by method:")
        for _, r in summary.iterrows():
            log(f"  {r['method']:10s}  PSNR {r['psnr']:5.2f} dB   SSIM {r['ssim']:.4f}")
        log(f"\n✅ Saved metrics to {OUTPUT_CSV}")
        log(f"🖼️  Images written to: {os.path.abspath(IMAGE_DIR)}")
    else:
        log("\nNo rows written. Did all models fail to load?")

if __name__ == "__main__":
    main()
