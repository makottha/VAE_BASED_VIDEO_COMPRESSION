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
PANEL_DIR        = "paper_panels"          # where panel .pngs go
EXPECTED_FRAMES  = 32
FRAME_SIZE       = (64, 64)                # (W, H)
FPS              = 16                      # only used if you re-encode (not needed here)
UPSCALE          = 3                       # 64->192 for paper visibility
MID_FRAME_IDX    = EXPECTED_FRAMES // 2    # representative frame per panel
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(PANEL_DIR, exist_ok=True)

# ============================================================
# MODEL REGISTRY (fill in your classes + ckpts)
# ============================================================
# TODO: replace imports with your actual modules/classes
# from your_vqvae_module import VQVAE3D
# from your_vqvae2_module import VQVAE2_3D
from MS_VQ_VAE_2.deeper_vqvae2_entropy_model import MultiScaleVQVAE2  # <-- you already have this

def load_vqvae(ckpt_path):
    """
    TODO: Replace with your VQ-VAE (single level, 3D) constructor.
    Must return a model that maps (B,C,T,H,W)->(B,C,T,H,W) in [0,1].
    """
    raise NotImplementedError("Implement load_vqvae() with your class & checkpoint")

def load_vqvae2(ckpt_path):
    """
    TODO: Replace with your VQ-VAE-2 (two-level, 3D) constructor.
    """
    raise NotImplementedError("Implement load_vqvae2() with your class & checkpoint")

def load_msvqvae(ckpt_path):
    model = MultiScaleVQVAE2().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model

# Fill these paths
MODEL_ZOO = {
    "VQ-VAE":   {"loader": load_vqvae,  "ckpt": "checkpoints/VQVAE/deeper_vqvae_epoch050.pt"},
    "VQ-VAE-2": {"loader": load_vqvae2, "ckpt": "checkpoints/VQ_VAE_2/vqvae2_epoch050.pt"},
    "MS-VQ-VAE":{"loader": load_msvqvae,"ckpt": "checkpoints/MS_VQVAE/vqvae2_epoch050.pt"},
}

# ============================================================
# I/O HELPERS
# ============================================================
to_tensor = ToTensor()

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
        print(f"⚠️  Skipping {os.path.basename(video_path)}: got {len(frames)} frames")
        return None
    # Stack to (1,C,T,H,W)
    clip = torch.stack(frames, dim=1).unsqueeze(0).to(DEVICE)
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
    if isinstance(out, (tuple, list)):
        recon = out[0]
    else:
        recon = out
    return recon.clamp(0, 1)

def clip_metrics(ref_uint8, dec_uint8):
    """
    ref_uint8/dec_uint8: (T,H,W,3) uint8
    returns (PSNR, SSIM) averaged over T frames.
    """
    ps, ss = [], []
    for r, d in zip(ref_uint8, dec_uint8):
        ps.append(psnr_metric(r, d, data_range=255))
        ss.append(ssim_metric(r, d, channel_axis=2, data_range=255))
    return float(np.mean(ps)), float(np.mean(ss))

def add_label_strip(img, text):
    """
    img: (H,W,3) uint8
    Returns image with a small black label strip on top and white text.
    """
    h, w = img.shape[:2]
    strip_h = max(14, h // 12)
    canvas = img.copy()
    cv2.rectangle(canvas, (0, 0), (w, strip_h), (0, 0, 0), thickness=-1)
    cv2.putText(canvas, text, (5, int(strip_h*0.75)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    return canvas

def upscale(img, scale=UPSCALE):
    if scale == 1: return img
    return cv2.resize(img, (img.shape[1]*scale, img.shape[0]*scale), interpolation=cv2.INTER_NEAREST)

def make_panel_frame(frames_dict, scale=UPSCALE):
    """
    frames_dict: Ordered dict-like mapping label -> (H,W,3) uint8
    Returns a single-row panel image.
    """
    tiles = []
    for label, im in frames_dict.items():
        im_l = add_label_strip(upscale(im, scale), label)
        tiles.append(im_l)
    return np.concatenate(tiles, axis=1)

# ============================================================
# MAIN
# ============================================================
def main():
    # Load all models
    models = {}
    for name, cfg in MODEL_ZOO.items():
        try:
            print(f"📦 Loading {name} from {cfg['ckpt']}")
            models[name] = cfg["loader"](cfg["ckpt"])
        except NotImplementedError as e:
            print(f"⏭️  Skipping {name}: {e}")
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")

    if not models:
        raise RuntimeError("No models loaded. Please implement the loaders / fix ckpt paths.")

    video_paths = sorted(glob(os.path.join(TEST_CLIPS_DIR, "*.avi")))
    print(f"🧪 Evaluating {len(video_paths)} clips...")
    rows = []

    counter = 0
    for vp in video_paths:
        if counter == 1:
            break
        counter +=1

        clip = load_clip_as_tensor(vp)
        if clip is None:
            continue

        # Reference frames (uint8)
        ref_frames = tensor_to_uint8_frames(clip)

        # Reconstruct with each model -> metrics + middle frame for panel
        middle_ref = ref_frames[MID_FRAME_IDX]  # (H,W,3)
        panel_dict = {"Original": middle_ref}

        for name, model in models.items():
            try:
                recon = forward_reconstruct(model, clip)                # (1,C,T,H,W)
                recon_frames = tensor_to_uint8_frames(recon)            # (T,H,W,3)
                p, s = clip_metrics(ref_frames, recon_frames)
                panel_dict[name] = recon_frames[MID_FRAME_IDX]

                rows.append({
                    "clip": os.path.basename(vp),
                    "method": name,
                    "psnr": round(p, 2),
                    "ssim": round(s, 4),
                })
                print(f"{os.path.basename(vp)} | {name:10s} | PSNR {p:5.2f} dB | SSIM {s:.4f}")
            except Exception as e:
                print(f"⚠️  {name} failed on {os.path.basename(vp)}: {e}")

        # Save panel image for this clip
        panel = make_panel_frame(panel_dict, scale=UPSCALE)
        out_png = os.path.join(PANEL_DIR, os.path.splitext(os.path.basename(vp))[0] + "_panel.png")
        cv2.imwrite(out_png, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    # Save metrics CSV (long format: one row per clip × method)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_CSV, index=False)
        # Summaries
        summary = df.groupby("method").agg(psnr=("psnr","mean"), ssim=("ssim","mean")).reset_index()
        print("\n📊 Mean metrics by method:")
        for _, r in summary.iterrows():
            print(f"  {r['method']:10s}  PSNR {r['psnr']:5.2f} dB   SSIM {r['ssim']:.4f}")
        print(f"\n✅ Saved metrics to {OUTPUT_CSV}")
        print(f"🖼️  Panels written to: {os.path.abspath(PANEL_DIR)}")
    else:
        print("\nNo rows written. Did all models fail to load?")

if __name__ == "__main__":
    main()
