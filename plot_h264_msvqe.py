#!/usr/bin/env python3
import argparse, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_csv(path):
    df = pd.read_csv(path)
    # coerce numerics
    for col in ("bpp","psnr","ssim"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["bpp"])

def summarize_h264(df):
    d = df[df["method"].str.upper() == "H264"].copy()
    if d.empty:
        return pd.DataFrame(columns=["crf","bpp","psnr","ssim"])
    d["crf"] = pd.to_numeric(d["crf"], errors="coerce")
    g = (d.dropna(subset=["crf"])
          .groupby("crf")
          .agg(bpp=("bpp","mean"), psnr=("psnr","mean"), ssim=("ssim","mean"))
          .reset_index()
          .sort_values("bpp"))
    return g

def summarize_msvqvae(df):
    d = df[df["method"].str.upper().isin(["MSVQVAE","MS-VQ-VAE","MS_VQVAE"])].copy()
    if d.empty:
        # handle older column naming just in case
        d = df[df["method"].str.contains("VQ", case=False, na=False)].copy()
    if d.empty:
        return pd.DataFrame(columns=["crf","bpp","psnr","ssim"])

    # In your writer, 'crf' holds "proxy"/"actual"
    d["crf"] = d["crf"].astype(str).str.lower()
    g = (d.groupby("crf")
          .agg(bpp=("bpp","mean"), psnr=("psnr","mean"), ssim=("ssim","mean"))
          .reset_index()
          .sort_values("bpp"))
    return g

def plot_rd(h264, msvq, outdir, logx=False):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------- PSNR ----------
    plt.figure()
    if not h264.empty:
        plt.plot(h264["bpp"], h264["psnr"], marker="o", label="H.264 (CRF mean)")
        # annotate CRFs
        for _, r in h264.iterrows():
            plt.annotate(int(r["crf"]), (r["bpp"], r["psnr"]), textcoords="offset points", xytext=(5,5), fontsize=8)
    if not msvq.empty:
        proxy = msvq[msvq["crf"]=="proxy"]
        actual = msvq[msvq["crf"]=="actual"]
        if not proxy.empty:
            plt.plot(proxy["bpp"], proxy["psnr"], marker="^", label="MS-VQ-VAE (proxy)")
        if not actual.empty:
            plt.plot(actual["bpp"], actual["psnr"], marker="s", label="MS-VQ-VAE (actual)")

    plt.xlabel("Bits per pixel (bpp)")
    plt.ylabel("PSNR (dB)")
    if logx: plt.xscale("log")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    psnr_png = outdir / "rd_psnr.png"
    plt.savefig(psnr_png, dpi=300)
    plt.close()

    # ---------- SSIM ----------
    plt.figure()
    if not h264.empty:
        plt.plot(h264["bpp"], h264["ssim"], marker="o", label="H.264 (CRF mean)")
        for _, r in h264.iterrows():
            plt.annotate(int(r["crf"]), (r["bpp"], r["ssim"]), textcoords="offset points", xytext=(5,5), fontsize=8)
    if not msvq.empty:
        proxy = msvq[msvq["crf"]=="proxy"]
        actual = msvq[msvq["crf"]=="actual"]
        if not proxy.empty:
            plt.plot(proxy["bpp"], proxy["ssim"], marker="^", label="MS-VQ-VAE (proxy)")
        if not actual.empty:
            plt.plot(actual["bpp"], actual["ssim"], marker="s", label="MS-VQ-VAE (actual)")

    plt.xlabel("Bits per pixel (bpp)")
    plt.ylabel("SSIM")
    if logx: plt.xscale("log")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    ssim_png = outdir / "rd_ssim.png"
    plt.savefig(ssim_png, dpi=300)
    plt.close()

    # Also save a tiny combined CSV of plotted points
    h264["method"] = "H264"
    msvq_plot = msvq.copy()
    msvq_plot["method"] = "MSVQVAE"
    combined = pd.concat([
        h264[["method","crf","bpp","psnr","ssim"]],
        msvq_plot[["method","crf","bpp","psnr","ssim"]],
    ], ignore_index=True)
    combined.to_csv(outdir / "rd_points_plotted.csv", index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h264_csv", default="./dataset_human_readable_64/test/h264_clips/rd_h264.csv")
    ap.add_argument("--ms_csv",   default="./dataset_human_readable_64/test/msvqvae_out/rd_msvqvae.csv")
    ap.add_argument("--outdir",   default="./plots_rd")
    ap.add_argument("--logx",     action="store_true", help="Log-scale x-axis")
    args = ap.parse_args()

    df_h = load_csv(args.h264_csv)
    df_m = load_csv(args.ms_csv)

    h264 = summarize_h264(df_h)
    msvq = summarize_msvqvae(df_m)

    if h264.empty and msvq.empty:
        raise SystemExit("No data to plot. Check CSV paths.")

    plot_rd(h264, msvq, args.outdir, logx=args.logx)
    print(f"Saved plots to: {pathlib.Path(args.outdir).resolve()}")

if __name__ == "__main__":
    main()
