"""
download_ucf101.py — Resilient UCF101 downloader with auto-resume

The CRCV server frequently drops connections mid-download.
This script retries automatically using HTTP Range requests until
the full file is present.

Usage:
    conda activate vae_env
    python download_ucf101.py
"""

import os
import ssl
import time
import urllib.request

# Bypass SSL verification — safe for downloading this public dataset on Windows
# where conda's certificate store may not include the server's issuer.
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

URL      = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
OUT_PATH = "../UCF101.rar"
EXPECTED_SIZE_GB = 6.5
MAX_RETRIES = 50
RETRY_WAIT  = 10   # seconds between retries


def get_remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as r:
        return int(r.headers.get("Content-Length", 0))


def download_with_resume(url: str, out_path: str):
    existing = os.path.getsize(out_path) if os.path.exists(out_path) else 0

    remote_size = 0
    try:
        remote_size = get_remote_size(url)
        print(f"[INFO] Remote size: {remote_size / 1e9:.2f} GB")
    except Exception as e:
        print(f"[WARN] Could not get remote size: {e}")

    if remote_size and existing >= remote_size:
        print(f"[DONE] File already complete ({existing / 1e9:.2f} GB).")
        return True

    for attempt in range(1, MAX_RETRIES + 1):
        existing = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        print(f"\n[Attempt {attempt}/{MAX_RETRIES}] Resuming from {existing / 1e9:.2f} GB...")

        req = urllib.request.Request(url)
        if existing > 0:
            req.add_header("Range", f"bytes={existing}-")

        try:
            with urllib.request.urlopen(req, timeout=60, context=SSL_CTX) as response:
                mode = "ab" if existing > 0 else "wb"
                with open(out_path, mode) as f:
                    chunk_size = 1024 * 1024  # 1 MB
                    downloaded = existing
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        pct = (downloaded / remote_size * 100) if remote_size else 0
                        print(f"\r  {downloaded / 1e9:.2f} GB / {remote_size / 1e9:.2f} GB  ({pct:.1f}%)", end="", flush=True)

            print()
            final_size = os.path.getsize(out_path)
            if remote_size and final_size >= remote_size:
                print(f"[DONE] Download complete: {final_size / 1e9:.2f} GB")
                return True
            else:
                print(f"[WARN] Got {final_size / 1e9:.2f} GB — retrying...")

        except Exception as e:
            print(f"\n[ERROR] {e} — waiting {RETRY_WAIT}s before retry...")
            time.sleep(RETRY_WAIT)

    print("[FAIL] Max retries reached.")
    return False


if __name__ == "__main__":
    os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)), exist_ok=True)
    success = download_with_resume(URL, os.path.abspath(OUT_PATH))
    if success:
        print("\nNext step: run extract_ucf101.py to unpack the .rar file.")
