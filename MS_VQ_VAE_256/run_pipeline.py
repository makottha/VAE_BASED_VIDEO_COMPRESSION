"""
run_pipeline.py — Master orchestration script for 256×256 MS-VQ-VAE-3 pipeline

Execution order:
  Step 1: Preprocess UCF-101 → dataset_256/  (skips already-done clips)
  Step 2: Train K=128  (Stage A → Stage B)
  Step 3: Train K=256  (Stage A → Stage B)
  Step 4: Train K=512  (Stage A → Stage B)
  Step 5: Train K=1024 (Stage A → Stage B)
  Step 6: Evaluate all K values → eval_results.csv

Resumable: state tracked in outputs_msvqvae_256/pipeline_status.json
Re-run this script at any time to continue from where it left off.

All logs written to outputs_msvqvae_256/:
  pipeline_status.json        — master status (which steps completed)
  pipeline_master.log         — human-readable timestamped master log
  log_preprocess.txt          — preprocessing stdout
  log_stageA_K{K}.csv         — per-step Stage A training metrics
  log_stageB_K{K}.csv         — per-step Stage B training metrics
  log_eval.txt                — evaluation stdout
  eval_results.csv            — final BPP/PSNR/SSIM/LPIPS table

Usage:
  conda activate vae_env
  cd MS_VQ_VAE_256
  python run_pipeline.py
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

# ─── CONFIG ──────────────────────────────────────────────────────────────────
OUT_DIR     = "./outputs_msvqvae_256"
STATUS_FILE = os.path.join(OUT_DIR, "pipeline_status.json")
MASTER_LOG  = os.path.join(OUT_DIR, "pipeline_master.log")
CONDA_ENV   = "vae_env"
K_VALUES    = [128, 256, 512, 1024]
PYTHON      = sys.executable   # use the current conda env's python
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(MASTER_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except PermissionError:
        pass  # another instance has the file open; stdout is the fallback


def load_status() -> dict:
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE) as f:
            return json.load(f)
    return {
        "preprocess_done": False,
        "train_done": {},       # {"128": False, "256": False, ...}
        "eval_done": False,
        "started_at": datetime.now().isoformat(),
        "last_updated": None,
    }


def save_status(status: dict):
    status["last_updated"] = datetime.now().isoformat()
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)


def run_step(cmd: list, log_file: str, step_name: str) -> bool:
    """Run a subprocess, tee output to log_file, return True if success."""
    log(f"START: {step_name}")
    log(f"  CMD: {' '.join(cmd)}")
    log(f"  LOG: {log_file}")

    mode = "a"   # append so partial logs are preserved on retry
    with open(log_file, mode, encoding="utf-8") as lf:
        lf.write(f"\n{'='*60}\n")
        lf.write(f"[{datetime.now().isoformat()}] {step_name}\n")
        lf.write(f"{'='*60}\n")
        lf.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()

        proc.wait()

    if proc.returncode == 0:
        log(f"DONE:  {step_name}  (exit 0)")
        return True
    else:
        log(f"FAIL:  {step_name}  (exit {proc.returncode}) — check {log_file}")
        return False


def step_preprocess(status: dict):
    if status.get("preprocess_done"):
        log("SKIP: Preprocessing already complete.")
        return

    # Quick sanity check — is UCF-101 there?
    ucf_dir = "../UCF-101"
    if not os.path.isdir(ucf_dir):
        log(f"ERROR: UCF-101 directory not found at {ucf_dir}. Run extract_ucf101.py first.")
        sys.exit(1)

    n_classes = len(os.listdir(ucf_dir))
    log(f"INFO: UCF-101 found — {n_classes} action classes.")

    ok = run_step(
        [PYTHON, "preprocess_256.py"],
        os.path.join(OUT_DIR, "log_preprocess.txt"),
        "Step 1: Preprocess UCF-101 → dataset_256",
    )
    if ok:
        status["preprocess_done"] = True
        save_status(status)
    else:
        log("ERROR: Preprocessing failed. Fix the issue and re-run.")
        sys.exit(1)


def step_train_k(K: int, status: dict):
    key = str(K)
    if status["train_done"].get(key):
        log(f"SKIP: Training K={K} already complete.")
        return

    ok = run_step(
        [PYTHON, "train_single_k_256.py", str(K)],
        os.path.join(OUT_DIR, f"log_train_K{K}.txt"),
        f"Step: Train K={K} (Stage A + B)",
    )
    if ok:
        status["train_done"][key] = True
        save_status(status)
    else:
        log(f"ERROR: Training K={K} failed. Check log_train_K{K}.txt. Re-run to retry from last checkpoint.")
        sys.exit(1)


def step_eval(status: dict):
    if status.get("eval_done"):
        log("SKIP: Evaluation already complete.")
        return

    ok = run_step(
        [PYTHON, "eval_codec_256.py"],
        os.path.join(OUT_DIR, "log_eval.txt"),
        "Step 6: Evaluate all K values",
    )
    if ok:
        status["eval_done"] = True
        save_status(status)
    else:
        log("ERROR: Evaluation failed. Check log_eval.txt.")
        sys.exit(1)


def print_summary(status: dict):
    log("")
    log("--- PIPELINE STATUS ---")
    log(f"  Preprocess:  {'DONE' if status['preprocess_done'] else 'pending'}")
    for K in K_VALUES:
        done = status['train_done'].get(str(K), False)
        log(f"  Train K={K:<4}: {'DONE' if done else 'pending'}")
    log(f"  Eval:        {'DONE' if status['eval_done'] else 'pending'}")
    log("-" * 53)

    results_csv = os.path.join(OUT_DIR, "eval_results.csv")
    if os.path.exists(results_csv):
        log("")
        log("  eval_results.csv:")
        with open(results_csv) as f:
            for line in f:
                log("    " + line.rstrip())


def main():
    log("="*60)
    log("MS-VQ-VAE-3  256×256  Pipeline")
    log("="*60)

    status = load_status()
    print_summary(status)
    log("")

    # Step 1: Preprocess
    step_preprocess(status)

    # Steps 2–5: Train each K
    for K in K_VALUES:
        step_train_k(K, status)

    # Step 6: Evaluate
    step_eval(status)

    log("")
    log("ALL STEPS COMPLETE")
    print_summary(status)


if __name__ == "__main__":
    main()
