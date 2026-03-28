# Experiment Context for Paper Improvement
## MS-VQ-VAE Rate–Distortion Extension
### For use as context when reviewing/improving main.tex

---

## 1. Prior Published Work (the paper being extended)

**Title:** Hierarchical Vector-Quantized Latents for Perceptual Low-Resolution Video Compression
**Authors:** Manikanta Kotthapalli, Banafsheh Rekabdar
**Venue:** ISVC 2025, published January 17, 2026
**Springer LNCS vol. 16397**, DOI: 10.1007/978-3-032-14495-9_25
**arXiv:** 2512.24547

**Key results of original paper:**
- PSNR: 25.96 dB, SSIM: 0.8375 on UCF101 test set
- +1.41 dB PSNR and +0.0248 SSIM over single-scale VQ-VAE baseline
- ~18.5M parameter model
- Did NOT include rate–distortion curve or bitrate analysis

---

## 2. Model Architecture (MS-VQ-VAE)

Two-level hierarchical VQ-VAE-2 for video compression.

### Encoder
- **Bottom encoder E_b**: 3D residual convolutions with GroupNorm
  - Spatial stride: 4× (64→16), Temporal stride: 2× (32→16)
  - Output: h_b ∈ R^{16×16×16×128}  (T'×H'×W'×D_b, D_b=128)
- **Top encoder E_t**: further 2× downsampling in each dimension
  - Output: h_t ∈ R^{8×8×8×256}  (T''×H''×W''×D_t, D_t=256)

### Vector Quantization
- Two independent codebooks (top and bottom), each with K entries
- Quantization: z_i = argmin_k || h_i - e_k ||_2
- Same K used for both codebooks in all experiments
- **EMA updates** (not gradient-based): see Section 3

### Decoder
- Transposed 3D convolutions with top-down skip connections
- Bottom features refined in context of coarser top representation
- Reconstructs x_hat from concatenated dequantized top+bottom embeddings

### Autoregressive Entropy Priors (Stage B)
- **TopPrior3D**: masked 3D conv, models p_θ(z_top) autoregressively
  - Raster-scan causal masking over 8×8×8 grid
  - N_t = 512 symbols per clip
- **BottomPriorConditional3D**: masked 3D conv conditioned on upsampled top indices
  - Models p_φ(z_bot | z_top) autoregressively
  - N_b = 4096 symbols per clip
- Expected BPP: sum of -log2(p(z_i | z_{<i})) over all symbols / (T×H×W)
- BPP bounded above by (N_t + N_b) / (T×H×W) × log2(K) = (4608/131072) × log2(K)

---

## 3. Training Protocol

### Why Two Stages?
Stage A (AE) and Stage B (priors) cannot be jointly trained because:
- The argmin quantization is non-differentiable
- No gradient path from rate (entropy) to encoder through discrete bottleneck
- Attempted Lagrangian RD (D + λR): VQ commitment loss dominated rate term
  by factor ~600,000× — rate signal negligible

### Stage A: Autoencoder Training
```
Loss = ||x - x_hat||_1  +  λ_VGG × L_VGG  +  β × (commit_top + commit_bottom)

λ_VGG = 0.1
β = 0.25 (commitment weight)
Optimizer: Adam, lr=2e-4, β1=0.9, β2=0.999
Epochs: 20
Batch size: 8
Subset fraction: 0.5 (50% of training clips per epoch)
```

### Stage B: Prior Training (AE frozen)
```
Loss = CrossEntropy(top_logits, z_top) + CrossEntropy(bot_logits, z_bot)
Same optimizer/lr as Stage A
Epochs: 15
```

### EMA Codebook Updates (critical fix)
Gradient-based VQ caused collapse at K≤512 (loss exploded 11→4444 in epoch 1).
Fixed by replacing gradient updates with EMA:
```
N_k ← γ·N_k + (1-γ)·c_k          [cluster count]
m_k ← γ·m_k + (1-γ)·Σ_{z_i=k} h_i  [embedding accumulator]
e_k = m_k / (N_k + ε)              [codebook entry]

γ = 0.99 (EMA decay)
ε = 1e-5 (Laplace smoothing)
```
Dead-code restart: entries with N_k < 1.0 re-initialised from random encoder output.

### K Sweep: Training Separate Models
Each K ∈ {128, 256, 512, 1024} trained from scratch under identical protocol.
Rationale: K sets hard information ceiling (log2(K) bits/symbol max).
Smaller K → lower max entropy → lower BPP at cost of quality.
Training separate models ensures encoder is fully adapted to its vocabulary size.

**All 4 models trained and evaluated. Checkpoints in outputs_msvqvae_rd/:**
- ae_K128_epoch020.pt, priors_K128_epoch015.pt
- ae_K256_epoch020.pt, priors_K256_epoch015.pt
- ae_K512_epoch020.pt, priors_K512_epoch015.pt
- ae_K1024_epoch020.pt, priors_K1024_epoch015.pt

---

## 4. Dataset

**UCF101** — action recognition dataset repurposed for video compression.
- Pre-processed to 64×64 spatial resolution, 32 frames @ 16 FPS
- Clips stored as pre-decoded float32 tensors (no codec artifacts in training)
- Test set: 500 held-out clips (same 500 used for ALL evaluations)
- Training: remainder of dataset, 50% sampled per epoch

---

## 5. Evaluation Setup

### Our Model (eval_codec.py)
- Loaded AE checkpoint + Prior checkpoint for each K
- Forward pass: encode → quantize → prior forward pass → compute expected bits
- BPP from Eq: sum(-log2(p(z_i|z_{<i}))) / (T×H×W) over both codebooks
- PSNR, SSIM: skimage metrics, per frame, averaged over clip
- LPIPS: VGG backbone (lpips library), frames as (T,3,H,W) in [-1,1], averaged

### H.264 Baseline (eval_h264.py)
- ffmpeg with libx264, -preset medium -tune psnr
- CRFs tested: 28, 32, 36
- BPP from actual encoded file size: file_size_bytes × 8 / (W×H×T)
- Same PSNR, SSIM, LPIPS implementations as model evaluation
- Same 500 test clips

---

## 6. Results

### MS-VQ-VAE K Sweep — 500 UCF101 test clips

| K    | Stage A val loss | Stage B val BPP | BPP    | PSNR (dB) | SSIM   | LPIPS↓ |
|------|-----------------|-----------------|--------|-----------|--------|--------|
| 128  | 0.335           | 0.060           | 0.0427 | 25.88     | 0.8780 | 0.1546 |
| 256  | 0.335           | 0.060           | 0.0487 | 26.28     | 0.8855 | 0.1386 |
| 512  | 0.335           | 0.060           | 0.0550 | 26.27     | 0.8966 | 0.1287 |
| 1024 | 0.300           | 0.070           | 0.0635 | 27.14     | 0.9043 | 0.1140 |

**Notable anomaly:** PSNR flat between K=256 and K=512 (26.28 vs 26.27) despite
SSIM and LPIPS both improving. Interpreted as L1 loss saturating while VGG loss
continues to improve higher-frequency perceptual detail.

**K=1024 note:** Stage B val BPP was still dropping at epoch 15 (0.089→0.070).
25–30 epochs would likely push BPP to ~0.055–0.060 and PSNR above 27.5 dB.

### H.264 Baseline — 500 UCF101 test clips (measured, not estimated)

| CRF | BPP    | PSNR (dB) | SSIM   | LPIPS↓ |
|-----|--------|-----------|--------|--------|
| 36  | 0.2105 | 29.52     | 0.9051 | 0.1275 |
| 32  | 0.2622 | 31.87     | 0.9391 | 0.0830 |
| 28  | 0.3525 | 34.47     | 0.9624 | 0.0494 |

### Key Crossover Points

| Finding | Detail |
|---------|--------|
| LPIPS crossover | K=512 (LPIPS=0.1287, BPP=0.055) ≈ H.264 CRF36 (LPIPS=0.1275, BPP=0.211) → same perceptual quality at **3.8× lower bitrate** |
| LPIPS superiority | K=1024 (LPIPS=0.114) **beats** H.264 CRF36 (LPIPS=0.128) at 3.3× lower BPP |
| SSIM crossover | K=1024 (SSIM=0.9043) ≈ H.264 CRF36 (SSIM=0.9051) at 3.3× lower BPP |
| PSNR gap | 2–8 dB below H.264 across all K — expected for perceptual codec, model optimises L1+VGG not PSNR |
| Bitrate regime | Our model: 0.043–0.064 bpp. H.264 minimum at this resolution: 0.211 bpp (3.3–5× gap) |

---

## 7. Codebook Analysis Results

### Utilisation (% active codebook entries per clip)

| K    | Top codebook | Bottom codebook |
|------|-------------|-----------------|
| 128  | ~95%+       | ~95%+           |
| 256  | ~90%+       | ~90%+           |
| 512  | ~75%        | ~90%+           |
| 1024 | ~57%        | ~90%+           |

Top codebook lower at K=1024 because top grid has only 8×8×8=512 symbols per clip,
so at most 512/1024=50% of entries can activate per clip (average ~57% over dataset).

### Entropy Efficiency (H(z) / log2(K))

Bottom codebooks: **70–85% efficiency** across all K.
Consistent across K values — encoder learns similar statistical structure
regardless of vocabulary size.
15–30% below ceiling = compression the prior achieves beyond naive log2(K) bits/symbol.

### Rate Decomposition (top vs. bottom bits)
Bottom codebook contributes **70–80% of total expected bits** at all K.
Explained by grid size ratio: N_b/N_t = (16×16×16)/(8×8×8) = 8×.

### Index Frequency Distribution
Power-law (Zipfian) distribution in bottom codebook at all K values.
Near-linear on log-log plot: small fraction of codes dominate usage.
This heavy-tailed structure is what makes the autoregressive prior effective —
it assigns short codewords to frequent symbols, compressing well below log2(K).

### Embedding PCA (bottom codebook, 2D projection)
- K=128: compact, roughly isotropic cluster
- K=256/512: expanding, directional structure emerging
- K=1024: distinct sub-clusters visible, wider spread
Variance explained by PC1+PC2 decreases with K (higher K fills more dimensions).

---

## 8. What Was Tried and Failed

### Stage C Joint Fine-tuning
Attempted: freeze nothing, train AE+priors jointly with soft rate term.
Approach: temperature-annealed softmax BPP as differentiable rate surrogate.
Result: VQ commitment loss dominated rate term by **~600,000×**.
The soft BPP collapsed to negligible values; rate gradient was invisible to encoder.
Conclusion: Lagrangian RD is not feasible for discrete VQ bottleneck without
fundamental architectural changes.

### Gradient-based VQ at small K
At K=128, 256, 512: gradient commitment loss caused codebook collapse within
epoch 1. VQ loss went from ~11 to >4444. Fixed by EMA updates.

---

## 9. Paper Files

**Location:** MS_VQ_VAE_RD/paper/
- main.tex (847 lines — NeurIPS style, numbered citations [1][2]...)
- references.bib (18 citations)
- neurips_2024.sty
- figures/
  - fig4_rd_combined.png — 3-panel RD curve (PSNR, SSIM, LPIPS vs BPP)
  - fig1_utilization.png — codebook utilisation bar chart
  - fig2_entropy_efficiency.png — entropy efficiency per K
  - fig3_rate_breakdown.png — top vs bottom bits stacked bar
  - fig4_index_histogram.png — power-law index frequency (log scale)
  - fig5_embedding_pca.png — PCA of bottom codebook embeddings

**To compile:** pdflatex main.tex → bibtex main → pdflatex ×2
(No LaTeX installed on local machine; use Overleaf)

---

## 10. Output Files (all in outputs_msvqvae_rd/)

| File | Contents |
|------|----------|
| metrics_K128/256/512/1024.csv | Per-clip BPP, PSNR, SSIM, LPIPS, codebook stats |
| rd_curve_summary.csv | One row per K, mean over 500 clips |
| h264_baseline/h264_per_clip.csv | Per-clip H.264 metrics |
| h264_baseline/h264_summary.csv | Mean per CRF |
| codebook_analysis/codebook_summary.csv | Utilisation, entropy efficiency per K |
| rd_curves/fig{1-4}*.png | Individual + combined RD curve figures |
| codebook_analysis/fig{1-5}*.png | Codebook analysis figures |
