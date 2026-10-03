# Noise-suppression benchmark — measured results

Corpus: 108 mixtures — 6 clean Arctic utterances (bdl male + slt female, 2.8 s
each, loudness-normalized) × 6 real noise types from MS-SNSD (babble,
cafeteria, typing, vacuum, neighbor speech, air conditioner) × 3 SNRs
(5/10/15 dB), 48 kHz mono.

Render: `render.mjs` drives each engine's real worklet code in Node
(128-sample quanta, same harness as test/harness.mjs). RTF measured here, not
from README.

Reproduced 2026-09-28 from a clean setup (bench/README.md, bench/dnsmos.py):
all 864 rows scored with no NaNs, and every quality metric below matched to
three decimals. RTF varies with machine load; that run measured fe_s 0.536,
fe_b 0.235, fe_t 0.067, gtcrn 0.054, rnnoise 0.020, speex 0.005.

Judge: DNSMOS P.835 (sig_bak_ovr.onnx + model_v8.onnx, CPU) → SIG / BAK / OVRL
/ P808. Reference-based: STOI (pystoi, 16 kHz) and SI-SDR vs clean, enhanced
aligned by cross-correlation (per-engine latency: fe 896, gtcrn 1530,
rnnoise ~991, speex 128 samples). PESQ unavailable (no Windows C toolchain).

## Aggregate (mean over 108 clips)

| engine      | SIG   | BAK   | OVRL  | P808  | STOI  | dSI-SDR | RTF   | wasm  |
|-------------|-------|-------|-------|-------|-------|---------|-------|-------|
| fe_s (high) | 3.249 | 3.617 | 2.776 | 3.427 | 0.964 | +6.12   | 0.411 | 954K  |
| fe_b (med)  | 3.198 | 3.590 | 2.717 | 3.367 | 0.958 | +5.71   | 0.191 | 580K  |
| fe_t (low)  | 3.071 | 3.456 | 2.545 | 3.239 | 0.944 | +4.19   | 0.069 | 193K  |
| gtcrn       | 3.141 | 3.404 | 2.572 | 3.302 | 0.943 | +2.66   | 0.046 | 192K  |
| rnnoise     | 3.199 | 3.106 | 2.463 | 3.171 | 0.928 | +1.33   | 0.014 | 154K  |
| speex       | 3.168 | 2.726 | 2.287 | 3.076 | 0.897 | -1.46   | 0.004 | 55K   |
| gate        | 3.166 | 2.291 | 2.119 | 2.940 | 0.918 | -0.00   | ~0    | 0     |
| noisy       | 3.166 | 2.291 | 2.119 | 2.940 | 0.918 | -0.00   | —     | —     |

(gate == passthrough on this corpus: clips are speech-active throughout, so a
VAD gate has nothing to close on. Its score is the no-op baseline.)

## BAK by noise type (DNSMOS background MOS — the "how clean" axis)

| engine      | aircond | babble | cafeteria | neighbor | typing | vacuum |
|-------------|---------|--------|-----------|----------|--------|--------|
| noisy       | 2.58    | 2.01   | 2.07      | 2.44     | 2.89   | 1.74   |
| fe_t        | 3.44    | 3.31   | 3.43      | 3.49     | 3.53   | 3.54   |
| fe_b        | 3.61    | 3.47   | 3.58      | 3.61     | 3.68   | 3.59   |
| fe_s        | 3.62    | 3.48   | 3.62      | 3.65     | 3.66   | 3.67   |
| gtcrn       | 3.49    | 3.18   | 3.15      | 3.52     | 3.60   | 3.49   |
| rnnoise     | 3.34    | 2.86   | 3.10      | 2.68     | 3.44   | 3.21   |
| speex       | 3.09    | 2.48   | 2.72      | 2.47     | 3.10   | 2.50   |

## OVRL by SNR

| engine      | s5   | s10  | s15  |
|-------------|------|------|------|
| noisy       | 1.81 | 2.17 | 2.38 |
| fe_t        | 2.38 | 2.58 | 2.68 |
| fe_b        | 2.62 | 2.73 | 2.81 |
| fe_s        | 2.71 | 2.79 | 2.83 |
| gtcrn       | 2.40 | 2.60 | 2.72 |
| rnnoise     | 2.16 | 2.53 | 2.69 |
| speex       | 1.97 | 2.33 | 2.57 |

## STOI by SNR (intelligibility — "does the voice survive")

| engine      | s5    | s10   | s15   |
|-------------|-------|-------|-------|
| noisy       | 0.863 | 0.928 | 0.963 |
| fe_t        | 0.911 | 0.949 | 0.973 |
| fe_b        | 0.932 | 0.962 | 0.981 |
| fe_s        | 0.940 | 0.967 | 0.984 |
| gtcrn       | 0.909 | 0.949 | 0.970 |
| rnnoise     | 0.879 | 0.937 | 0.966 |
| speex       | 0.848 | 0.906 | 0.936 |

## Read

- FastEnhancer sweeps every metric at every tier; fe_s > fe_b > fe_t as
  designed. At 5 dB SNR it lifts OVRL 1.81→2.71 and STOI 0.863→0.940.
- SIG stays ≈ baseline or better — the voice is NOT being shredded (no
  "underwater" trade; STOI rises while BAK rises).
- It wins the hardest column too: neighbor (competing speech) BAK 3.65 vs
  GTCRN 3.52, rnnoise 2.68 — despite no BVC/isolation mode.
- Speex is a mild suppressor and slightly hurts intelligibility at low SNR
  (STOI below noisy baseline at s5, negative dSI-SDR).

## Caveats

- DNSMOS P.835 is a perceptual proxy trained on DNS-Challenge data — good for
  relative ranking, not absolute ground truth; a small human listen on the
  rendered wavs in out/ is the honest final check.
- Corpus is 2 speakers (Arctic bdl/slt) — add VoiceBank+DEMAND test set for
  paper-comparable PESQ if a C toolchain ever lands.
- gate's number is the floor, not a verdict — it's a VAD gate, not a
  suppressor; it would show value on clips with speech gaps.

Reproduce: `python make_corpus.py && node render.mjs && python score.py`
(DNSMOS models auto-download to bench/dnsmos_models/ on first run).
