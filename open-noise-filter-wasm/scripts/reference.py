"""FastEnhancer offline reference — the equivalence anchor for the worklet.

Pipeline taken verbatim from the repo's scripts/test_onnx_spec.py:
  raw complex STFT (periodic hann, normalized=False, one-hop right pad)
  -> spec_in [1, 513, 1, 2] per frame + zero caches
  -> spec_out raw enhanced complex spectrum (power compression is INSIDE the graph)
  -> iSTFT, clamp [-1, 1]

Usage: python scripts/reference.py [t|b|s]   (any python with numpy+onnxruntime)
Writes reference_x.npy / reference_y.npy for the worklet diff test, plus
dbg_spec_in.npy / dbg_spec_out_<tier>.npy (first STFT frame and its enhanced
output, for specdiff.mjs's single-frame WASM check).
"""
import sys
import numpy as np
import onnxruntime as ort

TIER = (sys.argv[1] if len(sys.argv) > 1 else "b").lower()
import os
# a path (ending .onnx or containing a separator) selects a custom model;
# the tier name is then derived from the filename for dbg_spec_out_*.npy
if TIER.endswith(".onnx") or os.sep in TIER or "/" in TIER:
    MODEL_PATH, TIER = TIER, os.path.splitext(os.path.basename(TIER))[0].replace("_spec", "")
else:
    MODEL_PATH = f"models/fastenhancer_{TIER}_spec.onnx"
SR, N_FFT, HOP = 48000, 1024, 512
SEED = 7

# ---- signal: AM harmonic "voice" + 120 Hz hum + noise ----------------------
rng = np.random.default_rng(SEED)
t = np.arange(SR * 3) / SR
voice = sum((0.6 / k) * np.sin(2 * np.pi * 140 * k * t + rng.uniform(0, 6)) for k in range(1, 9))
voice *= 0.35 * (1 + np.sin(2 * np.pi * 3.0 * t)) ** 2
noisy = np.clip(voice + 0.08 * np.sin(2 * np.pi * 120 * t) + 0.10 * rng.standard_normal(len(t)), -1, 1)
clean = voice

# ---- STFT, torch-equivalent (periodic hann, right pad one hop) --------------
win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)   # periodic hann
x = np.concatenate([noisy, np.zeros(HOP)])
n_frames = 1 + (len(x) - N_FFT) // HOP
idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
X = np.fft.rfft(x[idx] * win, axis=1)                              # [T, 513]
spec = np.stack([X.real, X.imag], -1)                               # [T, 513, 2]

# ---- streaming inference ------------------------------------------------------
so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
sess = ort.InferenceSession(MODEL_PATH, so, providers=["CPUExecutionProvider"])
inputs = sess.get_inputs()
names = [i.name for i in inputs]
caches = {i.name: np.zeros(i.shape, np.float32) for i in inputs if i.name.startswith("cache_in_")}
out_frames = np.empty_like(spec)
import time
tic = time.perf_counter()
for f in range(n_frames):
    feeds = {names[0]: spec[f][None, :, None, :].astype(np.float32), **caches}
    res = sess.run(None, feeds)
    out_frames[f] = res[0][0, :, 0, :]
    for j in range(len(res) - 1):
        caches[f"cache_in_{j}"] = res[j + 1]
dt = time.perf_counter() - tic
print(f"tier={TIER}  frames={n_frames}  RTF={dt * SR / n_frames / HOP:.4f} (1 thread)")

# ---- iSTFT (torch-equivalent overlap-add) ------------------------------------
Y = out_frames[..., 0] + 1j * out_frames[..., 1]
enh = np.zeros(len(x))
wsum = np.zeros(len(x))
for f in range(n_frames):
    enh[f * HOP : f * HOP + N_FFT] += np.fft.irfft(Y[f], N_FFT) * win
    wsum[f * HOP : f * HOP + N_FFT] += win**2
enh = np.clip((enh / np.maximum(wsum, 1e-8))[: len(noisy)], -1, 1)

def sisdr(a, b):
    a, b = a - a.mean(), b - b.mean()
    al = np.dot(a, b) / max(np.dot(a, a), 1e-12)
    tn, e = al * a, b - al * a
    return 10 * np.log10(max(np.dot(tn, tn), 1e-12) / max(np.dot(e, e), 1e-12))

gap = slice(int(0.9 * SR), int(1.05 * SR))   # syllable trough
print(f"SI-SDR noisy    vs clean: {sisdr(clean, noisy):6.2f} dB")
print(f"SI-SDR enhanced vs clean: {sisdr(clean, enh):6.2f} dB")
print(f"floor RMS noisy/enh in trough: {noisy[gap].std():.4f} / {enh[gap].std():.4f}")
np.save("dbg_spec_in.npy", spec[0].astype(np.float32))          # frame 0, zeroed caches
np.save(f"dbg_spec_out_{TIER}.npy", out_frames[0].astype(np.float32))
np.save("reference_x.npy", noisy.astype(np.float32))
np.save("reference_y.npy", enh.astype(np.float32))
print("wrote reference_x.npy / reference_y.npy")
