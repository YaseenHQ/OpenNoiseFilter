#!/usr/bin/env python3
"""Build the benchmark corpus: clean Arctic speech + MS-SNSD noise at fixed SNRs.

Output layout (all 48 kHz mono 16-bit wav):
  corpus/{clip_id}_clean.wav   — the speech reference
  corpus/{clip_id}_noisy.wav   — the mixture fed to every engine
clip_id = c{spk}{utt}_n{noise}_s{snr}
"""
import os
import sys
import wave
from pathlib import Path

import numpy as np
import scipy.signal as sg

ROOT = Path(__file__).parent
ARCTIC_DIR = Path(os.environ.get("ARCTIC_DIR", ROOT / "arctic"))
ARCTIC = {
    "bdl": ARCTIC_DIR / "cmu_us_bdl_arctic" / "wav",
    "slt": ARCTIC_DIR / "cmu_us_slt_arctic" / "wav",
}
NOISE_DIR = ROOT / "noise"
OUT = ROOT / "corpus"

SR = 48000
CLIP_S = 2.8
# (speaker, utterance idx) — varied phonetic content, skip the first few (short)
CLEAN = [("bdl", 6), ("bdl", 13), ("bdl", 17), ("bdl", 20), ("slt", 7), ("slt", 16), ("slt", 19), ("slt", 35)]
NOISES = ["Babble_1", "CafeTeria_1", "Typing_1", "VacuumCleaner_1", "NeighborSpeaking_1", "AirConditioner_1"]
SNRS = [5, 10, 15]


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
    return x, sr


def resample48(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return x
    g = np.gcd(sr, SR)
    return sg.resample_poly(x, SR // g, sr // g)


def write_wav(path: Path, x: np.ndarray):
    y = np.clip(x, -1, 1)
    y16 = (y * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(y16.tobytes())


def rms(x):
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def main():
    for spk, d in ARCTIC.items():
        if not d.is_dir():
            sys.exit(
                f"Arctic wavs not found: {d}\n"
                "Set ARCTIC_DIR to a directory containing cmu_us_bdl_arctic/ and "
                "cmu_us_slt_arctic/ (see bench/README.md), or place them under bench/arctic/."
            )
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(20260921)

    cleans = []
    for spk, idx in CLEAN:
        x, sr = read_wav(ARCTIC[spk] / f"arctic_a{idx:04d}.wav")
        x = resample48(x, sr)
        n = int(CLIP_S * SR)
        # take the loudest 4 s window so clips are speech-active throughout
        e = sg.convolve(x * x, np.ones(480), "valid")
        start = int(np.argmax(e))
        x = x[start : start + n]
        if len(x) < n:
            print(f"short clip {spk}{idx}, skipped"); continue
        x = x / max(rms(x), 1e-6) * 0.1  # normalize to rms 0.1 (~ -20 dBFS)
        cleans.append((f"c{spk}{idx}", x))

    noises = {}
    for name in NOISES:
        x, sr = read_wav(NOISE_DIR / f"{name}.wav")
        x = resample48(x, sr)
        noises[name] = x

    n_clips = 0
    manifest = []
    for cname, c in cleans:
        for nname, nz in noises.items():
            n = len(c)
            for snr in SNRS:
                # random noise window (files are 20-30 s+); wrap if short
                off = int(rng.integers(0, max(1, len(nz) - n)))
                seg = nz[off : off + n]
                if len(seg) < n:
                    seg = np.resize(seg, n)
                seg = seg / max(rms(seg), 1e-9) * rms(c) / (10 ** (snr / 20))
                mix = c + seg
                peak = np.abs(mix).max()
                if peak > 0.98:  # headroom
                    mix *= 0.98 / peak
                cid = f"{cname}_n{nname.split('_')[0].lower()}_s{snr}"
                write_wav(OUT / f"{cid}_clean.wav", c)
                write_wav(OUT / f"{cid}_noisy.wav", mix)
                manifest.append((cid, nname, snr))
                n_clips += 1
    with open(OUT / "manifest.txt", "w") as f:
        f.write("\n".join(m[0] for m in manifest))
    print(f"wrote {n_clips} clip pairs to {OUT}")


if __name__ == "__main__":
    main()
