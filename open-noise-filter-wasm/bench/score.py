#!/usr/bin/env python3
# Copyright 2026 YaseenHQ
# SPDX-License-Identifier: Apache-2.0

"""Score bench outputs: DNSMOS P.835 (SIG/BAK/OVRL) + STOI + SI-SDR.

Reads bench/corpus/{id}_clean.wav + {id}_noisy.wav and bench/out/<engine>/{id}.wav.
Enhanced signals are aligned to clean by cross-correlation against the noisy
input (each engine has its own algorithmic latency).

Usage: python score.py            — all engines in bench/out
       python score.py fe_b gate  — subset
Writes bench/out/scores.csv and prints the aggregate table.
"""
import csv
import sys
import wave
from pathlib import Path

import numpy as np
import scipy.signal as sg

ROOT = Path(__file__).parent
CORPUS = ROOT / "corpus"
OUT = ROOT / "out"
SR = 48000
METRIC_SR = 16000  # resample everything to 16k for STOI/SI-SDR + DNSMOS

from dnsmos import DNSMOSComputeScore, _ensure_dnsmos_models  # noqa: E402


def read_wav(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768
    return x


def to16k(x: np.ndarray) -> np.ndarray:
    return sg.resample_poly(x, 1, 3)  # 48k -> 16k


def sisdr(ref: np.ndarray, est: np.ndarray) -> float:
    n = min(len(ref), len(est))
    ref, est = ref[:n] - ref[:n].mean(), est[:n] - est[:n].mean()
    a = np.dot(est, ref) / max(np.dot(ref, ref), 1e-12)
    e = a * ref - est
    return float(10 * np.log10(np.dot(a * ref, a * ref) / max(np.dot(e, e), 1e-12)))


def find_lag(noisy: np.ndarray, est: np.ndarray, max_lag=9600) -> int:
    """FFT-based xcorr; returns lag L so est[t] aligns with noisy[t-L]."""
    n = len(noisy) + len(est)
    nf = 1 << (n - 1).bit_length()
    N = np.fft.rfft(noisy, nf)
    E = np.fft.rfft(est, nf)
    corr = np.fft.irfft(E * np.conj(N), nf)
    # corr[k] = sum_t est[t] * noisy[t-k] for k>=0 (wrap for negative k)
    lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(nf - max_lag, nf)])
    k = lags[np.argmax(np.abs(corr[lags]))]
    return int(k if k <= max_lag else k - nf)


def stoi_score(ref16: np.ndarray, est16: np.ndarray) -> float:
    from pystoi import stoi
    n = min(len(ref16), len(est16))
    return float(stoi(ref16[:n], est16[:n], METRIC_SR, extended=False))


def main():
    engines = sys.argv[1:] or sorted(
        p.name for p in OUT.iterdir() if p.is_dir()
    )
    model_dir = ROOT / "dnsmos_models"
    _ensure_dnsmos_models(model_dir, allow_download=True)
    scorer = DNSMOSComputeScore(
        model_dir / "sig_bak_ovr.onnx", model_dir / "model_v8.onnx", provider="cpu"
    )

    clips = [p.name.replace("_noisy.wav", "") for p in CORPUS.glob("*_noisy.wav")]
    rows = []
    for eng in engines:
        edir = OUT / eng
        if not edir.is_dir():
            continue
        for cid in sorted(clips):
            ep = edir / f"{cid}.wav"
            if not ep.exists():
                continue
            clean = read_wav(CORPUS / f"{cid}_clean.wav")
            noisy = read_wav(CORPUS / f"{cid}_noisy.wav")
            est = read_wav(ep)

            lag = find_lag(noisy, est)
            est_al = est[lag : lag + len(clean)] if lag >= 0 else est[: len(clean)]
            if len(est_al) < len(clean):
                est_al = np.pad(est_al, (0, len(clean) - len(est_al)))
            n = len(clean)
            si = sisdr(to16k(clean), to16k(est_al[:n]))
            si_n = sisdr(to16k(clean), to16k(noisy[:n]))
            try:
                st = stoi_score(to16k(clean), to16k(est_al[:n]))
                st_n = stoi_score(to16k(clean), to16k(noisy[:n]))
            except Exception:
                st = st_n = float("nan")
            try:
                d = scorer(str(ep))  # scorer resamples to 16 k internally
            except Exception as exc:
                d = {"SIG": float("nan"), "BAK": float("nan"), "OVRL": float("nan"), "P808": float("nan")}
                print(f"dnsmos fail {eng}/{cid}: {exc}", file=sys.stderr)
            rows.append({
                "engine": eng, "clip": cid, "lag": lag,
                "SIG": d["SIG"], "BAK": d["BAK"], "OVRL": d["OVRL"], "P808": d["P808"],
                "STOI": st, "STOI_noisy": st_n,
                "SISDR": si, "SISDR_noisy": si_n, "dSISDR": si - si_n,
            })
            print(f"{eng:12s} {cid:38s} lag={lag:5d} SIG={d['SIG']:.2f} BAK={d['BAK']:.2f} OVL={d['OVRL']:.2f} STOI={st:.3f} dSISDR={si - si_n:+.2f}")

    csv_path = OUT / "scores.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== AGGREGATE (mean over clips) ===")
    print(f"{'engine':12s} {'SIG':>6s} {'BAK':>6s} {'OVRL':>6s} {'P808':>6s} {'STOI':>6s} {'dSI-SDR':>8s}")
    for eng in engines:
        rs = [r for r in rows if r["engine"] == eng]
        if not rs:
            continue
        m = lambda k: float(np.nanmean([r[k] for r in rs]))
        print(f"{eng:12s} {m('SIG'):6.3f} {m('BAK'):6.3f} {m('OVRL'):6.3f} {m('P808'):6.3f} {m('STOI'):6.3f} {m('dSISDR'):+8.2f}")
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
