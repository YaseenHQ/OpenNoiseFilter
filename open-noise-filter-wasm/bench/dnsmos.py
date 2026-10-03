#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""DNSMOS P.835 scoring (SIG, BAK, OVRL, P808) for bench/score.py.

Extracted from real-tse/utils/dnsmos_eval.py — only the pieces score.py uses
(DNSMOSComputeScore, _ensure_dnsmos_models and their helpers); the
pandas/tqdm/dataset_lang dependencies were dropped.

MIT License — Copyright (c) 2025 REAL-TSE

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# DNSMOS constants (aligned with Microsoft DNS-Challenge dnsmos_local.py)
SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

DNSMOS_HF_REPO = "Vyvo-Research/dnsmos"
DNSMOS_FILES = ("sig_bak_ovr.onnx", "model_v8.onnx")


def _download_dnsmos_models(model_dir: Path) -> None:
    """Download sig_bak_ovr.onnx and model_v8.onnx from Hugging Face if missing."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit(
            "Auto-download requires huggingface_hub. Install with: pip install huggingface_hub"
        ) from e
    model_dir.mkdir(parents=True, exist_ok=True)
    for fname in DNSMOS_FILES:
        dest = model_dir / fname
        if dest.exists():
            continue
        print(f"[DNSMOS] Downloading {fname} from {DNSMOS_HF_REPO} ...")
        path = hf_hub_download(
            repo_id=DNSMOS_HF_REPO,
            filename=fname,
            local_dir=str(model_dir),
        )
        print(f"[DNSMOS] Saved to {path}")


def _ensure_dnsmos_models(model_dir: Path, allow_download: bool) -> None:
    """Ensure both ONNX files exist; optionally download from Hugging Face if missing."""
    primary = model_dir / "sig_bak_ovr.onnx"
    p808 = model_dir / "model_v8.onnx"
    if primary.exists() and p808.exists():
        return
    if allow_download:
        _download_dnsmos_models(model_dir)
        return
    raise SystemExit(
        f"DNSMOS models not found in {model_dir}. "
        "Need sig_bak_ovr.onnx and model_v8.onnx."
    )


# ----- DNSMOS inference (aligned with Microsoft DNS-Challenge dnsmos_local.py) -----


def _audio_melspec(
    audio: np.ndarray,
    n_mels: int = 120,
    frame_size: int = 320,
    hop_length: int = 160,
    sr: int = 16000,
    to_db: bool = True,
) -> np.ndarray:
    import librosa

    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
    )
    if to_db:
        mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
    return mel_spec.T


def _get_polyfit_val(
    sig_raw: float, bak_raw: float, ovr_raw: float, is_personalized_MOS: bool
) -> Tuple[float, float, float]:
    if is_personalized_MOS:
        p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
        p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
        p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
    else:
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
    sig_poly = float(p_sig(sig_raw))
    bak_poly = float(p_bak(bak_raw))
    ovr_poly = float(p_ovr(ovr_raw))
    return sig_poly, bak_poly, ovr_poly


def _get_onnx_providers(provider: str) -> List[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise SystemExit(
                "CUDAExecutionProvider not available. Install onnxruntime-gpu or set --provider cpu."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    # auto: prefer CUDA if available
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class DNSMOSComputeScore:
    """DNSMOS scorer using sig_bak_ovr.onnx and model_v8.onnx (P808)."""

    def __init__(
        self,
        primary_model_path: Path,
        p808_model_path: Path,
        is_personalized_MOS: bool = False,
        provider: str = "auto",
    ) -> None:
        import onnxruntime as ort

        self._is_personalized = is_personalized_MOS
        providers = _get_onnx_providers(provider)
        self._onnx_sess = ort.InferenceSession(
            str(primary_model_path),
            providers=providers,
        )
        self._p808_sess = ort.InferenceSession(
            str(p808_model_path),
            providers=providers,
        )

    def __call__(self, audio_path: Path, sampling_rate: int = SAMPLING_RATE) -> Dict[str, float]:
        import librosa
        import soundfile as sf

        aud, input_fs = sf.read(str(audio_path))
        if input_fs != sampling_rate:
            audio = librosa.resample(aud, orig_sr=input_fs, target_sr=sampling_rate)
        else:
            audio = aud
        len_samples = int(INPUT_LENGTH * sampling_rate)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / sampling_rate) - INPUT_LENGTH) + 1
        hop_len_samples = sampling_rate
        pred_sig, pred_bak, pred_ovr, pred_p808 = [], [], [], []

        for idx in range(num_hops):
            start = int(idx * hop_len_samples)
            end = int((idx + INPUT_LENGTH) * hop_len_samples)
            audio_seg = audio[start:end]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg, dtype=np.float32)[np.newaxis, :]
            p808_input = np.array(
                _audio_melspec(audio=audio_seg[:-160]), dtype=np.float32
            )[np.newaxis, :, :]
            oi = {"input_1": input_features}
            p808_oi = {"input_1": p808_input}

            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self._onnx_sess.run(None, oi)[0][0]
            sig, bak, ovr = _get_polyfit_val(
                float(mos_sig_raw),
                float(mos_bak_raw),
                float(mos_ovr_raw),
                self._is_personalized,
            )
            p808_mos = float(self._p808_sess.run(None, p808_oi)[0][0][0])
            pred_sig.append(sig)
            pred_bak.append(bak)
            pred_ovr.append(ovr)
            pred_p808.append(p808_mos)

        return {
            "SIG": float(np.mean(pred_sig)),
            "BAK": float(np.mean(pred_bak)),
            "OVRL": float(np.mean(pred_ovr)),
            "P808": float(np.mean(pred_p808)),
        }
