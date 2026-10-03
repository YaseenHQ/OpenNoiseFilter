# bench — the noise-suppression benchmark

Renders a fixed noisy corpus through the built-in engines and, optionally,
externally supplied comparison engines. The measured table lives in
[RESULTS.md](RESULTS.md).

Run every command in this guide from `open-noise-filter-wasm/bench/`.
The saved results cover six utterances from two speakers and six noise
recordings. Automated scores measure this corpus; they do not establish
general voice-isolation ability or real-time performance across devices.

## Corpus

- **Clean speech**: CMU Arctic (bdl male + slt female), resampled to 48 kHz,
  2.8 s clips, loudness-normalized.
- **Noise**: 6 real recordings from
  [MS-SNSD](https://github.com/microsoft/MS-SNSD) (`noise_train/`):
  Babble, CafeTeria, Typing, VacuumCleaner, NeighborSpeaking, AirConditioner.
- **Mixtures**: each clean clip × each noise at 5/10/15 dB SNR → 108 pairs.

## Setup

```bash
python -m venv ../.venv            # or reuse an existing env
../.venv/Scripts/pip install -r requirements.txt   # Windows
# ../.venv/bin/pip install -r requirements.txt     # macOS/Linux
```

## Fetch the data

**Noise** — six files from MS-SNSD `noise_train/` into `bench/noise/`:

```bash
mkdir -p noise
for n in Babble_1 CafeTeria_1 Typing_1 VacuumCleaner_1 NeighborSpeaking_1 AirConditioner_1; do
  curl -L -o "noise/$n.wav" "https://raw.githubusercontent.com/microsoft/MS-SNSD/master/noise_train/$n.wav"
done
```

**Clean speech** — CMU Arctic 0.95-release tarballs, extracted so the layout is
`$ARCTIC_DIR/cmu_us_{bdl,slt}_arctic/wav/arctic_aNNNN.wav`:

```bash
mkdir -p arctic && cd arctic
curl -LO http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_bdl_arctic-0.95-release.tar.bz2
curl -LO http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_slt_arctic-0.95-release.tar.bz2
tar xf cmu_us_bdl_arctic-0.95-release.tar.bz2
tar xf cmu_us_slt_arctic-0.95-release.tar.bz2
cd ..
```

`make_corpus.py` reads `$ARCTIC_DIR` (default `bench/arctic/`).

## Run it

```bash
python make_corpus.py   # writes corpus/*_{clean,noisy}.wav (108 pairs)
node render.mjs         # built-in engines only
python score.py fe_t fe_b fe_s passthrough  # scores this run's built-in engines
```

To include the optional comparison engines, point `COMPARISON_DIR` at a
compatible extracted `dist` directory. It should contain
`rnnoise/workletProcessor.js`, `rnnoise_simd.wasm`,
`gtcrn/workletProcessor.js`, `gtcrn.wasm`, `speex/workletProcessor.js`,
`speex.wasm`, and `noiseGate/workletProcessor.js`. Then run
`COMPARISON_DIR=/path/to/dist node render.mjs` (on PowerShell, set the
environment variable with `$env:COMPARISON_DIR = 'C:/path/to/dist'`).
The assets remain outside this repository; obtain them separately and follow
their included license notices. To render only selected engines, pass their
IDs, for example `node render.mjs fe_b rnnoise`.

`score.py` auto-downloads the DNSMOS P.835 ONNX models
(`sig_bak_ovr.onnx` + `model_v8.onnx`) into `bench/dnsmos_models/` on first
run. PESQ is not computed (no prebuilt Windows wheel).

## How scoring works

- **Alignment**: each engine's output is cross-correlated against the noisy
  input to find its algorithmic latency, then compared against the clean
  reference. Measured lags: fastenhancer 896, gtcrn 1530, rnnoise ~991,
  speex 128 samples.
- **DNSMOS P.835** → SIG / BAK / OVRL / P808 (perceptual, no reference).
- **STOI** (pystoi, 16 kHz) and **SI-SDR** vs the clean reference.
- **RTF** is measured inside `render.mjs` — wall time ÷ clip duration.

## Notes

- `gate` scores identically to `passthrough` on this corpus — the clips are
  speech-active throughout, so a VAD gate never closes. That's the no-op
  floor, not a failure of the gate.
- corpus/, out/, noise/, dnsmos_models/, arctic/ are all git-ignored — they're
  regenerable artifacts.
