# OpenNoiseFilter — Rust

Local speech noise suppression for native programs. The
`open-noise-filter-rs` crate uses the same FastEnhancer checkpoints as the
[Web Audio package](../open-noise-filter/README.md).

Status: **experimental 0.1**. Model equivalence and streaming behavior have
been tested on Windows. Cross-platform CI is configured; broader device and
real-time workload validation is still needed. This library targets speech
noise suppression; echo cancellation and speaker isolation require separate
components.

The model is the vendored `onnx2c`-generated C from the
[`open-noise-filter-wasm`](../open-noise-filter-wasm/README.md) tools component (`csrc/`),
compiled at build time by `cc` — no ONNX Runtime, no runtime deps beyond
the C toolchain. The STFT → model → iSTFT → overlap-add front-end is a
port of the browser library's `FrameEngine` (`src/dsp.js`).

## Install

For the GitHub `v0.1.0` release, add:

```toml
[dependencies]
open-noise-filter-rs = { git = "https://github.com/YaseenHQ/OpenNoiseFilter.git", tag = "v0.1.0" }
```

After the crate is published to crates.io, you can use:

```toml
[dependencies]
open-noise-filter-rs = "0.1"
```

The crate is named `open-noise-filter-rs`; its Rust import name is
`open_noise_filter`. To work from source, run `cargo test` in this component
directory with a C compiler installed.

## Contract

- `DenoiseState` is the raw engine: **48 kHz, 512-sample hops, mono**.
  First `process_frame` after `new()`/`reset()` only primes the window and
  returns `false`; subsequent calls return `true` and fill `out`.
- `ResampledDenoiseState` wraps it with a Catmull-Rom adapter for context
  rates from 8–192 kHz; blocks can be any length and chunking does not change
  the output stream.
- `DenoiseChannels` runs one state per channel with interleaved or planar
  access at 8–192 kHz. The direct 48 kHz path has a fixed 1024-sample
  startup delay (21.3 ms); resetting restores that delay. Device buffers and
  caller scheduling add their own latency.
- `DenoiseState` is `Send`. Calls are serialized internally — the
  generated C keeps scratch tensors in `static` storage, so a global mutex
  covers each model call. Multiple streams share that lock. Run the model
  on a processing thread rather than assuming a lock-free audio callback.

```rust
use open_noise_filter::{DenoiseChannels, DenoiseState, Tier};

// raw 48 kHz engine
let mut st = DenoiseState::new(Tier::Base);
let input = [0.0f32; 512];
let mut out = [0.0f32; 512];
assert!(!st.process_frame(&input, &mut out)); // priming hop

// stereo at an arbitrary context rate, interleaved in place
let mut stereo = DenoiseChannels::new(Tier::Base, 2, 44100.0);
let mut buf = vec![0.0f32; 2 * 128];
stereo.process_interleaved(&mut buf);
```

## Tiers

`Tier::Tiny` / `Tier::Base` / `Tier::Small` — same checkpoints as the
browser package's `low` / `medium` / `high`. Measured on the same dev
machine as the WASM build: ~0.50 / ~1.69 / ~4.18 ms per frame
(RTF 0.05 / 0.16 / 0.39). A frame represents 10.67 ms of audio. These are
single-stream measurements on one machine; throughput and deadline safety
depend on hardware, workload, and contention from other instances.

## Accuracy

`cargo test` compares the native pipeline against golden vectors generated
by the shipped WASM worklet on the shared reference clip:

| tier | maximum absolute sample difference |
|---|---:|
| Tiny 48 kHz | 8.3e-7 |
| Base 48 kHz | 4.8e-6 |
| Small 48 kHz | 3.2e-6 |
| Base 44.1 kHz (resampled, aligned) | 6.6e-7 |

The causal channel wrapper adds 128 samples to the worklet's 896-sample
direct delay. At 44.1 kHz, the causal resampler is aligned by five context
samples against the worklet reference before measuring the difference.
The raw frame engine's model output is unchanged.

The WASM worklet output itself verifies against ONNX Runtime at ~4e-6, so
the native build is equivalent within fast-math noise.

## Rebuilding the model sources

`csrc/fe_{t,b,s}_norm.c` are generated artifacts — regenerate them with
`node build.mjs` in the sibling `open-noise-filter-wasm` component, then copy
`build/fastenhancer_{t,b,s}_norm.c` to `csrc/fe_{t,b,s}_norm.c`, respectively.
To swap in a custom FastEnhancer-compatible
checkpoint, build it there with `--model`/`--shim`, vendor the generated
`.c` under a new name, and add a `Tier` variant + `sys` extern.

## Build requirements

A C compiler via `cc` (tested with MSVC 2022 Build Tools; GCC/Clang
should work — the crate only needs C17 + `-ffast-math`-equivalent flags).
Build time is ~2 min for all three tiers.

## License

Apache-2.0 for original OpenNoiseFilter code — see [LICENSE](LICENSE) and [NOTICE](NOTICE). The generated C model sources derive from
FastEnhancer checkpoints by AHN Sung Hwan; the attribution and license are
included in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
