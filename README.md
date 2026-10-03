# OpenNoiseFilter

Local speech noise suppression for browser and native audio applications.

OpenNoiseFilter runs FastEnhancer speech-enhancement models as standalone
WASM in the browser and generated C behind a Rust API. It includes a Web
Audio interface, a LiveKit processor, three model tiers, and model build
and benchmark tools. Audio inference runs locally.

**Experimental 0.1.0.** Model equivalence, streaming behavior, and Chromium
integration have been tested. Heavy models can underrun under load, and
broader browser/device validation is still needed. Use `low` on constrained
devices and test your target workloads before deployment.

This release is distributed on GitHub. npm and crates.io publication is
planned separately; use the source guides or GitHub package artifacts
until the registry packages are available.

## Components

| Directory | Purpose | Package |
|---|---|---|
| [open-noise-filter](open-noise-filter/README.md) | Web Audio, WASM, and LiveKit | npm: `open-noise-filter` |
| [open-noise-filter-rs](open-noise-filter-rs/README.md) | Native speech suppression | crates.io: `open-noise-filter-rs` |
| [open-noise-filter-wasm](open-noise-filter-wasm/README.md) | Model generation, verification, timing, and benchmarks | Source tooling; not an npm runtime package |

The neural tiers reduce background noise in speech. A separate JavaScript
gate attenuates quiet sections. Echo cancellation and speaker isolation
require separate components. OpenNoiseFilter uses upstream FastEnhancer
checkpoints; it does not train or claim ownership of those models.

## Try the browser demo

With Node 22.12 or newer:

```bash
git clone https://github.com/YaseenHQ/OpenNoiseFilter.git
cd OpenNoiseFilter/open-noise-filter
npm ci
npm run demo
```

Open `http://localhost:8080/demo/` and use headphones: the demo plays your
microphone back. Component guides cover package installation, APIs,
bundler asset loading, sample rates, latency, and development checks.

## Check the native library

Install Rust and a C compiler, then run from the repository root:

```bash
cd open-noise-filter-rs
cargo test
```

The crate's import name is `open_noise_filter`. The direct multi-channel
wrapper has a fixed 1024-sample startup delay at 48 kHz (21.3 ms).
Generated model calls share a mutex, so plan processing threads accordingly.

## Measurements and limits

The [saved benchmark](open-noise-filter-wasm/bench/RESULTS.md) covers 108
mixtures from six utterances, two speakers, six noise recordings, and three
SNR levels. Medium/high improve the recorded quality metrics on that
corpus. Automated scores are estimates, not a listening-panel result or a
guarantee for every voice and recording environment.

Use 48 kHz for neural processing where possible. The current cubic
resampler has no anti-alias filter. Timing measurements describe the test
machine; real-time behavior depends on hardware, model tier, scheduling,
and concurrent streams. Firefox, Safari, and mobile compatibility still
need dedicated validation.

## Development and releases

Run commands inside the relevant component folder. Root-level CI runs
browser/package checks, tooling tests, and Rust tests across Windows,
Linux, and macOS. The [repository guide](open-noise-filter/docs/MONOREPO.md)
describes the layout; the [release guide](docs/RELEASING.md) explains the
independent npm, crate, and GitHub release steps.

## License

Apache-2.0 for original OpenNoiseFilter code — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Each component also includes its own license
and third-party notices so they travel with separately distributed
packages. FastEnhancer model attribution is retained in those notices.
