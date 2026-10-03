# OpenNoiseFilter — Web Audio

Local speech noise suppression for the Web Audio API and LiveKit.
The `open-noise-filter` package runs FastEnhancer models compiled to WASM,
with an AudioWorklet and optional workers. Model weights are embedded in
the WASM assets, which load when you initialize the filter. Audio processing
stays in the browser; no inference service or ONNX Runtime is needed.

Status: **experimental 0.1**. Model equivalence, buffering, and browser
integration have been tested. Real-time performance depends on the device,
model tier, and competing workloads. Prefer `low` on constrained devices.
The neural tiers reduce background noise in speech; the `gate` tier
attenuates quiet sections. Echo cancellation and speaker isolation need
separate components.

The project also includes a [native Rust library](../open-noise-filter-rs/README.md)
and [model build and benchmark tools](../open-noise-filter-wasm/README.md).
The [repository guide](docs/MONOREPO.md) describes their sibling-folder layout
under `YaseenHQ/OpenNoiseFilter`.

The npm registry command below applies once the package is published there.
For the GitHub-only `v0.1.0` release, install its package artifact directly:

```bash
npm install https://github.com/YaseenHQ/OpenNoiseFilter/releases/download/v0.1.0/open-noise-filter-0.1.0.tgz
```

To try the source locally, use the [demo](#demo).

```
npm install open-noise-filter
```

## Web Audio usage

```ts
import { createNoiseFilter } from "open-noise-filter";

// `source` is an AudioNode in this context, such as a microphone source.
const ctx = new AudioContext({ sampleRate: 48000 });
const filter = await createNoiseFilter(ctx, { quality: "medium" });
source.connect(filter.node);
filter.node.connect(ctx.destination);

filter.setEnabled(false); // copy-through bypass
filter.destroy();         // releases the worklet, worker, and ports
```

## LiveKit usage

```ts
import { LiveKitNoiseFilter, isLiveKitNoiseFilterSupported } from "open-noise-filter/livekit";

// `room` is your connected LiveKit Room.
if (isLiveKitNoiseFilterSupported()) {
  const publication = await room.localParticipant.setMicrophoneEnabled(true);
  const track = publication?.audioTrack;
  if (track) {
    await track.setProcessor(LiveKitNoiseFilter({ quality: "medium" }));
  }
}
```

## Options

Core (`createNoiseFilter(ctx, options)`):

| option | values | default | notes |
|--------|--------|---------|-------|
| `quality` | `"low" \| "medium" \| "high" \| "gate"` | `"medium"` | FastEnhancer Tiny/Base/Small, or a pure-JS noise gate |
| `thread` | `"audio" \| "worker"` | `"worker"` for `medium` and `high`; else `"audio"` | worker mode moves the DSP off the audio thread (+512 samples latency in mono); always `"audio"` for OfflineAudioContext and for `gate` |
| `maxChannels` | `1 \| 2` | `1` | `1` downmixes stereo input to mono; `2` runs one engine per channel |
| `gate` | `{ openThreshold?, closeThreshold?, holdMs? }` | −45 / −55 dBFS, 200 ms | only used when `quality` is `"gate"` |
| `enabled` | boolean | `true` | initial DSP state |
| `urls` | `{ worklet?, worker?, wasm?, scalarWasm? }` | bundled | override if self-hosting assets |
| `onStats` | `(stats) => void` | — | worker-mode jitter buffer stats: underruns / overruns / skips / target |

LiveKit extras (`open-noise-filter/livekit`):

| option | notes |
|--------|-------|
| `debugLogs` | lifecycle + worker-stats logging |

## Quality tiers

| quality | model | wasm | default thread | initial latency |
|---------|-------|------|----------------|-----------------|
| `low`    | FastEnhancer Tiny  | 193 KB | audio  | 896 samples ≈ 18.7 ms |
| `medium` | FastEnhancer Base  | 580 KB | worker | 1408 samples ≈ 29.3 ms |
| `high`   | FastEnhancer Small | 954 KB | worker | 1408 samples ≈ 29.3 ms |
| `gate`   | —    | none | audio  | 0 |

`medium` defaults to a worker because its measured audio-thread p99 frame
cost was 3.1 ms on the dev machine, above the 2.67 ms render quantum.
You can opt into `thread: "audio"` after testing your target devices.
`low` measured p99 0.7 ms on the same machine; costs vary with hardware
and load.

At 48 kHz, worker mode uses an adaptive jitter buffer: latency starts at 1408 samples
≈ 29.3 ms (1920 ≈ 40 ms in stereo, which starts one hop deeper to absorb
two-worker startup jitter) and grows by one hop per underrun, up to 2432
samples ≈ 50.7 ms (it does not shrink back within a session). `high` needs
it — a Small frame costs ~4.5 ms, more than the 2.67 ms audio quantum.
The `latency` property describes initial latency at 48 kHz. It excludes
capture/output device buffers, transport latency, and later rebuffering.
Resampling and worker scheduling can add delay at other rates.

`gate` is a noise gate, not an enhancer: near-zero CPU, zero added latency,
works at any sample rate — it attenuates the signal when the level drops
below `closeThreshold` instead of removing noise during speech. Per-quantum
RMS level (linked across channels) with open/close hysteresis, a `holdMs`
hold timer, and per-sample gain smoothing (5 ms attack, 50 ms release).

## Sample rates

A 48 kHz AudioContext is recommended. Other rates run through a built-in
cubic resampler inside the worklet (a `console.warn` is emitted once).
Measured vs the native 48 kHz output: ~33 dB SNR at 44.1 kHz and ~25 dB at
16 kHz — the latter limited by aliasing, since the cubic interpolator has no
anti-alias filter.

With livekit-client, supply a 48 kHz context via RoomOptions:

```ts
import { Room } from "livekit-client";

const audioContext = new AudioContext({ sampleRate: 48000 });
const room = new Room({ webAudioMix: { audioContext } });
```

## Alternate node API (`open-noise-filter/compat`)

If you want to manage the worklet module and WASM binary yourself instead of
the all-in-one `createNoiseFilter`, the compat entry exposes a
load-then-construct pattern:

```ts
import { loadNoiseFilter, NoiseFilterWorkletNode } from "open-noise-filter/compat";

// scalarUrl is the non-SIMD fallback (ships as *_scalar.wasm next to each model)
const wasmBinary = await loadNoiseFilter({
  url: "/fastenhancer_b.wasm", scalarUrl: "/fastenhancer_b_scalar.wasm",
});
// Copy the exported worklet.js asset to this served URL first.
await ctx.audioWorklet.addModule("/noise-filter/worklet.js");
const node = new NoiseFilterWorkletNode(ctx, { wasmBinary, maxChannels: 2 });
source.connect(node); node.connect(ctx.destination);
node.setEnabled(false); // copy-through bypass
```

A pure-JS `NoiseGateWorkletNode` is also available (options
`openThreshold`, `closeThreshold`, `holdMs`, `maxChannels`). Compat nodes
run on the audio thread only: for `high` or `medium`, use
`createNoiseFilter()` so the DSP runs in a Worker.

## Custom models

The neural tiers are just WASM files exposing the `fe_init` / `fe_run` /
`fe_in_ptr` / `fe_out_ptr` ABI — swap in your own build via `urls`:

```ts
import { createNoiseFilter } from "open-noise-filter";

createNoiseFilter(ctx, {
  quality: "medium",                 // picks the thread policy
  urls: {
    wasm: "/models/my_model.wasm",          // used when WASM SIMD exists
    scalarWasm: "/models/my_model_scalar.wasm",
  },
});
```

Build your own from a FastEnhancer-compatible spec ONNX (same I/O contract:
`spec_in` [1,513,1,2] + `cache_in_*`/`cache_out_*`) in the
`open-noise-filter-wasm` component:

```bash
node build.mjs --model path/to/my_spec.onnx --name my_model --shim b --variant all
```

`--shim` selects the cache geometry your checkpoint was trained with
(t: 2×24×20, b: 3×36×36, s: 3×48×48). Custom binaries are still verified
against ONNX Runtime — from that same component directory:

```bash
python scripts/reference.py path/to/my_spec.onnx   # writes reference + dbg npys
FE_WASM=out/my_model.wasm node scripts/specdiff.mjs my   # "my" = onnx basename minus _spec
FE_WASM=out/my_model.wasm node scripts/equiv.mjs
```

## Bundlers

For Vite, import the assets as URLs and pass them explicitly. This lets
Vite emit the worklet, worker, and model files in production builds:

```ts
import { createNoiseFilter } from "open-noise-filter";
import worklet from "open-noise-filter/worklet.js?url";
import worker from "open-noise-filter/worker.js?url";
import wasm from "open-noise-filter/wasm/fastenhancer_b.wasm?url";
import scalarWasm from "open-noise-filter/wasm/fastenhancer_b_scalar.wasm?url";

const filter = await createNoiseFilter(ctx, {
  quality: "medium",
  urls: { worklet, worker, wasm, scalarWasm },
});
```

For other bundlers, copy these files to a public asset directory and pass
their served URLs through `urls`. The default relative URLs work when
serving the ESM distribution directly with its directory layout intact.

Both ESM and CommonJS builds ship (`require("open-noise-filter")` resolves
to `dist/index.cjs`). The CJS bundle can't resolve bundled assets, so there
`urls.worklet`/`urls.wasm` (and `urls.worker` for worker mode) are required.

## Browser requirements

Use a secure context (HTTPS or localhost) with AudioWorklet support.
`isNoiseFilterSupported()` / `isLiveKitNoiseFilterSupported()` check for a
browser window and AudioWorkletNode. They do not test real-time performance
or validate your hosting configuration. Worker mode also needs Web Workers.

WebAssembly SIMD is **recommended but not required**: browsers without it
get the bundled scalar (`*_scalar.wasm`) builds instead, with reduced
defaults — measured scalar frame cost forces most tiers off the audio
thread (see the tools component's `cost.mjs`):

| quality | mono | stereo |
|---------|------|--------|
| `low` | audio | worker |
| `medium` | worker | worker |
| `high` | **falls back to `medium`** (a scalar Small frame ≈ 11.6 ms > the 10.67 ms hop) | same |
| `gate` | audio | audio |

Self-hosting: pass `urls.scalarWasm` to override the scalar binary, or
`urls.wasm` to override outright. `loadNoiseFilter({ url, scalarUrl })` in
`/compat` does the same selection.

## Quality

Measured on 108 noisy mixtures (Arctic speech × 6 MS-SNSD noise types ×
5/10/15 dB SNR), scored with DNSMOS P.835, STOI, and SI-SDR vs the clean
reference:

| engine | SIG | BAK | OVRL | STOI | ΔSI-SDR |
|--------|-----|-----|------|------|---------|
| `high`   | 3.25 | 3.62 | 2.78 | 0.964 | +6.1 |
| `medium` | 3.20 | 3.59 | 2.72 | 0.958 | +5.7 |
| `low`    | 3.07 | 3.46 | 2.55 | 0.944 | +4.2 |
| GTCRN    | 3.14 | 3.40 | 2.57 | 0.943 | +2.7 |
| RNNoise  | 3.20 | 3.11 | 2.46 | 0.928 | +1.3 |
| Speex    | 3.17 | 2.73 | 2.29 | 0.897 | −1.5 |
| unprocessed | 3.17 | 2.29 | 2.12 | 0.918 | — |

These measurements cover six utterances from two speakers and six noise
recordings. They describe this corpus, rather than all voices or recording
conditions; DNSMOS is an automated quality estimate, not a listening panel.

The [benchmark harness](../open-noise-filter-wasm/bench/README.md) and model
build pipeline live in `open-noise-filter-wasm/`. No listening-panel score,
voice-isolation guarantee, or cross-device performance claim is implied.

## Demo

```
npm ci
npm run demo
```

then open http://localhost:8080/demo/ (use headphones — it plays the filtered
microphone back).

## Development checks

```bash
npm ci
npm run typecheck
npm test
npx playwright install chromium
npm run test:browser
```

The browser check uses a packed npm artifact in Vite development and
production builds. It checks asset loading, every quality tier, scalar
WASM, resampled worker output, bypass, suspend/resume, and the LiveKit
processor lifecycle. It plays synthetic audio through a muted output.
Use Node 22.12 or newer for these development checks.
Set `BROWSER_EXECUTABLE` to test with an existing Chromium, Chrome, or Edge
executable instead of downloading Chromium.

## License

Apache-2.0 for original OpenNoiseFilter code — see [LICENSE](LICENSE) and [NOTICE](NOTICE); third-party components in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
