// Copyright 2026 YaseenHQ
// SPDX-License-Identifier: Apache-2.0

/**
 * open-noise-filter — FastEnhancer (arXiv:2509.21867) noise suppression for
 * Web Audio. WASM + AudioWorklet; heavy tiers run in Workers behind an
 * adaptive jitter buffer. Stereo via `maxChannels: 2`; non-48 kHz contexts
 * resample inside the worklet.
 *
 *   const filter = await createNoiseFilter(ctx, { quality: "medium" });
 *   source.connect(filter.node);
 *   filter.node.connect(ctx.destination);
 */

import { isSimdSupported } from "./simd.js";

export { isSimdSupported };

export type Quality = "low" | "medium" | "high" | "gate";
type WasmQuality = Exclude<Quality, "gate">;

export interface FilterStats {
  underruns: number;
  overruns: number;
  skips: number;
  target: number;
}

export interface GateOptions {
  /** Open the gate at or above this RMS level, dBFS (default -45). */
  openThreshold?: number;
  /** Begin closing once the level stays below this, dBFS (default -55). */
  closeThreshold?: number;
  /** Hold time before the gate closes, ms (default 200). */
  holdMs?: number;
}

export interface NoiseFilterOptions {
  /** "low" | "medium" | "high" map to FastEnhancer Tiny/Base/Small;
   * "gate" is a pure-JS noise gate (latency 0, any sample rate).
   * Default "medium". */
  quality?: Quality;
  /** "audio" runs the DSP on the audio thread; "worker" moves it to a Worker
   * per channel (+512 samples latency, growing +512 per underrun up to
   * +1536). Default: "worker" when quality is "medium" or "high" (measured
   * medium p99 exceeds the 2.67 ms audio quantum), else "audio". Always
   * "audio" for an OfflineAudioContext: a worker can't
   * keep up with faster-than-realtime rendering; an explicit "worker"
   * request there is warned about and overridden. "gate" always runs on the
   * audio thread. Without WASM SIMD (scalar build) the default is "worker"
   * for everything except "low" in mono, and "high" falls back to "medium"
   * with a warning. */
  thread?: "audio" | "worker";
  /** Max channels processed: 1 (default) downmixes stereo input to mono;
   * 2 runs one engine per channel. */
  maxChannels?: 1 | 2;
  /** Gate thresholds; only used when quality is "gate". */
  gate?: GateOptions;
  /** Initial DSP state (default true). Equivalent to a later setEnabled. */
  enabled?: boolean;
  /** Self-hosted asset URLs. Defaults resolve next to this module (worklet.js, worker.js, wasm/).
   * `wasm` overrides the model binary outright; `scalarWasm` is used only when WASM SIMD is unavailable. */
  urls?: { worklet?: string; worker?: string; wasm?: string; scalarWasm?: string };
  /** Worker-mode jitter-buffer stats, ~once per second of audio when they change. */
  onStats?: (s: FilterStats) => void;
}

export interface NoiseFilter {
  /** The worklet node. Wire it: `source.connect(f.node); f.node.connect(dest)`. */
  readonly node: AudioWorkletNode;
  readonly quality: Quality;
  readonly thread: "audio" | "worker";
  /** Initial output latency in seconds: 896/48000 audio, 1408/48000 worker
   * mono, 1920/48000 worker stereo, 0 gate. */
  readonly latency: number;
  readonly enabled: boolean;
  /** Copy-through bypass toggle; never rebuilds the graph. */
  setEnabled(enabled: boolean): void;
  /** Posts destroy to the worklet, disconnects the node, terminates the
   * workers, closes the channel ports. Idempotent. */
  destroy(): void;
}

const CJS_ASSET_ERROR =
  "[open-noise-filter] can't locate bundled assets (CommonJS build): " +
  "pass options.urls.worklet/wasm (and worker for worker mode)";

// literal new URL() expressions — bundlers can't trace a template; the CJS
// build has no import.meta.url and takes the throw (options.urls required).
function defaultUrls() {
  try {
    return {
      worklet: new URL("./worklet.js", import.meta.url).href,
      worker: new URL("./worker.js", import.meta.url).href,
      wasm: {
        low: new URL("./wasm/fastenhancer_t.wasm", import.meta.url).href,
        medium: new URL("./wasm/fastenhancer_b.wasm", import.meta.url).href,
        high: new URL("./wasm/fastenhancer_s.wasm", import.meta.url).href,
      } as Record<WasmQuality, string>,
      wasmScalar: {
        low: new URL("./wasm/fastenhancer_t_scalar.wasm", import.meta.url).href,
        medium: new URL("./wasm/fastenhancer_b_scalar.wasm", import.meta.url).href,
        high: new URL("./wasm/fastenhancer_s_scalar.wasm", import.meta.url).href,
      } as Record<WasmQuality, string>,
    };
  } catch {
    throw new Error(CJS_ASSET_ERROR);
  }
}

/**
 * Scalar (non-SIMD) thread policy. Measured frame cost (tools repo
 * cost.mjs): low p99 1.97 ms, medium p99 5.98 ms, high median 11.6 ms vs a
 * 2.67 ms quantum / 10.67 ms hop — audio thread only fits low/mono, and
 * high can't keep up even in a worker (createNoiseFilter downgrades it).
 */
function defaultThread(q: Quality, channels: 1 | 2, simd: boolean): "audio" | "worker" {
  if (!simd && q !== "gate") {
    return q === "low" && channels === 1 ? "audio" : "worker";
  }
  return q === "medium" || q === "high" ? "worker" : "audio";
}

export function isNoiseFilterSupported(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof AudioWorkletNode !== "undefined"
  );
}

export async function createNoiseFilter(
  ctx: BaseAudioContext,
  options: NoiseFilterOptions = {},
): Promise<NoiseFilter> {
  let quality = options.quality ?? "medium";
  const maxChannels = options.maxChannels === 2 ? 2 : 1;
  const enabled0 = options.enabled ?? true;
  const simd = isSimdSupported();
  const offline =
    typeof OfflineAudioContext !== "undefined" && ctx instanceof OfflineAudioContext;
  if (quality === "high" && !simd) {
    // scalar Small costs ~11.6 ms/frame median — more than a 10.67 ms hop,
    // so it can't keep up even in a worker (see defaultThread)
    console.warn(
      "[open-noise-filter] WebAssembly SIMD is unavailable and 'high' can't " +
      "keep up as a scalar build — falling back to 'medium'.",
    );
    quality = "medium";
  }
  let thread: "audio" | "worker" =
    options.thread ?? defaultThread(quality, maxChannels, simd);
  if (quality === "gate") {
    if (options.thread === "worker") {
      console.warn("[open-noise-filter] the gate tier always runs on the audio thread.");
    }
    thread = "audio";
  }
  if (offline && thread === "worker") {
    console.warn(
      "[open-noise-filter] worker mode can't keep up with OfflineAudioContext's " +
      "faster-than-realtime rendering — running on the audio thread instead.",
    );
    thread = "audio";
  }

  if (quality !== "gate" && ctx.sampleRate !== 48000) {
    console.warn(
      `[open-noise-filter] AudioContext is ${ctx.sampleRate} Hz, not 48000 — ` +
      "audio is resampled inside the worklet. For best quality, use a 48 kHz AudioContext.",
    );
  }

  const workletUrl = options.urls?.worklet ?? defaultUrls().worklet;
  await ctx.audioWorklet.addModule(workletUrl);

  const nodeOpts: AudioWorkletNodeOptions = {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    channelCount: maxChannels,
    channelCountMode: "explicit",
    channelInterpretation: "speakers",
    outputChannelCount: [maxChannels],
  };

  const workers: Worker[] = [];
  const channels: MessageChannel[] = [];
  let node: AudioWorkletNode | undefined;
  try {
    if (quality === "gate") {
      node = new AudioWorkletNode(ctx, "fastenhancer-worklet", {
        ...nodeOpts,
        processorOptions: {
          mode: "gate", channels: maxChannels, enabled: enabled0,
          gate: options.gate,
        },
      });
      await waitForNode(node, options);
    } else {
      // non-SIMD browsers prefer scalarWasm over wasm — a wasm URL that only
      // ships SIMD code would fail to compile there
      const wasmUrl = simd
        ? options.urls?.wasm ?? defaultUrls().wasm[quality as WasmQuality]
        : options.urls?.scalarWasm ??
          options.urls?.wasm ??
          defaultUrls().wasmScalar[quality as WasmQuality];
      const res = await fetch(wasmUrl);
      if (!res.ok) {
        throw new Error(`wasm fetch failed: ${res.status} ${res.statusText} (${wasmUrl})`);
      }
      const wasm = await res.arrayBuffer();

      if (thread === "worker") {
        const workerUrl = options.urls?.worker ?? defaultUrls().worker;
        const workerReady: Promise<void>[] = [];
        for (let c = 0; c < maxChannels; c++) {
          const w = new Worker(workerUrl);
          const ch = new MessageChannel();
          workers.push(w);
          channels.push(ch);
          workerReady.push(new Promise<void>((res2, rej2) => {
            const to = setTimeout(() => rej2(new Error("worker ready timeout")), 10000);
            w.onmessage = (ev: MessageEvent) => {
              const d = ev.data as { type: string; error?: string };
              if (d.type === "ready") { clearTimeout(to); res2(); }
              if (d.type === "error") { clearTimeout(to); rej2(new Error(d.error)); }
            };
          }));
          // transfer a per-channel copy; the last worker gets the original
          const wbuf = c === maxChannels - 1 ? wasm : wasm.slice(0);
          w.postMessage({ type: "init", wasm: wbuf, port: ch.port2 }, [wbuf, ch.port2]);
        }

        node = new AudioWorkletNode(ctx, "fastenhancer-worklet", {
          ...nodeOpts,
          processorOptions: { mode: "worker", channels: maxChannels, enabled: enabled0 },
        });
        node.port.postMessage(
          { type: "ports", ports: channels.map((ch) => ch.port1) },
          channels.map((ch) => ch.port1),
        );
        await Promise.all([...workerReady, waitForNode(node, options)]);
      } else {
        node = new AudioWorkletNode(ctx, "fastenhancer-worklet", {
          ...nodeOpts,
          processorOptions: { wasm, channels: maxChannels, enabled: enabled0 },
        });
        await waitForNode(node, options);
      }
    }
  } catch (e) {
    for (const w of workers) { try { w.terminate(); } catch { /* gone */ } }
    for (const ch of channels) {
      try { ch.port1.close(); ch.port2.close(); } catch { /* gone */ }
    }
    try { node?.disconnect(); } catch { /* not connected */ }
    throw e;
  }

  const n = node;
  let destroyed = false;
  let enabled = enabled0;
  const filter: NoiseFilter = {
    node: n,
    quality,
    thread,
    // worker latency = algo 896 + (initial target − one in-flight hop)
    latency: quality === "gate" ? 0
      : (thread === "worker" ? 384 + (maxChannels === 2 ? 1536 : 1024) : 896) / 48000,
    get enabled() { return enabled; },
    setEnabled(v: boolean) {
      enabled = !!v;
      try { n.port.postMessage({ type: "setEnabled", value: enabled }); } catch { /* gone */ }
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      try { n.port.postMessage({ type: "destroy" }); } catch { /* gone */ }
      for (const w of workers) { try { w.terminate(); } catch { /* gone */ } }
      for (const ch of channels) {
        try { ch.port1.close(); ch.port2.close(); } catch { /* gone */ }
      }
      try { n.disconnect(); } catch { /* not connected */ }
    },
  };
  return filter;
}

function waitForNode(node: AudioWorkletNode, options: NoiseFilterOptions): Promise<void> {
  return new Promise<void>((res2, rej2) => {
    const to = setTimeout(() => rej2(new Error("worklet ready timeout")), 10000);
    node.port.onmessage = (ev: MessageEvent) => {
      const d = ev.data as { type: string; error?: string } & FilterStats;
      if (d.type === "ready") { clearTimeout(to); res2(); }
      if (d.type === "error") { clearTimeout(to); rej2(new Error(d.error)); }
      if (d.type === "stats" && options.onStats) {
        options.onStats({ underruns: d.underruns, overruns: d.overruns, skips: d.skips, target: d.target });
      }
    };
  });
}
