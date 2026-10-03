// Copyright 2026 YaseenHQ
// SPDX-License-Identifier: Apache-2.0

/**
 * open-noise-filter/compat — an alternate node-class API for callers that
 * prefer managing the worklet module and wasm binary themselves:
 *
 *   await ctx.audioWorklet.addModule("open-noise-filter/worklet.js");
 *   const wasmBinary = await loadNoiseFilter({ url: ".../fastenhancer_b.wasm" });
 *   const node = new NoiseFilterWorkletNode(ctx, { wasmBinary });
 *   source.connect(node); node.connect(ctx.destination);
 *   node.setEnabled(false);
 *
 * These nodes run on the audio thread only — for "high", or "medium" in
 * stereo, use createNoiseFilter() instead so the DSP runs in a Worker.
 */

import { isSimdSupported } from "./simd.js";

export { isSimdSupported };

/** Fetch a wasm binary; falls back to `scalarUrl` when WASM SIMD is missing. */
export async function loadNoiseFilter(opts: {
  url: string;
  scalarUrl?: string;
}): Promise<ArrayBuffer> {
  let url = opts.url;
  if (!isSimdSupported()) {
    if (!opts.scalarUrl) {
      throw new Error(
        "[open-noise-filter] WebAssembly SIMD is not supported in this " +
        "browser and no scalarUrl fallback was given",
      );
    }
    url = opts.scalarUrl;
  }
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`wasm fetch failed: ${res.status} ${res.statusText} (${url})`);
  }
  return res.arrayBuffer();
}

export interface CompatWorkletNode extends AudioWorkletNode {
  setEnabled(b: boolean): void;
  destroy(): void;
}

const nodeOptions = (maxChannels: 1 | 2, processorOptions: object): AudioWorkletNodeOptions => ({
  numberOfInputs: 1,
  numberOfOutputs: 1,
  channelCount: maxChannels,
  channelCountMode: "explicit",
  channelInterpretation: "speakers",
  outputChannelCount: [maxChannels],
  processorOptions,
});

// AudioWorkletNode resolved at construction, not module load: SSR import
// must not throw, and a later-installed global still yields a working class.
function workletNodeBase(): typeof AudioWorkletNode {
  const B = (globalThis as { AudioWorkletNode?: typeof AudioWorkletNode }).AudioWorkletNode;
  if (!B) {
    throw new Error(
      "[open-noise-filter] AudioWorklet is not available in this environment",
    );
  }
  return B;
}

function withCompatMethods(B: typeof AudioWorkletNode) {
  return class extends B {
    setEnabled(b: boolean): void {
      this.port.postMessage({ type: "setEnabled", value: !!b });
    }
    destroy(): void {
      try { this.port.postMessage({ type: "destroy" }); } catch { /* gone */ }
      try { this.disconnect(); } catch { /* not connected */ }
    }
  };
}

let FastEnhancerNode: (new (
  context: BaseAudioContext,
  options: { wasmBinary: ArrayBuffer; maxChannels?: 1 | 2; enabled?: boolean },
) => CompatWorkletNode) | undefined;

let GateNode: (new (
  context: BaseAudioContext,
  options?: {
    openThreshold?: number; closeThreshold?: number; holdMs?: number;
    maxChannels?: 1 | 2;
  },
) => CompatWorkletNode) | undefined;

export const NoiseFilterWorkletNode: new (
  context: BaseAudioContext,
  options: { wasmBinary: ArrayBuffer; maxChannels?: 1 | 2; enabled?: boolean },
) => CompatWorkletNode = class {
  // the real class only exists after first construction
  static [Symbol.hasInstance](inst: unknown): boolean {
    return FastEnhancerNode !== undefined && inst instanceof FastEnhancerNode;
  }
  constructor(
    context: BaseAudioContext,
    options: { wasmBinary: ArrayBuffer; maxChannels?: 1 | 2; enabled?: boolean },
  ) {
    if (!FastEnhancerNode) {
      const B = workletNodeBase();
      FastEnhancerNode = class extends withCompatMethods(B) {
        constructor(
          ctx: BaseAudioContext,
          o: { wasmBinary: ArrayBuffer; maxChannels?: 1 | 2; enabled?: boolean },
        ) {
          const channels = o.maxChannels === 2 ? 2 : 1;
          super(ctx, "fastenhancer-worklet", nodeOptions(channels, {
            wasm: o.wasmBinary, channels, enabled: o.enabled !== false,
          }));
        }
      } as never;
    }
    return new FastEnhancerNode(context, options) as never;
  }
} as never;

export const NoiseGateWorkletNode: new (
  context: BaseAudioContext,
  options?: {
    openThreshold?: number; closeThreshold?: number; holdMs?: number;
    maxChannels?: 1 | 2;
  },
) => CompatWorkletNode = class {
  static [Symbol.hasInstance](inst: unknown): boolean {
    // the real class only exists after first construction
    return GateNode !== undefined && inst instanceof GateNode;
  }
  constructor(
    context: BaseAudioContext,
    options: {
      openThreshold?: number; closeThreshold?: number; holdMs?: number;
      maxChannels?: 1 | 2;
    } = {},
  ) {
    if (!GateNode) {
      const B = workletNodeBase();
      GateNode = class extends withCompatMethods(B) {
        constructor(
          ctx: BaseAudioContext,
          o: {
            openThreshold?: number; closeThreshold?: number; holdMs?: number;
            maxChannels?: 1 | 2;
          } = {},
        ) {
          const channels = o.maxChannels === 2 ? 2 : 1;
          super(ctx, "fastenhancer-worklet", nodeOptions(channels, {
            mode: "gate", channels,
            gate: {
              openThreshold: o.openThreshold,
              closeThreshold: o.closeThreshold,
              holdMs: o.holdMs,
            },
          }));
        }
      } as never;
    }
    return new GateNode(context, options) as never;
  }
} as never;
