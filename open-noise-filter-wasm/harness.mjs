// Copyright 2026 YaseenHQ
// SPDX-License-Identifier: Apache-2.0

/**
 * Loads the library's AudioWorkletProcessor via `new Function` so a built wasm
 * can be driven frame-by-frame in Node. The library repo is located via
 * $ONF_DIR (default: ../open-noise-filter).
 */
import { join, dirname, resolve } from "path";
import { fileURLToPath } from "url";
import { buildSync } from "esbuild";

const HERE = dirname(fileURLToPath(import.meta.url));
const ONF = resolve(process.env.ONF_DIR ?? join(HERE, "..", "open-noise-filter"));

export function makeProcessor(sampleRate, processorOptions) {
  const r = buildSync({
    entryPoints: [join(ONF, "src", "worklet.js")],
    bundle: true,
    format: "iife",
    write: false,
  });
  const src = r.outputFiles[0].text;
  let Proc;
  class AudioWorkletProcessor {
    constructor() {
      this.port = { messages: [], postMessage(m) { this.messages.push(m); }, onmessage: null };
    }
  }
  const registerProcessor = (name, cls) => { Proc = cls; };
  new Function("sampleRate", "AudioWorkletProcessor", "registerProcessor", src)(
    sampleRate, AudioWorkletProcessor, registerProcessor);
  return new Proc({ processorOptions });
}
