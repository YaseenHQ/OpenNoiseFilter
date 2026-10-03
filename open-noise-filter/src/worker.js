// Copyright 2026 YaseenHQ
// SPDX-License-Identifier: Apache-2.0

/**
 * open-noise-filter — worker-side FastEnhancer engine.
 *
 * Receives {type:"init", wasm, port} on the worker global scope; compiles and
 * instantiates the tier wasm here (async is fine — no audio-thread deadline),
 * then serves {type:"hop", gen, buf} messages on the transferred port:
 * each hop is processed in place by the shared FrameEngine and replied as
 * {type:"out", gen, buf, produced} with the buffer transferred back.
 * {type:"reset", gen} clears DSP state and adopts a new generation; hops with
 * a stale gen are dropped. Replies {type:"ready"} or {type:"error"}.
 */
import { FrameEngine } from "./dsp.js";

let eng = null;
let gen = 0;

self.onmessage = async (ev) => {
  const d = ev.data;
  if (!d || d.type !== "init") return;
  try {
    const { instance } = await WebAssembly.instantiate(d.wasm, {});
    eng = new FrameEngine(instance.exports);
    // warm up: first fe_run pays lazy wasm compilation (~100 ms) — it must
    // not land on a live hop
    const z = new Float32Array(512), zo = new Float32Array(512);
    for (let i = 0; i < 16; i++) eng.processHop(z, zo);
    eng.reset();
    const port = d.port;
    port.onmessage = (e2) => {
      const m = e2.data;
      if (!m) return;
      if (m.type === "reset") {
        gen = m.gen;
        eng.reset();
      } else if (m.type === "hop") {
        if (m.gen !== gen) return; // stale generation
        const produced = eng.processHop(m.buf, m.buf);
        port.postMessage({ type: "out", gen, buf: m.buf, produced }, [m.buf.buffer]);
      }
    };
    self.postMessage({ type: "ready" });
  } catch (err) {
    self.postMessage({ type: "error", error: String(err) });
  }
};
