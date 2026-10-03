/**
 * Node harness: loads the esbuild-bundled worklet (and worker) sources via
 * `new Function` so each instance can get its own sampleRate. The fake port
 * captures postMessage and accepts onmessage assignments.
 */
import { readFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { buildSync } from "esbuild";
import { FrameEngine } from "../src/dsp.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = join(HERE, "..", "src");
const WASM_DIR = join(HERE, "..", "wasm");

function bundle(entry) {
  const r = buildSync({
    entryPoints: [join(SRC, entry)],
    bundle: true,
    format: "iife",
    write: false,
  });
  return r.outputFiles[0].text;
}

export function makeProcessor(sampleRate, processorOptions) {
  const src = bundle("worklet.js");
  let Proc;
  class AudioWorkletProcessor {
    constructor() {
      this.port = {
        messages: [],
        postMessage(m) { this.messages.push(m); },
        onmessage: null,
      };
    }
  }
  const registerProcessor = (name, cls) => { Proc = cls; };
  new Function("sampleRate", "AudioWorkletProcessor", "registerProcessor", src)(
    sampleRate, AudioWorkletProcessor, registerProcessor);
  return new Proc({ processorOptions });
}

export function wasmBytes(tier = "b") {
  const b = readFileSync(join(WASM_DIR, `fastenhancer_${tier}.wasm`));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

/** Feed Float32Array `x` through `proc` in 128-sample quanta; returns output. */
export function run(proc, x) {
  const y = new Float32Array(Math.ceil(x.length / 128) * 128);
  const inp = new Float32Array(128), out = new Float32Array(128);
  for (let q = 0; q * 128 < x.length; q++) {
    inp.set(x.subarray(q * 128, q * 128 + 128));
    proc.process([[inp]], [[out]], {});
    y.set(out, q * 128);
  }
  return y;
}

/** Multi-channel variant of run(): xs is an array of per-channel inputs;
 * returns an array of per-channel outputs. `tick` runs after each quantum. */
export function runChannels(proc, xs, tick) {
  const C = xs.length;
  const nq = Math.ceil(xs[0].length / 128);
  const ys = xs.map(() => new Float32Array(nq * 128));
  const ins = xs.map(() => new Float32Array(128)), outs = xs.map(() => new Float32Array(128));
  for (let q = 0; q < nq; q++) {
    for (let c = 0; c < C; c++) ins[c].set(xs[c].subarray(q * 128, q * 128 + 128));
    proc.process([ins], [outs], {});
    for (let c = 0; c < C; c++) ys[c].set(outs[c], q * 128);
    if (tick) tick();
  }
  return ys;
}

export function postMessage(proc, data) {
  proc.port.onmessage && proc.port.onmessage({ data });
}

/**
 * Fake worker for worklet "worker" mode: an in-process FrameEngine behind a
 * fake MessagePort. Hop replies are queued and delivered by tick() — each hop
 * gets a delay cycling 0..maxDelay ticks (0 = delivered by the next tick).
 * Set { auto: false } to never process hops (pool exhaustion tests); replies
 * can still be pushed manually via flush().
 */
export function makeFakeWorker(wasm, { maxDelay = 0, auto = true } = {}) {
  const inst = new WebAssembly.Instance(new WebAssembly.Module(wasm));
  const eng = new FrameEngine(inst.exports);
  let gen = 0, seq = 0;
  const pending = [];
  const resets = [];
  const port = {
    onmessage: null,
    // worklet → worker direction
    postMessage(m) {
      if (m.type === "hop") pending.push({ m, delay: seq++ % (maxDelay + 1), age: 0 });
      else if (m.type === "reset") { gen = m.gen; resets.push(m.gen); eng.reset(); pending.length = 0; }
    },
  };
  const deliver = (p) => {
    const produced = p.m.gen === gen ? eng.processHop(p.m.buf, p.m.buf) : false;
    port.onmessage && port.onmessage({ data: { type: "out", gen: p.m.gen, buf: p.m.buf, produced } });
  };
  return {
    port,
    resets,
    get pending() { return pending.length; },
    tick() {
      if (!auto) return;
      for (let i = pending.length - 1; i >= 0; i--) {
        if (pending[i].age++ >= pending[i].delay) { const p = pending.splice(i, 1)[0]; deliver(p); }
      }
    },
    flush() { while (pending.length) deliver(pending.shift()); },
  };
}

/** Attach fake worker(s) to a worker-mode processor (delivers the ports msg). */
export function attachWorker(proc, workers) {
  const ws = Array.isArray(workers) ? workers : [workers];
  postMessage(proc, { type: "ports", ports: ws.map((w) => w.port) });
}

/**
 * Load the bundled src/worker.js with a shimmed `self`. Returns
 * { self, sent } — call self.onmessage({data:{type:"init",wasm,port}}) with a
 * fake port pair, then await a microtask for the async instantiate.
 */
export function makeWorkerProc() {
  const src = bundle("worker.js");
  const sent = [];
  const self = { postMessage(m) { sent.push(m); }, onmessage: null };
  new Function("self", src)(self);
  return { self, sent };
}

/** A fake MessageChannel: a.postMessage → b.onmessage, and vice versa. */
export function makeChannel() {
  const ch = {
    a: { onmessage: null, postMessage(m) { queueMicrotask(() => ch.b.onmessage && ch.b.onmessage({ data: m })); } },
    b: { onmessage: null, postMessage(m) { queueMicrotask(() => ch.a.onmessage && ch.a.onmessage({ data: m })); } },
  };
  return ch;
}
