/**
 * Benchmark renderer: runs every corpus clip through every engine's REAL
 * worklet code in Node (same shim approach as test/harness.mjs).
 *
 *   node render.mjs [engineId ...]   (default: all)
 *
 * Writes bench/out/<engine>/<clip>.wav (16-bit PCM 48 kHz) + bench/out/rtf.json
 */
import { readFileSync, writeFileSync, readdirSync, mkdirSync, existsSync } from "fs";
import { join, dirname, resolve } from "path";
import { fileURLToPath } from "url";
import { performance } from "perf_hooks";
import { makeProcessor as harnessProcessor } from "../harness.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const PKG = join(HERE, "..");
const CORPUS = join(HERE, "corpus");
const OUT = join(HERE, "out");
const SR = 48000, QUANTUM = 128;

globalThis.sampleRate = SR;
globalThis.AudioWorkletProcessor = class {
  constructor() {
    this.port = {
      postMessage() {},
      onmessage: null,
      addEventListener(type, fn) { if (type === "message") this._onmsg = fn; },
      removeEventListener() {},
    };
  }
};
globalThis.registerProcessor = (name, cls) => { globalThis.__lastProc = cls; };

const COMPARISON_DIR = process.env.COMPARISON_DIR ? resolve(process.env.COMPARISON_DIR) : null;
const fe = (tier) => ({
  harness: true, // loaded via test/harness.mjs (fresh scope per instance)
  opts: () => ({ wasm: readFileSync(join(PKG, `out/fastenhancer_${tier}.wasm`)).buffer.slice(0) }),
  wasmBytes: readFileSync(join(PKG, `out/fastenhancer_${tier}.wasm`)).length,
});
const externalWasmEngine = (name, wasmFile) => ({
  file: join(COMPARISON_DIR, name, "workletProcessor.js"),
  opts: () => ({ wasmBinary: readFileSync(join(COMPARISON_DIR, wasmFile)).buffer.slice(0), maxChannels: 1 }),
  wasmBytes: readFileSync(join(COMPARISON_DIR, wasmFile)).length,
});

const ENGINES = {
  fe_t: fe("t"),
  fe_b: fe("b"),
  fe_s: fe("s"),
  passthrough: null,
};
if (COMPARISON_DIR) {
  Object.assign(ENGINES, {
    rnnoise: externalWasmEngine("rnnoise", "rnnoise_simd.wasm"),
    gtcrn: externalWasmEngine("gtcrn", "gtcrn.wasm"),
    speex: externalWasmEngine("speex", "speex.wasm"),
    gate: {
      file: join(COMPARISON_DIR, "noiseGate", "workletProcessor.js"),
      opts: () => ({ openThreshold: -50, closeThreshold: -60, holdMs: 120, maxChannels: 1 }),
      wasmBytes: 0,
    },
  });
}

function readWav(path) {
  const b = readFileSync(path);
  const dataOff = b.indexOf("data", 12);
  const n = b.readUInt32LE(dataOff + 4) / 2;
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = b.readInt16LE(dataOff + 8 + i * 2) / 32768;
  return x;
}

function writeWav(path, x) {
  const buf = Buffer.alloc(44 + x.length * 2);
  buf.write("RIFF", 0); buf.writeUInt32LE(36 + x.length * 2, 4); buf.write("WAVE", 8);
  buf.write("fmt ", 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(SR, 24); buf.writeUInt32LE(SR * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34);
  buf.write("data", 36); buf.writeUInt32LE(x.length * 2, 40);
  for (let i = 0; i < x.length; i++) {
    const v = Math.max(-1, Math.min(1, x[i]));
    buf.writeInt16LE(Math.round(v * 32767), 44 + i * 2);
  }
  writeFileSync(path, buf);
}

function makeProcessor(engine) {
  if (!engine) return null; // passthrough
  if (engine.harness) return harnessProcessor(SR, engine.opts());
  let src = readFileSync(engine.file, "utf8");
  // vendored worklets carry emscripten's import.meta.url (wasm auto-locate);
  // we pass wasmBinary explicitly so a harmless literal suffices
  src = src.replace(/import\.meta\.url/g, JSON.stringify("file:///" + engine.file.replace(/\\/g, "/")));
  globalThis.__dirname = dirname(engine.file);
  globalThis.__filename = engine.file;
  (0, eval)(src);
  const Proc = globalThis.__lastProc;
  globalThis.__lastProc = undefined;
  return new Proc({ processorOptions: engine.opts() });
}

async function waitReady(proc, engine) {
  if (!proc) return;
  // Some WASM engines initialize asynchronously; poll, then warm up with silence.
  for (let i = 0; i < 600 && !proc.processor && !proc.ready; i++) {
    await new Promise((r) => setTimeout(r, 5));
  }
  const inp = new Float32Array(QUANTUM), out = new Float32Array(QUANTUM);
  for (let i = 0; i < 40; i++) proc.process([[inp]], [[out]], {}); // ~107 ms warmup
}

function run(proc, x) {
  const y = new Float32Array(x.length);
  const inp = new Float32Array(QUANTUM), out = new Float32Array(QUANTUM);
  const nq = Math.floor(x.length / QUANTUM);
  const t0 = performance.now();
  for (let q = 0; q < nq; q++) {
    inp.set(x.subarray(q * QUANTUM, q * QUANTUM + QUANTUM));
    proc ? proc.process([[inp]], [[out]], {}) : out.set(inp);
    y.set(out, q * QUANTUM);
  }
  const ms = performance.now() - t0;
  return { y, rtf: ms / (nq * QUANTUM / SR * 1000) };
}

const only = process.argv.slice(2);
const clips = readdirSync(CORPUS).filter((f) => f.endsWith("_noisy.wav")).map((f) => f.replace(/_noisy\.wav$/, ""));
const ids = only.length ? only : Object.keys(ENGINES);
const unavailable = ids.filter((id) => !Object.hasOwn(ENGINES, id));
if (unavailable.length) {
  const externalIds = new Set(["rnnoise", "gtcrn", "speex", "gate"]);
  const missingExternal = unavailable.filter((id) => externalIds.has(id));
  if (missingExternal.length && !COMPARISON_DIR) {
    throw new Error(`Comparison engine(s) ${missingExternal.join(", ")} require COMPARISON_DIR; see bench/README.md.`);
  }
  throw new Error(`Unknown engine id(s): ${unavailable.join(", ")}`);
}
const rtf = {};

for (const id of ids) {
  const engine = ENGINES[id];
  const dir = join(OUT, id);
  mkdirSync(dir, { recursive: true });
  const t0 = performance.now();
  let sumRtf = 0;
  for (const cid of clips) {
    const outPath = join(dir, `${cid}.wav`);
    const x = readWav(join(CORPUS, `${cid}_noisy.wav`));
    const proc = makeProcessor(engine);
    await waitReady(proc, engine);
    const { y, rtf: r } = run(proc, x);
    writeWav(outPath, y);
    sumRtf += r;
    proc?.destroy?.();
  }
  rtf[id] = { rtf: sumRtf / clips.length, wasmBytes: engine?.wasmBytes ?? 0, wallS: (performance.now() - t0) / 1000 };
  console.log(`${id}: rtf ${rtf[id].rtf.toFixed(3)}  wasm ${(((engine?.wasmBytes) ?? 0) / 1024).toFixed(0)} KB  wall ${rtf[id].wallS.toFixed(1)}s`);
}
writeFileSync(join(OUT, "rtf.json"), JSON.stringify(rtf, null, 2));
