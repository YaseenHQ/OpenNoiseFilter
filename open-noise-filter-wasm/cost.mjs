// Copyright 2026 YaseenHQ
// SPDX-License-Identifier: Apache-2.0

/**
 * Per-frame CPU cost for every tier × variant, measured through the library's
 * worklet code (same path the audio thread takes).
 *
 *   node cost.mjs [--variant simd|scalar|all] [--tier t|b|s]
 *
 * Method: 60 s of uniform noise (±0.05, fixed-seed LCG) at 48 kHz through
 * makeProcessor; only process() calls where `produced` advanced are timed
 * (that's where a model frame runs); the first 400 quanta are skipped.
 * Reports median / p99 / max and % of frames over the 2.667 ms quantum budget.
 *
 * Library location: $ONF_DIR (default ../open-noise-filter).
 */
import { readFileSync } from "fs";
import { join, dirname, resolve } from "path";
import { fileURLToPath } from "url";
import { makeProcessor } from "./harness.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const argList = (flag) => {
  const out = [];
  for (let i = 0; i < process.argv.length; i++) {
    if (process.argv[i] === flag) out.push(process.argv[++i]);
    else if (process.argv[i].startsWith(flag + "=")) out.push(process.argv[i].slice(flag.length + 1));
  }
  return out;
};
const tiers = argList("--tier").length ? argList("--tier") : ["t", "b", "s"];
const variants = argList("--variant").length
  ? argList("--variant").flatMap((v) => (v === "all" ? ["simd", "scalar"] : [v]))
  : ["simd", "scalar"];

const QUANTUM_BUDGET_MS = 128 / 48; // 2.667 ms
console.log("tier  variant  median ms  p99 ms  max ms  >2.667ms");
for (const tier of tiers) {
  for (const variant of variants) {
    const file = join(HERE, "out", `fastenhancer_${tier}${variant === "scalar" ? "_scalar" : ""}.wasm`);
    const wasm = readFileSync(file).buffer.slice(0);
    const p = makeProcessor(48000, { wasm });
    const inp = new Float32Array(128), out = new Float32Array(128);
    const times = [];
    let seed = 1;
    const nq = Math.ceil(48000 * 60 / 128) + 400;
    for (let q = 0; q < nq; q++) {
      for (let i = 0; i < 128; i++) { seed = (seed * 1103515245 + 12345) & 0x7fffffff; inp[i] = (seed / 0x40000000 - 1) * 0.05; }
      const prev = p.produced;
      const t0 = performance.now();
      p.process([[inp]], [[out]], {});
      const dt = performance.now() - t0;
      if (q >= 400 && p.produced !== prev) times.push(dt);
    }
    times.sort((a, b) => a - b);
    const med = times[Math.floor(times.length / 2)];
    const p99 = times[Math.floor(times.length * 0.99)];
    const mx = times[times.length - 1];
    const over = (100 * times.filter((t) => t > QUANTUM_BUDGET_MS).length) / times.length;
    console.log(`${tier}     ${variant.padEnd(7)}  ${med.toFixed(3).padStart(8)}  ${p99.toFixed(3).padStart(6)}  ${mx.toFixed(3).padStart(6)}  ${over.toFixed(1)}%  (n=${times.length})`);
  }
}
