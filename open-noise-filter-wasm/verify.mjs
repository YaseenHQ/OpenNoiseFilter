// Copyright 2026 YaseenHQ
// SPDX-License-Identifier: Apache-2.0

/**
 * Full equivalence check, all tiers: ONNX Runtime reference → single-frame
 * spectrum diff → whole-clip audio diff, against the built WASM in out/.
 *
 *   node verify.mjs [--variant simd|scalar] [--tier t|b|s ...]
 *
 * Python resolution order: $PYTHON, .venv/Scripts/python.exe,
 * .venv/bin/python, then python3/python on PATH.
 */
import { execFileSync } from "child_process";
import { existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const ROOT = dirname(fileURLToPath(import.meta.url));
process.chdir(ROOT);

const argList = (flag) => {
  const out = [];
  for (let i = 0; i < process.argv.length; i++) {
    if (process.argv[i] === flag) out.push(process.argv[++i]);
    else if (process.argv[i].startsWith(flag + "=")) out.push(process.argv[i].slice(flag.length + 1));
  }
  return out;
};
const tiers = argList("--tier").length ? argList("--tier") : ["t", "b", "s"];
const variant = (argList("--variant")[0] ?? "simd");
const wasmFor = (t) => join(ROOT, "out", `fastenhancer_${t}${variant === "scalar" ? "_scalar" : ""}.wasm`);

const PYTHON_CANDIDATES = [
  process.env.PYTHON,
  join(ROOT, ".venv", "Scripts", "python.exe"),
  join(ROOT, ".venv", "bin", "python"),
  "python3",
  "python",
].filter(Boolean);

let py = null;
for (const c of PYTHON_CANDIDATES) {
  try {
    execFileSync(c, ["-c", "import numpy, onnxruntime"], { stdio: "pipe" });
    py = c;
    break;
  } catch { /* try next */ }
}
if (!py) {
  console.error("no working python found (need numpy + onnxruntime).");
  console.error("tried:", PYTHON_CANDIDATES.join(", "));
  console.error("set $PYTHON, or: python -m venv .venv && .venv/Scripts/pip install -r scripts/requirements.txt");
  process.exit(1);
}
console.log(`python: ${py}  variant: ${variant}`);

for (const t of tiers) {
  if (!existsSync(wasmFor(t))) {
    console.error(`missing ${wasmFor(t)} — run node build.mjs --tier ${t} --variant ${variant}`);
    process.exit(1);
  }
}

const results = [];
for (const tier of tiers) {
  const env = { ...process.env, FE_WASM: wasmFor(tier) };
  const steps = [
    [py, ["scripts/reference.py", tier], "reference.py"],
    [process.execPath, ["scripts/specdiff.mjs", tier], "specdiff"],
    [process.execPath, ["scripts/equiv.mjs", tier], "equiv"],
  ];
  for (const [cmd, args, name] of steps) {
    try {
      execFileSync(cmd, args, { stdio: "inherit", env });
      results.push(`${tier}/${name}: PASS`);
    } catch {
      results.push(`${tier}/${name}: FAIL`);
      console.log("\n=== verify summary ===");
      for (const r of results) console.log(" ", r);
      process.exit(1);
    }
  }
}
console.log("\n=== verify summary ===");
for (const r of results) console.log(" ", r);
console.log(`all tiers verified (${variant})`);
