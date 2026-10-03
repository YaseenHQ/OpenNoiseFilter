/**
 * Reproducible FastEnhancer → WASM pipeline.
 *
 *   node build.mjs [--tier t|b|s ...] [--variant simd|scalar|all]
 *   node build.mjs --model path/to/model_spec.onnx --name mymodel --shim b
 *
 * Custom models: --model builds an arbitrary FastEnhancer-compatible spec
 * ONNX (same io contract: spec_in [1,513,1,2] + cache_in/cache_out tensors).
 * --shim picks the C shim whose cache geometry matches the checkpoint
 * (t: 2 caches 24×20, b: 3 caches 36×36, s: 3 caches 48×48 — see shims/).
 * Output lands at
 * out/<name>.wasm (+ out/<name>_scalar.wasm).
 *
 * Per tier: lower_gru.py (GRU T=1 → basic ops, Gather → Slice+Squeeze,
 * ORT-validated) → normalize.py (cache-outs-first output order, required
 * for onnx2c to compile exact) → onnx2c → dealias.sh (scratch union →
 * struct, mandatory — onnx2c's lifetime analysis clobbers live data on
 * this graph) → fastmatmul.py (ikj matmul rewrite) → emcc + per-tier shim.
 *
 * Tool paths are configurable:
 *   ONNX2C   path to the onnx2c binary; on Windows, Unix-style paths run
 *            through `wsl --exec`. On Unix they run natively.
 *   EMCC     emcc executable (default "emcc" on PATH)
 *   PYTHON   python with numpy + onnx + onnxruntime (default: repo .venv,
 *            then python3/python on PATH)
 */
import { execFileSync } from "child_process";
import { existsSync, mkdirSync, copyFileSync, statSync, writeFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { normalizeOnnx2cPath, onnx2cInvocation } from "./onnx2c-command.mjs";

const ROOT = dirname(fileURLToPath(import.meta.url));
const BUILD = join(ROOT, "build");
const OUT = join(ROOT, "out");
mkdirSync(BUILD, { recursive: true });
mkdirSync(OUT, { recursive: true });

const TIERS = ["t", "b", "s"];
const argList = (flag) => {
  const out = [];
  for (let i = 0; i < process.argv.length; i++) {
    if (process.argv[i] === flag) out.push(process.argv[++i]);
    else if (process.argv[i].startsWith(flag + "=")) out.push(process.argv[i].slice(flag.length + 1));
  }
  return out;
};
const tiers = argList("--tier").length ? argList("--tier") : TIERS;
const variants = argList("--variant").length
  ? argList("--variant").flatMap((v) => (v === "all" ? ["simd", "scalar"] : [v]))
  : ["simd", "scalar"];
// --model/--name/--shim: build a custom FastEnhancer-compatible checkpoint
const customModel = argList("--model")[0] ?? null;
const customName = argList("--name")[0] ?? (customModel ? "custom" : null);
const customShim = argList("--shim")[0] ?? null;
if (customModel && !TIERS.includes(customShim)) {
  console.error("--model requires --shim t|b|s (the cache geometry must match the checkpoint)");
  process.exit(1);
}
const jobs = customModel
  ? [{ spec: customModel, name: customName, shim: customShim }]
  : tiers.map((t) => ({ spec: join(ROOT, "models", `fastenhancer_${t}_spec.onnx`), name: `fastenhancer_${t}`, shim: t }));

const ONNX2C = normalizeOnnx2cPath(process.env.ONNX2C ?? "onnx2c");
// MSYS/Git Bash rewrites "/mnt/..." env values to "C:/Program Files/Git/mnt/..."
const EMCC = process.env.EMCC ?? "emcc";
const PYTHON = process.env.PYTHON ?? firstExisting([
  join(ROOT, ".venv", "Scripts", "python.exe"),
  join(ROOT, ".venv", "bin", "python"),
  "python3",
  "python",
]);
function firstExisting(candidates) {
  for (const c of candidates) {
    if (c.includes("/") || c.includes("\\")) { if (existsSync(c)) return c; continue; }
    try { execFileSync(c, ["--version"], { stdio: "pipe" }); return c; } catch { /* next */ }
  }
  return candidates.at(-1);
}

function run(cmd, args, { wslPaths = false, captureTo = null } = {}) {
  const invocation = wslPaths
    ? onnx2cInvocation(cmd, args, process.platform, { wslPaths })
    : { command: cmd, args };
  const stdio = captureTo ? ["inherit", "pipe", "inherit"] : "inherit";
  console.log(`$ ${invocation.command} ${invocation.args.join(" ")}${captureTo ? " > " + captureTo : ""}`);
  const res = execFileSync(invocation.command, invocation.args, { stdio, maxBuffer: 256 << 20 });
  if (captureTo) writeFileSync(captureTo, res);
}

// emcc flags reconstructed from the shipped binaries' export surface
// (memory + fe_* + _initialize + stack helpers, no imports, STANDALONE_WASM).
const EMCC_FLAGS = [
  "-O3", "-ffast-math",
  "-sSTANDALONE_WASM", "--no-entry",
  "-Wl,--export=fe_init", "-Wl,--export=fe_run",
  "-Wl,--export=fe_in_ptr", "-Wl,--export=fe_out_ptr",
];

for (const job of jobs) {
  const spec = join(BUILD, `${job.name}_spec.onnx`);
  copyFileSync(job.spec, spec);
  run(PYTHON, [join(ROOT, "scripts", "lower_gru.py"), spec]);
  const lowered = spec.replace(".onnx", "_lowered.onnx");
  const norm = join(BUILD, `${job.name}_norm.onnx`);
  run(PYTHON, [join(ROOT, "scripts", "normalize.py"), lowered, norm]);
  const cFile = norm.replace(/\.onnx$/, ".c");
  run(ONNX2C, [norm], { wslPaths: true, captureTo: cFile }); // onnx2c → stdout
  if (!existsSync(cFile)) throw new Error(`onnx2c did not produce ${cFile}`);
  run("sh", [join(ROOT, "scripts", "dealias.sh"), cFile]);
  run(PYTHON, [join(ROOT, "scripts", "fastmatmul.py"), cFile]);

  for (const variant of variants) {
    const simd = variant === "simd";
    const out = join(OUT, `${job.name}${simd ? "" : "_scalar"}.wasm`);
    run(EMCC, [
      ...(simd ? ["-msimd128"] : []),
      ...EMCC_FLAGS,
      cFile, join(ROOT, "shims", `shim_${job.shim}.c`),
      "-o", out,
    ]);
    console.log(`→ ${out} (${(statSync(out).size / 1024).toFixed(0)} KB)`);
  }
}
console.log("done — run: node verify.mjs --variant simd|scalar");
