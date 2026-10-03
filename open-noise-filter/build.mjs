// Copyright 2026 YaseenHQ
// SPDX-License-Identifier: Apache-2.0

import { build } from "esbuild";
import { cpSync, mkdirSync, rmSync } from "fs";
import { execFileSync } from "child_process";

rmSync("dist", { recursive: true, force: true });
mkdirSync("dist/wasm", { recursive: true });
const entries = { index: "src/index.ts", livekit: "src/livekit.ts", compat: "src/compat.ts" };
// library entries: esm + code splitting (shared core lands in a chunk whose
// relative import.meta.url still resolves dist/worklet.js and dist/wasm/)
await build({
  entryPoints: entries,
  bundle: true,
  format: "esm",
  splitting: true,
  platform: "browser",
  outdir: "dist",
  outExtension: { ".js": ".mjs" },
  external: ["livekit-client"],
});
// CommonJS builds: no splitting (duplicated core code is fine), import.meta
// resolves to an empty object — options.urls is required there.
for (const [name, entry] of Object.entries(entries)) {
  await build({
    entryPoints: [entry],
    bundle: true,
    format: "cjs",
    platform: "browser",
    outfile: `dist/${name}.cjs`,
    external: ["livekit-client"],
    logOverride: { "empty-import-meta": "silent" },
  });
}
// worklet/worker must be self-contained (bundlers copy them as assets)
for (const name of ["worklet", "worker"]) {
  await build({
    entryPoints: [`src/${name}.js`],
    bundle: true,
    format: "iife",
    platform: "browser",
    outfile: `dist/${name}.js`,
  });
}
execFileSync(process.execPath, ["node_modules/typescript/bin/tsc", "-p", "tsconfig.build.json"], { stdio: "inherit" });
for (const t of ["t", "b", "s"]) {
  cpSync(`wasm/fastenhancer_${t}.wasm`, `dist/wasm/fastenhancer_${t}.wasm`);
  cpSync(`wasm/fastenhancer_${t}_scalar.wasm`, `dist/wasm/fastenhancer_${t}_scalar.wasm`);
}
console.log("built dist/");
