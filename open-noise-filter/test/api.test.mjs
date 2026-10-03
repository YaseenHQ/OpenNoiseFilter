// Copyright 2026 YaseenHQ
// SPDX-License-Identifier: Apache-2.0

import { test, before } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "module";
import { mkdtempSync } from "fs";
import { join, dirname } from "path";
import { tmpdir } from "os";
import { fileURLToPath, pathToFileURL } from "url";
import { buildSync } from "esbuild";
import { installFakes } from "./fakes.mjs";
import { wasmBytes } from "./harness.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = join(HERE, "..", "src");
const WASM_B = wasmBytes("b");

// bundle the library entries into a temp dir and import them — tests the real
// esm output shape (import.meta.url assets resolve relative to the bundle)
let tmp;
let api, lk;
before(() => {
  tmp = mkdtempSync(join(tmpdir(), "onf-"));
  const r = buildSync({
    entryPoints: { index: join(SRC, "index.ts"), livekit: join(SRC, "livekit.ts"), compat: join(SRC, "compat.ts") },
    bundle: true, format: "esm", splitting: true, platform: "browser",
    outdir: tmp, outExtension: { ".js": ".mjs" }, external: ["livekit-client"],
    write: true, logLevel: "silent",
  });
  if (r.errors.length) throw new Error(JSON.stringify(r.errors));
  const c = buildSync({
    entryPoints: { index: join(SRC, "index.ts"), compat: join(SRC, "compat.ts") },
    bundle: true, format: "cjs", platform: "browser",
    outdir: tmp, outExtension: { ".js": ".cjs" }, external: ["livekit-client"],
    write: true, logLevel: "silent", logOverride: { "empty-import-meta": "silent" },
  });
  if (c.errors.length) throw new Error(JSON.stringify(c.errors));
});
async function loadApi() {
  api ??= await import(pathToFileURL(join(tmp, "index.mjs")).href);
  return api;
}
async function loadLivekit() {
  lk ??= await import(pathToFileURL(join(tmp, "livekit.mjs")).href);
  return lk;
}

test("SSR safety: importing the bundle touches no browser globals", async () => {
  await loadApi(); // plain node, none of the fakes installed — must not throw
  const lkMod = await loadLivekit();
  assert.equal(typeof api.createNoiseFilter, "function");
  assert.equal(api.isNoiseFilterSupported(), false, "no window → not supported");
  assert.equal(lkMod.isLiveKitNoiseFilterSupported(), false);
});

test("thread defaults: medium/high→worker, low→audio; offline→audio+warn", async () => {
  const { createNoiseFilter } = await loadApi();
  const f = installFakes({ wasmBytes: WASM_B });
  try {
    const fh = await createNoiseFilter(f.makeCtx(), { quality: "high" });
    assert.equal(fh.thread, "worker");
    assert.equal(f.workers.length, 1);
    const fm = await createNoiseFilter(f.makeCtx(), { quality: "medium" });
    const fl = await createNoiseFilter(f.makeCtx(), { quality: "low" });
    assert.equal(fm.thread, "worker");
    assert.equal(fl.thread, "audio");
    // worker explicitly requested on an offline context → warned + audio
    const fo = await createNoiseFilter(f.makeOfflineCtx(), { quality: "low", thread: "worker" });
    assert.equal(fo.thread, "audio");
    assert.ok(f.warnings.some((w) => w.includes("OfflineAudioContext")), "no offline warning");
    // explicit thread choice respected
    const fw = await createNoiseFilter(f.makeCtx(), { quality: "medium", thread: "worker" });
    assert.equal(fw.thread, "worker");
  } finally { f.uninstall(); }
});

test("latency: 896/48000 audio, 1408/48000 worker", async () => {
  const { createNoiseFilter } = await loadApi();
  const f = installFakes({ wasmBytes: WASM_B });
  try {
    const a = await createNoiseFilter(f.makeCtx(), { quality: "low" });
    const w = await createNoiseFilter(f.makeCtx(), { quality: "medium", thread: "worker" });
    assert.equal(a.latency, 896 / 48000);
    assert.equal(w.latency, 1408 / 48000);
  } finally { f.uninstall(); }
});

test("failed init rejects, terminates the worker, leaves nothing connected", async () => {
  const { createNoiseFilter } = await loadApi();
  const f = installFakes({ wasmBytes: WASM_B });
  try {
    f.workletReply = "error";
    await assert.rejects(createNoiseFilter(f.makeCtx(), { quality: "high" }), /init failed|timeout|error/);
    assert.equal(f.workers[0].terminated, true, "worker not terminated after failed init");
    assert.equal(f.log.filter((e) => e.ev === "connect").length, 0, "node got connected");

    // fetch 404 path
    f.workletReply = "ready";
    f.fetchFails = true;
    await assert.rejects(createNoiseFilter(f.makeCtx(), { quality: "low" }), /wasm fetch failed: 404/);
  } finally { f.uninstall(); }
});

test("processor: constraints after ready, restore on destroy, setEnabled pre-init", async () => {
  const { LiveKitNoiseFilter } = await loadLivekit();
  const f = installFakes({ wasmBytes: WASM_B });
  try {
    const track = f.makeTrack();
    const ctx = f.makeCtx();
    const p = LiveKitNoiseFilter({ quality: "low" });
    // setEnabled before init → carried into processorOptions
    await p.setEnabled(false);
    await p.init({ track, audioContext: ctx });
    const node = f.nodes.at(-1);
    assert.equal(node.processorOptions.enabled, false, "enabled:false not passed to worklet");
    // constraints applied with only noiseSuppression disabled
    assert.equal(f.constraintsApplied.length, 1);
    const [t0, c0] = f.constraintsApplied[0];
    assert.equal(t0, "track");
    assert.equal(c0.noiseSuppression, false);
    assert.equal(c0.echoCancellation, true, "user's AEC/AGC must be preserved");
    assert.ok(f.processedTrack !== undefined || p.processedTrack, "no processedTrack");
    // destroy restores original constraints before disconnecting
    f.log.length = 0;
    await p.destroy();
    assert.equal(f.constraintsApplied.length, 2);
    assert.deepEqual(f.constraintsApplied[1][1], { echoCancellation: true, noiseSuppression: true });
    const restoreIdx = f.log.findIndex((e) => e.ev === "applyConstraints");
    const discIdx = f.log.findIndex((e) => e.ev === "disconnect");
    assert.ok(restoreIdx !== -1 && discIdx !== -1 && restoreIdx < discIdx,
      "constraints must be restored before teardown");
  } finally { f.uninstall(); }
});

test("processor: enabled option is honored and setEnabled can override it", async () => {
  const { LiveKitNoiseFilter } = await loadLivekit();
  const f = installFakes({ wasmBytes: WASM_B });
  try {
    const p = LiveKitNoiseFilter({ quality: "low", enabled: false });
    assert.equal(p.isEnabled(), false);
    await p.init({ track: f.makeTrack(), audioContext: f.makeCtx() });
    assert.equal(f.nodes.at(-1).processorOptions.enabled, false);
    await p.setEnabled(true);
    assert.equal(p.isEnabled(), true);
    assert.ok(f.nodes.at(-1).port.posted.some((m) => m.type === "setEnabled" && m.value === true));
    await p.destroy();
  } finally { f.uninstall(); }
});

test("processor: failed init never touches the track's constraints", async () => {
  const { LiveKitNoiseFilter } = await loadLivekit();
  const f = installFakes({ wasmBytes: WASM_B });
  try {
    f.workletReply = "error";
    const p = LiveKitNoiseFilter({ quality: "low" });
    await assert.rejects(p.init({ track: f.makeTrack(), audioContext: f.makeCtx() }));
    assert.equal(f.constraintsApplied.length, 0, "constraints applied despite failed init");
  } finally { f.uninstall(); }
});

test("destroy() is idempotent", async () => {
  const { createNoiseFilter } = await loadApi();
  const { LiveKitNoiseFilter } = await loadLivekit();
  const f = installFakes({ wasmBytes: WASM_B });
  try {
    const nf = await createNoiseFilter(f.makeCtx(), { quality: "low" });
    nf.destroy(); nf.destroy();
    const p = LiveKitNoiseFilter({ quality: "low" });
    await p.init({ track: f.makeTrack(), audioContext: f.makeCtx() });
    await p.destroy(); await p.destroy();
  } finally { f.uninstall(); }
});

test("maxChannels: stereo node options; medium defaults to worker", async () => {
  const { createNoiseFilter } = await loadApi();
  const f = installFakes({ wasmBytes: WASM_B });
  try {
    const nf = await createNoiseFilter(f.makeCtx(), { quality: "medium", maxChannels: 2 });
    assert.equal(nf.thread, "worker", "medium must default to worker");
    assert.equal(f.workers.length, 2, "expected one worker per channel");
    const node = f.nodes.at(-1);
    assert.equal(node.options.channelCount, 2);
    assert.equal(node.options.channelCountMode, "explicit");
    assert.equal(node.options.channelInterpretation, "speakers");
    assert.deepEqual(node.options.outputChannelCount, [2]);
    assert.equal(node.processorOptions.channels, 2);
    const portsMsg = node.port.posted.find((m) => m.type === "ports");
    assert.ok(portsMsg && portsMsg.ports.length === 2, "no {type:'ports'} with 2 ports");
    // low stays audio even in stereo
    const nl = await createNoiseFilter(f.makeCtx(), { quality: "low", maxChannels: 2 });
    assert.equal(nl.thread, "audio");
    assert.equal(f.nodes.at(-1).processorOptions.channels, 2);
  } finally { f.uninstall(); }
});

test("gate tier: no fetch, no worker, latency 0, always audio", async () => {
  const { createNoiseFilter } = await loadApi();
  const f = installFakes({ wasmBytes: WASM_B });
  try {
    const nf = await createNoiseFilter(f.makeCtx(), { quality: "gate", gate: { holdMs: 100 } });
    assert.equal(nf.latency, 0);
    assert.equal(nf.thread, "audio");
    assert.equal(f.fetchCalls.length, 0, "gate fetched wasm");
    assert.equal(f.workers.length, 0, "gate spawned a worker");
    assert.equal(f.nodes.at(-1).processorOptions.mode, "gate");
    assert.equal(f.nodes.at(-1).processorOptions.gate.holdMs, 100);
    nf.destroy();
  } finally { f.uninstall(); }
});

test("compat: node classes get right options; loadNoiseFilter SIMD fallback", async () => {
  const compat = await import(pathToFileURL(join(tmp, "compat.mjs")).href);
  const f = installFakes({ wasmBytes: WASM_B });
  try {
    const ctx = f.makeCtx();
    const node = new compat.NoiseFilterWorkletNode(ctx, { wasmBinary: WASM_B.slice(0), maxChannels: 2, enabled: false });
    assert.equal(node.processorOptions.channels, 2);
    assert.equal(node.processorOptions.enabled, false);
    assert.equal(node.options.outputChannelCount[0], 2);
    node.setEnabled(true);
    assert.ok(node.port.posted.some((m) => m.type === "setEnabled" && m.value === true));
    assert.ok(node instanceof compat.NoiseFilterWorkletNode);
    assert.ok(node instanceof AudioWorkletNode);
    const gateNode = new compat.NoiseGateWorkletNode(ctx, { openThreshold: -40 });
    assert.equal(gateNode.processorOptions.mode, "gate");
    assert.equal(gateNode.processorOptions.gate.openThreshold, -40);
    gateNode.destroy();

    // SIMD present → primary url
    const ab = await compat.loadNoiseFilter({ url: "https://x/fastenhancer_b.wasm" });
    assert.ok(ab.byteLength > 1000);
    assert.equal(f.fetchCalls.at(-1), "https://x/fastenhancer_b.wasm");
    // SIMD missing → scalarUrl
    const origValidate = WebAssembly.validate;
    WebAssembly.validate = () => false;
    try {
      await compat.loadNoiseFilter({ url: "https://x/simd.wasm", scalarUrl: "https://x/scalar.wasm" });
      assert.equal(f.fetchCalls.at(-1), "https://x/scalar.wasm");
      await assert.rejects(compat.loadNoiseFilter({ url: "https://x/simd.wasm" }), /scalarUrl|SIMD/);
    } finally { WebAssembly.validate = origValidate; }
    // fetch failure mentions the URL
    f.fetchFails = true;
    await assert.rejects(compat.loadNoiseFilter({ url: "https://x/missing.wasm" }), /missing\.wasm/);
  } finally { f.uninstall(); }
});

test("compat SSR: import in plain node doesn't throw; constructing does", async () => {
  // the module was already imported above; check the stub path by unsetting globals
  const req = createRequire(join(tmp, "index.cjs"));
  assert.equal(typeof req("./index.cjs").createNoiseFilter, "function");
  assert.equal(typeof req("./compat.cjs").loadNoiseFilter, "function");
});

test("non-SIMD: scalar wasm URL, worker defaults, high downgrades to medium", async () => {
  const { createNoiseFilter, isNoiseFilterSupported } = await loadApi();
  const origValidate = WebAssembly.validate;
  WebAssembly.validate = () => false; // pretend no WASM SIMD
  const f = installFakes({ wasmBytes: WASM_B });
  try {
    assert.equal(isNoiseFilterSupported(), true, "scalar fallback should keep it supported");
    const nl = await createNoiseFilter(f.makeCtx(), { quality: "low" });
    assert.equal(nl.thread, "audio");
    assert.ok(f.fetchCalls.at(-1).includes("fastenhancer_t_scalar.wasm"), f.fetchCalls.at(-1));
    const nm = await createNoiseFilter(f.makeCtx(), { quality: "medium" });
    assert.equal(nm.thread, "worker", "scalar medium must default to worker");
    assert.ok(f.fetchCalls.at(-1).includes("fastenhancer_b_scalar.wasm"), f.fetchCalls.at(-1));
    const nls = await createNoiseFilter(f.makeCtx(), { quality: "low", maxChannels: 2 });
    assert.equal(nls.thread, "worker", "scalar low stereo must default to worker");
    const nh = await createNoiseFilter(f.makeCtx(), { quality: "high" });
    assert.equal(nh.quality, "medium", "scalar high must downgrade to medium");
    assert.equal(nh.thread, "worker");
    assert.ok(f.warnings.some((w) => w.includes("SIMD")), "no downgrade warning");
    // explicit scalar url wins; it also wins over wasm — a SIMD-only binary
    // would fail to compile in this browser
    await createNoiseFilter(f.makeCtx(), {
      quality: "low",
      urls: {
        worklet: "https://x/worklet.js",
        wasm: "https://x/my_simd.wasm",
        scalarWasm: "https://x/my_scalar.wasm",
      },
    });
    assert.equal(f.fetchCalls.at(-1), "https://x/my_scalar.wasm");
  } finally { WebAssembly.validate = origValidate; f.uninstall(); }
});

test("CJS: createNoiseFilter without urls rejects with the CommonJS message; with urls works", async () => {
  const req = createRequire(join(tmp, "index.cjs"));
  const { createNoiseFilter } = req("./index.cjs");
  const f = installFakes({ wasmBytes: WASM_B });
  try {
    await assert.rejects(
      createNoiseFilter(f.makeCtx(), { quality: "low" }),
      /can't locate bundled assets \(CommonJS build\)/,
    );
    const nf = await createNoiseFilter(f.makeCtx(), {
      quality: "low",
      urls: { worklet: "https://x/worklet.js", wasm: "https://x/b.wasm" },
    });
    assert.equal(nf.thread, "audio");
    nf.destroy();
  } finally { f.uninstall(); }
});
