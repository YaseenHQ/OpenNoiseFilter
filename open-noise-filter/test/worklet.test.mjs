import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { makeProcessor, wasmBytes, run, runChannels, postMessage, makeFakeWorker, attachWorker, makeWorkerProc, makeChannel } from "./harness.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const f32 = (name) => {
  const b = readFileSync(join(HERE, name));
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
};
const X = f32("reference_x.f32");
const Y_REF = f32("reference_y.f32");
const WASM_B = wasmBytes("b");

// Band-limited resampler for building resampler-path inputs and references
// (windowed sinc, 64 taps each side). Measured baselines for this comparison:
// 44100 Hz → lag 585, SNR 32.98 dB, zero-run 0; 16000 Hz → lag 213, SNR 25.04
// dB, zero-run 0.
function sincResample(x, from, to) {
  const n = Math.floor(x.length * to / from), y = new Float32Array(n);
  const fc = Math.min(1, to / from), A = 64;
  for (let i = 0; i < n; i++) {
    const c = i * from / to, j0 = Math.floor(c);
    let s = 0;
    for (let j = j0 - A; j <= j0 + A; j++) {
      if (j < 0 || j >= x.length) continue;
      const d = c - j, w = 0.5 + 0.5 * Math.cos(Math.PI * d / (A + 1));
      s += x[j] * fc * (d === 0 ? 1 : Math.sin(Math.PI * fc * d) / (Math.PI * fc * d)) * w;
    }
    y[i] = s;
  }
  return y;
}

test("48k: output matches ONNX reference (lag 896, max < 1e-6)", () => {
  const p = makeProcessor(48000, { wasm: WASM_B });
  const y = run(p, X);
  const skip = 4096;
  let mx = 0;
  for (let i = skip; i < X.length - skip; i++) {
    mx = Math.max(mx, Math.abs(y[i + 896] - Y_REF[i]));
  }
  assert.ok(mx < 1e-6, `max diff ${mx}`);
});

test("memory stays bounded (60 s at 48 kHz and 44.1 kHz)", () => {
  for (const sr of [48000, 44100]) {
    const p = makeProcessor(sr, { wasm: WASM_B });
    const inp = new Float32Array(128), out = new Float32Array(128);
    let seed = 1;
    const nq = Math.ceil(sr * 60 / 128);
    for (let q = 0; q < nq; q++) {
      for (let i = 0; i < 128; i++) { seed = (seed * 1103515245 + 12345) & 0x7fffffff; inp[i] = (seed / 0x40000000 - 1) * 0.1; }
      p.process([[inp]], [[out]], {});
    }
    for (const k of Object.keys(p)) {
      const v = p[k];
      assert.ok(!(Array.isArray(v) && v.length > 16384), `own array prop "${k}" grew to ${v && v.length} at ${sr} Hz`);
      if (v && ArrayBuffer.isView(v) && k !== "mem" && !(p.engines && p.engines.some((e) => v === e.mem))) {
        assert.ok(v.length <= 16384, `own typed prop "${k}" is ${v.length} at ${sr} Hz`);
      }
    }
  }
});

for (const [SR, expLag, minSnr] of [[44100, 585, 30], [16000, 213, 22]]) {
  test(`${SR} Hz resampler: SNR and timing vs band-limited reference`, () => {
    const y48 = run(makeProcessor(48000, { wasm: WASM_B }), X);
    const x = sincResample(X, 48000, SR);
    const y = run(makeProcessor(SR, { wasm: WASM_B }), x);
    const ref = sincResample(y48, 48000, SR);
    const a = Math.floor(0.3 * SR), b = Math.floor(ref.length - 0.3 * SR);
    let best = 0, bestE = Infinity;
    for (let lag = expLag - 64; lag <= expLag + 64; lag++) {
      let e = 0;
      for (let i = a; i < b; i += 7) { const d = (y[i + lag] ?? 0) - ref[i]; e += d * d; }
      if (e < bestE) { bestE = e; best = lag; }
    }
    assert.ok(Math.abs(best - expLag) < 64, `lag search hit window edge (${best})`);
    let e = 0, s = 0;
    for (let i = a; i < b; i++) { const d = y[i + best] - ref[i]; e += d * d; s += ref[i] * ref[i]; }
    const snr = 10 * Math.log10(s / e);
    let zeroRun = 0, maxZeroRun = 0;
    for (let i = Math.floor(0.1 * SR); i < y.length - 256; i++) {
      zeroRun = y[i] === 0 ? zeroRun + 1 : 0;
      maxZeroRun = Math.max(maxZeroRun, zeroRun);
    }
    assert.ok(snr >= minSnr, `SNR ${snr.toFixed(2)} dB < ${minSnr} dB`);
    assert.ok(maxZeroRun <= 8, `zero-run ${maxZeroRun} > 8`);
  });
}

test("setEnabled: exact copy-through; re-enable resets DSP state", () => {
  const K = 60; // quantum at which we re-enable
  const p = makeProcessor(48000, { wasm: WASM_B });
  const inp = new Float32Array(128), out = new Float32Array(128);
  const yA = new Float32Array(X.length);

  for (let q = 0; q * 128 < X.length; q++) {
    inp.set(X.subarray(q * 128, q * 128 + 128));
    if (q === 40) postMessage(p, { type: "setEnabled", value: false });
    if (q === K) postMessage(p, { type: "setEnabled", value: true });
    p.process([[inp]], [[out]], {});
    yA.set(out, q * 128);
    if (q >= 40 && q < K) {
      for (let i = 0; i < 128; i++) assert.equal(out[i], inp[i], `copy-through failed at q=${q}`);
    }
  }

  const fresh = makeProcessor(48000, { wasm: WASM_B });
  const yB = run(fresh, X.subarray(K * 128));
  let mx = 0;
  for (let i = 0; i < yB.length; i++) mx = Math.max(mx, Math.abs(yA[K * 128 + i] - yB[i]));
  assert.ok(mx < 1e-6, `re-enabled output diverges from fresh processor: ${mx}`);
});

test("destroy: process() returns false", () => {
  const p = makeProcessor(48000, { wasm: WASM_B });
  const io = () => [[new Float32Array(128)], [new Float32Array(128)]];
  assert.equal(p.process(...io()), true);
  postMessage(p, { type: "destroy" });
  assert.equal(p.process(...io()), false);
});

test("48k: scalar (non-SIMD) build matches ONNX reference (max < 1e-5)", () => {
  const y = run(makeProcessor(48000, { wasm: wasmBytes("b_scalar") }), X);
  const skip = 4096;
  let mx = 0;
  for (let i = skip; i < X.length - skip; i++) {
    mx = Math.max(mx, Math.abs(y[i + 896] - Y_REF[i]));
  }
  assert.ok(mx < 1e-5, `scalar max diff ${mx}`);
});

test("all tiers instantiate and emit ready", () => {
  for (const tier of ["t", "b", "s"]) {
    const p = makeProcessor(48000, { wasm: wasmBytes(tier) });
    assert.ok(p.port.messages.some((m) => m && m.type === "ready"), `tier ${tier} did not emit ready`);
  }
});

function runWorker(proc, worker, x) {
  const y = new Float32Array(Math.ceil(x.length / 128) * 128);
  const inp = new Float32Array(128), out = new Float32Array(128);
  for (let q = 0; q * 128 < x.length; q++) {
    inp.set(x.subarray(q * 128, q * 128 + 128));
    proc.process([[inp]], [[out]], {});
    y.set(out, q * 128);
    worker.tick();
  }
  return y;
}

test("worker mode: output = audio output delayed by exactly 512 samples", () => {
  const pa = makeProcessor(48000, { wasm: WASM_B });
  const ya = run(pa, X);

  const pw = makeProcessor(48000, { mode: "worker" });
  const w = makeFakeWorker(WASM_B, { maxDelay: 3 }); // replies delayed 0–3 quanta
  attachWorker(pw, w);
  const yw = runWorker(pw, w, X);

  let mx = 0;
  for (let i = 0; i < ya.length - 512; i++) mx = Math.max(mx, Math.abs(yw[i + 512] - ya[i]));
  assert.ok(mx < 1e-6, `worker vs audio +512 shift: max diff ${mx}`);
  assert.equal(pw.underruns, 0, `underruns ${pw.underruns}`);
});

test("worker mode: mid-stream stall rebuffers and recovers", () => {
  const pw = makeProcessor(48000, { mode: "worker" });
  const w = makeFakeWorker(WASM_B, { auto: false });
  attachWorker(pw, w);
  const inp = new Float32Array(128), out = new Float32Array(128);
  const y = new Float32Array(X.length);
  for (let q = 0; q * 128 < X.length; q++) {
    inp.set(X.subarray(q * 128, q * 128 + 128));
    // normal service, then a stall longer than the slack, then service resumes
    if (q < 100 || q >= 120) w.flush();
    assert.doesNotThrow(() => pw.process([[inp]], [[out]], {}));
    y.set(out, q * 128);
    for (let i = 0; i < 128; i++) assert.ok(Number.isFinite(out[i]), `non-finite at q=${q}`);
  }
  assert.ok(pw.underruns > 0, "no underrun counted");
  let tail = 0;
  for (let i = X.length - 4096; i < X.length; i++) tail += Math.abs(y[i]);
  assert.ok(tail > 0, "output did not resume after stall");
});

test("worker mode: pool exhaustion counts overrun, resets, ignores stale gen", () => {
  const pw = makeProcessor(48000, { mode: "worker" });
  const w = makeFakeWorker(WASM_B, { auto: false }); // never replies
  attachWorker(pw, w);
  const inp = new Float32Array(128), out = new Float32Array(128);
  for (let q = 0; q < 40; q++) { inp.fill(0.01); pw.process([[inp]], [[out]], {}); }
  assert.ok(pw.overruns > 0, `overruns ${pw.overruns}`);
  assert.ok(w.resets.length > 0, "no reset posted to worker");
  const produced = pw.produced;
  // a reply tagged with the stale gen must be ignored (buffer still returns)
  postMessageFake(pw, { type: "out", gen: w.resets[0] - 1, buf: new Float32Array(512).fill(0.5), produced: true });
  assert.equal(pw.produced, produced, "stale-gen reply was applied");
});

function postMessageFake(proc, data) {
  // deliver a worker→worklet message directly
  proc.wports[0].onmessage({ data });
}

test("worker mode: setEnabled copy-through and reset on re-enable", () => {
  const pw = makeProcessor(48000, { mode: "worker" });
  const w = makeFakeWorker(WASM_B, { maxDelay: 1 });
  attachWorker(pw, w);
  const inp = new Float32Array(128), out = new Float32Array(128);
  for (let q = 0; q < 40; q++) { inp.fill(0.02); pw.process([[inp]], [[out]], {}); w.tick(); }
  postMessage(pw, { type: "setEnabled", value: false });
  for (let q = 0; q < 8; q++) { inp.fill(0.03); pw.process([[inp]], [[out]], {}); w.tick(); }
  for (let i = 0; i < 128; i++) assert.equal(out[i], inp[i], "not copy-through while disabled");
  const genBefore = pw.gen;
  postMessage(pw, { type: "setEnabled", value: true });
  assert.ok(pw.gen > genBefore, "gen not bumped on re-enable");
  assert.ok(w.resets.includes(pw.gen), "reset not posted to worker");
});

test("worker mode: repeated stalls grow target to 4H and bound latency", () => {
  const H_ = 512;
  const pw = makeProcessor(48000, { mode: "worker" });
  const w = makeFakeWorker(WASM_B, { auto: false });
  attachWorker(pw, w);
  const pa = makeProcessor(48000, { wasm: WASM_B });

  const S = 20 * 48000;
  const x = new Float32Array(S);
  for (let i = 0; i < S; i++) x[i] = X[i % X.length];
  const y = new Float32Array(S), ya = new Float32Array(S);
  const inp = new Float32Array(128), out = new Float32Array(128), oa = new Float32Array(128);
  const hist = []; // {q, underruns, target} whenever an underrun is counted
  let uPrev = 0;
  for (let q = 0; q * 128 < S; q++) {
    inp.set(x.subarray(q * 128, q * 128 + 128));
    pa.process([[inp]], [[oa]], {});
    ya.set(oa, q * 128);
    pw.process([[inp]], [[out]], {});
    y.set(out, q * 128);
    // stall ~2048 samples (16 quanta) every ~2 s (375 quanta): long enough to
    // drain the deepest target (4H) yet short enough not to exhaust the pool
    const stalling = q >= 375 && (q % 375) < 16;
    if (!stalling) w.flush();
    assert.ok(pw.produced - pw.consumed <= pw.target + 2 * H_,
      `produced-consumed ${pw.produced - pw.consumed} > target+H+H at q=${q}`);
    if (pw.underruns !== uPrev) { uPrev = pw.underruns; hist.push({ q, u: pw.underruns, t: pw.target }); }
  }
  for (let i = 0; i < y.length; i++) assert.ok(Number.isFinite(y[i]), `non-finite at ${i}`);
  assert.ok(hist.length >= 2, `expected repeated underruns, got ${hist.length}`);
  // target grows by exactly one hop per underrun and caps at 4H
  for (const h of hist) assert.ok(h.t === Math.min(2 * H_ + h.u * H_, 4 * H_),
    `target ${h.t} != min(2H+${h.u}H, 4H)`);
  // final delay ≤ 896 + target + H vs the audio-mode output
  const cap = 896 + pw.target + H_;
  let best = -1, bestE = Infinity;
  const i0 = y.length - 2 * 48000, i1 = y.length;
  for (let lag = 0; lag <= 896 + 4 * H_ + H_; lag++) {
    let e = 0;
    for (let i = i0; i < i1; i += 13) { const d = y[i] - (ya[i - lag] ?? 0); e += d * d; }
    if (e < bestE) { bestE = e; best = lag; }
  }
  assert.ok(best >= 0 && best <= cap, `final delay ${best} > ${cap} (target ${pw.target})`);
});

test("worker mode: ring guard — a 12-reply burst can't overwrite unread samples", () => {
  const pw = makeProcessor(48000, { mode: "worker" });
  const w = makeFakeWorker(WASM_B, { auto: false });
  attachWorker(pw, w);
  for (let k = 0; k < 12; k++) {
    postMessageFake(pw, { type: "out", gen: pw.gen, buf: new Float32Array(512).fill(0.1), produced: true });
    assert.ok(pw.produced - pw.consumed <= 4096,
      `produced-consumed ${pw.produced - pw.consumed} exceeded ring after reply ${k}`);
  }
  assert.ok(pw.skips > 0, "no skip counted for the burst");
});

test("worker mode: 44.1k stall resumes cleanly (no repeated edge hits)", () => {
  const SR = 44100;
  const x = sincResample(X, 48000, SR);
  const pw = makeProcessor(SR, { mode: "worker" });
  const w = makeFakeWorker(WASM_B, { auto: false });
  attachWorker(pw, w);
  const nq = Math.ceil(x.length / 128);
  const y = new Float32Array(nq * 128);
  const inp = new Float32Array(128), out = new Float32Array(128);
  const stallQ = Math.floor(nq / 2), stallLen = 12; // ~1.6k ring samples: forces an underrun, fits the pool
  for (let q = 0; q < nq; q++) {
    inp.set(x.subarray(q * 128, q * 128 + 128));
    pw.process([[inp]], [[out]], {});
    y.set(out, q * 128);
    if (q < stallQ || q >= stallQ + stallLen) w.flush();
  }
  assert.ok(pw.underruns > 0, "no underrun counted");
  // find where output resumes after the stall, then scan the following 2 s
  const stallEnd = (stallQ + stallLen) * 128;
  let resume = -1;
  for (let i = stallEnd; i < y.length; i++) if (y[i] !== 0) { resume = i; break; }
  assert.ok(resume > 0, "output never resumed after the stall");
  const uAfter = pw.underruns;
  let zeroRun = 0, maxZeroRun = 0;
  const end = Math.min(resume + 2 * SR, y.length - 256);
  for (let i = resume; i < end; i++) {
    zeroRun = y[i] === 0 ? zeroRun + 1 : 0;
    maxZeroRun = Math.max(maxZeroRun, zeroRun);
  }
  assert.ok(maxZeroRun <= 8, `zero-run ${maxZeroRun} > 8 after resume`);
  assert.equal(pw.underruns, uAfter, "underrun count kept increasing after resume");
});

test("resampled worker: sustained 44.1k and 16k playback has no false skips", () => {
  for (const SR of [44100, 16000]) {
    const pw = makeProcessor(SR, { mode: "worker" });
    const w = makeFakeWorker(WASM_B, { auto: false });
    attachWorker(pw, w);
    const inp = new Float32Array(128), out = new Float32Array(128);
    let nonzero = 0;
    const quanta = Math.ceil(SR * 4 / 128);
    for (let q = 0; q < quanta; q++) {
      for (let i = 0; i < 128; i++) inp[i] = Math.sin((q * 128 + i) * 0.07) * 0.1;
      pw.process([[inp]], [[out]], {});
      for (let i = 0; i < 128; i++) if (out[i] !== 0) nonzero++;
      w.flush(); // deliver all replies without adding artificial worker stalls
      assert.ok(pw.produced - pw.readCursor <= 4096,
        `${SR} Hz unread ring span ${pw.produced - pw.readCursor} exceeded capacity`);
    }
    assert.ok(nonzero > 0, `${SR} Hz worker never emitted audio`);
    assert.equal(pw.underruns, 0, `${SR} Hz had ${pw.underruns} underruns`);
    assert.equal(pw.skips, 0, `${SR} Hz had ${pw.skips} false skips`);
  }
});

test("resampled stereo worker: burst skips both channel readers before ring wrap", () => {
  const pw = makeProcessor(44100, { mode: "worker", channels: 2 });
  const ws = [
    makeFakeWorker(WASM_B, { auto: false }),
    makeFakeWorker(WASM_B, { auto: false }),
  ];
  attachWorker(pw, ws);
  for (let k = 0; k < 12; k++) {
    for (let c = 0; c < 2; c++) {
      pw.onWorkerMessage({ type: "out", gen: pw.gen, buf: new Float32Array(512).fill(c ? 0.2 : 0.1), produced: true }, c);
      assert.ok(pw.producedC[c] - pw.readCursor <= 4096,
        `channel ${c} would overwrite unread resampler data at burst ${k}`);
    }
  }
  assert.ok(pw.outRead > 0, "burst did not move the resampled read cursor");
  assert.ok(pw.skips > 0, "burst did not count a necessary resampled skip");
  assert.equal(pw.readCursor, Math.floor(pw.outRead) - 1, "stereo ring guard lost interpolation history");
});

test("worker mode: setEnabled re-enable resets target to 2*H", () => {
  const pw = makeProcessor(48000, { mode: "worker" });
  const w = makeFakeWorker(WASM_B, { auto: false });
  attachWorker(pw, w);
  const inp = new Float32Array(128).fill(0.02), out = new Float32Array(128);
  for (let q = 0; q < 60; q++) { pw.process([[inp]], [[out]], {}); w.flush(); }
  for (let q = 0; q < 30; q++) pw.process([[inp]], [[out]], {}); // stall → underrun
  assert.ok(pw.target > 2 * 512, `target did not grow (${pw.target})`);
  postMessage(pw, { type: "setEnabled", value: false });
  postMessage(pw, { type: "setEnabled", value: true });
  assert.equal(pw.target, 2 * 512, `target ${pw.target} not reset to 2H`);
});

test("worker.js: hop replies match a directly-driven FrameEngine", async () => {
  const { FrameEngine } = await import("../src/dsp.js");
  const inst = new WebAssembly.Instance(new WebAssembly.Module(WASM_B));
  const ref = new FrameEngine(inst.exports);

  const { self, sent } = makeWorkerProc();
  const ch = makeChannel();
  const outs = [];
  ch.b.onmessage = (ev) => outs.push(ev.data);
  self.onmessage({ data: { type: "init", wasm: WASM_B, port: ch.a } });
  for (let i = 0; i < 100 && !sent.length; i++) await new Promise((r) => setTimeout(r, 10));
  assert.ok(sent.some((m) => m.type === "ready"), `worker never readied: ${JSON.stringify(sent)}`);

  const hops = 8, refOut = [];
  for (let k = 0; k < hops; k++) {
    const hop = new Float32Array(512);
    for (let i = 0; i < 512; i++) hop[i] = Math.sin((k * 512 + i) * 0.02) * 0.5;
    const ro = new Float32Array(512);
    const produced = ref.processHop(hop.slice(), ro);
    refOut.push({ produced, ro });
    ch.b.postMessage({ type: "hop", gen: 0, buf: hop });
  }
  await new Promise((r) => setTimeout(r, 50));
  assert.equal(outs.length, hops);
  for (let k = 0; k < hops; k++) {
    assert.equal(outs[k].type, "out");
    assert.equal(outs[k].produced, refOut[k].produced);
    if (refOut[k].produced) {
      for (let i = 0; i < 512; i++) assert.equal(outs[k].buf[i], refOut[k].ro[i], `hop ${k} sample ${i}`);
    }
  }
  // reset adopts a new gen; stale-gen hops are dropped
  ch.b.postMessage({ type: "reset", gen: 7 });
  ch.b.postMessage({ type: "hop", gen: 0, buf: new Float32Array(512) });
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(outs.length, hops, "stale-gen hop was not dropped");
});


test("stereo audio mode: each channel equals a mono run of that channel", () => {
  const R = new Float32Array(X.length);
  for (let i = 0; i < X.length; i++) R[i] = 0.5 * X[X.length - 1 - i];
  const ys = runChannels(makeProcessor(48000, { wasm: WASM_B, channels: 2 }), [X, R]);
  const yl = run(makeProcessor(48000, { wasm: WASM_B }), X);
  const yr = run(makeProcessor(48000, { wasm: WASM_B }), R);
  let mx0 = 0, mx1 = 0;
  for (let i = 0; i < yl.length; i++) mx0 = Math.max(mx0, Math.abs(ys[0][i] - yl[i]));
  for (let i = 0; i < yr.length; i++) mx1 = Math.max(mx1, Math.abs(ys[1][i] - yr[i]));
  assert.ok(mx0 < 1e-6, `left diff ${mx0}`);
  assert.ok(mx1 < 1e-6, `right diff ${mx1}`);
});

test("stereo worker mode: +1024 shift (3H start), per-channel delays, 0 underruns", () => {
  const R = new Float32Array(X.length);
  for (let i = 0; i < X.length; i++) R[i] = 0.5 * X[X.length - 1 - i];
  const ya = runChannels(makeProcessor(48000, { wasm: WASM_B, channels: 2 }), [X, R]);

  const pw = makeProcessor(48000, { mode: "worker", channels: 2 });
  const w0 = makeFakeWorker(WASM_B, { maxDelay: 2 });
  const w1 = makeFakeWorker(WASM_B, { maxDelay: 3 }); // ch1 one quantum later
  attachWorker(pw, [w0, w1]);
  const yw = runChannels(pw, [X, R], () => { w0.tick(); w1.tick(); });

  for (const c of [0, 1]) {
    let mx = 0;
    for (let i = 0; i < ya[c].length - 1024; i++) mx = Math.max(mx, Math.abs(yw[c][i + 1024] - ya[c][i]));
    assert.ok(mx < 1e-6, `ch${c} worker vs audio +1024: ${mx}`);
  }
  assert.equal(pw.underruns, 0, `underruns ${pw.underruns}`);
});

test("memory stays bounded with 2 channels (60 s at 48 kHz and 44.1 kHz)", () => {
  for (const sr of [48000, 44100]) {
    const p = makeProcessor(sr, { wasm: WASM_B, channels: 2 });
    const i0 = new Float32Array(128), i1 = new Float32Array(128);
    const o0 = new Float32Array(128), o1 = new Float32Array(128);
    let seed = 1;
    const nq = Math.ceil(sr * 60 / 128);
    for (let q = 0; q < nq; q++) {
      for (let i = 0; i < 128; i++) { seed = (seed * 1103515245 + 12345) & 0x7fffffff; i0[i] = (seed / 0x40000000 - 1) * 0.1; i1[i] = -i0[i] * 0.5; }
      p.process([[i0, i1]], [[o0, o1]], {});
    }
    for (const k of Object.keys(p)) {
      const v = p[k];
      if (v && ArrayBuffer.isView(v) && k !== "mem" && !(p.engines && p.engines.some((e) => v === e.mem))) {
        assert.ok(v.length <= 16384, `own typed prop "${k}" is ${v.length} at ${sr} Hz`);
      }
    }
  }
});

test("stereo worker mode: an overrun on one channel resets both", () => {
  const pw = makeProcessor(48000, { mode: "worker", channels: 2 });
  const w0 = makeFakeWorker(WASM_B, { auto: false }); // ch0 never replies → pool exhausts
  const w1 = makeFakeWorker(WASM_B, { auto: true });
  attachWorker(pw, [w0, w1]);
  const i0 = new Float32Array(128).fill(0.02), i1 = new Float32Array(128).fill(0.03);
  const o0 = new Float32Array(128), o1 = new Float32Array(128);
  for (let q = 0; q < 80; q++) { pw.process([[i0, i1]], [[o0, o1]], {}); w1.tick(); }
  assert.ok(pw.overruns > 0, `overruns ${pw.overruns}`);
  assert.ok(w0.resets.length > 0, "no reset posted to channel 0");
  assert.ok(w1.resets.length > 0, "no reset posted to channel 1");
});


function noiseAmp(dbfs) { return Math.pow(10, dbfs / 20) * Math.SQRT2; }
function runGate(proc, seconds, amp, sr = 48000) {
  const inp = new Float32Array(128), out = new Float32Array(128);
  let seed = 7;
  const nq = Math.ceil(seconds * sr / 128);
  const ys = new Float32Array(nq * 128), xs = new Float32Array(nq * 128);
  for (let q = 0; q < nq; q++) {
    for (let i = 0; i < 128; i++) { seed = (seed * 1103515245 + 12345) & 0x7fffffff; inp[i] = (seed / 0x40000000 - 1) * amp; }
    proc.process([[inp]], [[out]], {});
    xs.set(inp, q * 128); ys.set(out, q * 128);
  }
  return { x: xs, y: ys };
}
const rms = (a, i0, i1) => { let e = 0; for (let i = i0; i < i1; i++) e += a[i] * a[i]; return Math.sqrt(e / (i1 - i0)); };

test("gate: -70 dBFS noise decays to gain < 0.01 within hold + 5×release", () => {
  const p = makeProcessor(48000, { mode: "gate" }); // defaults: hold 200 ms, release 50 ms
  runGate(p, (200 + 5 * 50 + 10) / 1000, noiseAmp(-70));
  assert.ok(p.gateGain < 0.01, `gateGain ${p.gateGain}`);
});

test("gate: -20 dBFS tone opens and passes at gain > 0.99 after attack", () => {
  const p = makeProcessor(48000, { mode: "gate" });
  runGate(p, 0.5, noiseAmp(-70)); // close first
  assert.ok(p.gateGain < 0.5, `not closed (${p.gateGain})`);
  const inp = new Float32Array(128), out = new Float32Array(128);
  for (let i = 0; i < 128; i++) inp[i] = Math.sin(i * 0.1) * noiseAmp(-20);
  let openedQ = -1;
  for (let q = 0; q < 30; q++) {
    p.process([[inp]], [[out]], {});
    if (p.gateGain > 0.99 && openedQ < 0) openedQ = q;
  }
  assert.ok(openedQ >= 0, "gate never reached gain > 0.99");
  assert.ok(openedQ * 128 / 48000 < 0.03, `reached 0.99 after ${openedQ * 128 / 48} ms — slower than attack`);
});

test("gate: level between thresholds keeps the current state (hysteresis)", () => {
  const p = makeProcessor(48000, { mode: "gate" });
  runGate(p, 0.3, noiseAmp(-50)); // between -55 close and -45 open; started open
  assert.ok(p.gateGain > 0.9, `open state not held (${p.gateGain})`);
  runGate(p, 0.5, noiseAmp(-70)); // close fully
  assert.ok(p.gateGain < 0.05, `not closed (${p.gateGain})`);
  runGate(p, 0.3, noiseAmp(-50)); // between thresholds again
  assert.ok(p.gateGain < 0.05, `closed state not held (${p.gateGain})`);
});

test("gate: per-sample gain change never exceeds the attack coefficient", () => {
  const p = makeProcessor(48000, { mode: "gate" });
  runGate(p, 0.5, noiseAmp(-70)); // closed
  // reopen with a constant DC level: out[i]/in[i] tracks the gain exactly
  const atkBound = 1 - Math.exp(-1 / (0.005 * 48000));
  const inp = new Float32Array(128).fill(0.5), out = new Float32Array(128);
  let prevG = p.gateGain; // the step bound measures continuity from the real state
  for (let q = 0; q < 20; q++) {
    inp.fill(0.5); // force-open: constant 0.5 → -6 dBFS
    p.process([[inp]], [[out]], {});
    for (let i = 0; i < 128; i++) {
      const g = out[i] / 0.5;
      assert.ok(g - prevG <= atkBound + 1e-9, `per-sample gain step ${g - prevG} > ${atkBound}`);
      prevG = g;
    }
  }
});

test("gate: stereo gating is linked (max level across channels)", () => {
  const p = makeProcessor(48000, { mode: "gate", channels: 2 });
  runGate(p, 0.5, noiseAmp(-70)); // close
  // left loud, right quiet: gate opens on left; right passes through unmuted
  const i0 = new Float32Array(128), i1 = new Float32Array(128);
  const o0 = new Float32Array(128), o1 = new Float32Array(128);
  let seed = 3;
  for (let q = 0; q < 200; q++) {
    for (let i = 0; i < 128; i++) {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      i0[i] = (seed / 0x40000000 - 1) * noiseAmp(-20);
      i1[i] = (i0[i] * 1e-4); // -100 dBFS-ish echo of the loud channel
    }
    p.process([[i0, i1]], [[o0, o1]], {});
  }
  const g1 = rms(o1, 0, 128) / rms(i1, 0, 128);
  assert.ok(g1 > 0.9, `quiet channel was attenuated (${g1}) — gating not linked`);
});

test("gate: setEnabled bypasses and destroy returns false", () => {
  const p = makeProcessor(48000, { mode: "gate" });
  const inp = new Float32Array(128).fill(0.4), out = new Float32Array(128);
  postMessage(p, { type: "setEnabled", value: false });
  p.process([[inp]], [[out]], {});
  for (let i = 0; i < 128; i++) assert.equal(out[i], inp[i]);
  postMessage(p, { type: "destroy" });
  assert.equal(p.process([[inp]], [[out]], {}), false);
});
