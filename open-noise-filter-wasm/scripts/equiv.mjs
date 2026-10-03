/**
 * Equivalence test: compiled WASM vs the ONNX Runtime reference.
 * Rebuilds reference.py's exact pipeline (periodic hann, right-pad one hop,
 * raw complex STFT), streams frames through the WASM module, and diffs the
 * final iSTFT audio against reference_y.npy (the ORT-produced output).
 *
 * Usage: node scripts/equiv.mjs [t|b|s]
 */
import { readFileSync } from "fs";
import { join } from "path";

const tier = (process.argv[2] ?? "b").toLowerCase();
const N_FFT = 1024, HOP = 512, SR = 48000;

function loadNpyF32(p) {
  const b = readFileSync(p);
  if (b.subarray(0, 6).toString("latin1") !== "\x93NUMPY") throw new Error("not npy: " + p);
  const hlen = b.readUInt16LE(8);
  const hdr = b.subarray(10, 10 + hlen).toString("latin1");
  const descr = hdr.match(/'descr':\s*'([^']+)'/)[1];
  const fortran = /'fortran_order':\s*True/.test(hdr);
  const count = Number(hdr.match(/'shape':\s*\((\d+)/)[1]);
  if (!descr.includes("f4") || fortran) throw new Error("expect <f4 c-order");
  return new Float32Array(b.buffer, b.byteOffset + 10 + hlen, count);
}

// in-place radix-2 complex FFT
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len, half = len >> 1;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < half; k++) {
        const c = Math.cos(ang * k), s = Math.sin(ang * k);
        const ur = re[i + k], ui = im[i + k];
        const xr = re[i + k + half], xi = im[i + k + half];
        const vr = xr * c - xi * s, vi = xr * s + xi * c;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + half] = ur - vr; im[i + k + half] = ui - vi;
      }
    }
  }
}

const wasmPath = process.env.FE_WASM ?? join("out", `fastenhancer_${tier}.wasm`);
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath));
const e = instance.exports;
const inF = e.fe_in_ptr() >>> 2;
const outF = e.fe_out_ptr() >>> 2;
e.fe_init();

const x = loadNpyF32("reference_x.npy");
const yRef = loadNpyF32("reference_y.npy");
const win = new Float64Array(N_FFT);
for (let i = 0; i < N_FFT; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N_FFT);

const padded = new Float64Array(x.length + HOP);
padded.set(x);
const nFrames = 1 + Math.floor((padded.length - N_FFT) / HOP);
const enh = new Float64Array(padded.length);
const wsum = new Float64Array(padded.length);
const re = new Float64Array(N_FFT), im = new Float64Array(N_FFT);

const t0 = performance.now();
for (let f = 0; f < nFrames; f++) {
  const off = f * HOP;
  re.fill(0); im.fill(0);
  for (let i = 0; i < N_FFT; i++) re[i] = padded[off + i] * win[i];
  fft(re, im);

  let mem = new Float32Array(e.memory.buffer);
  for (let b = 0; b <= 512; b++) { mem[inF + 2 * b] = re[b]; mem[inF + 2 * b + 1] = im[b]; }
  e.fe_run();
  mem = new Float32Array(e.memory.buffer);
  for (let b = 0; b <= 512; b++) { re[b] = mem[outF + 2 * b]; im[b] = mem[outF + 2 * b + 1]; }

  // hermitian mirror + inverse fft (swap trick: ifft = fft(conj)/N, output real)
  for (let b = 513; b < N_FFT; b++) { re[b] = re[N_FFT - b]; im[b] = -im[N_FFT - b]; }
  for (let b = 0; b < N_FFT; b++) im[b] = -im[b];
  fft(re, im);
  for (let i = 0; i < N_FFT; i++) {
    enh[off + i] += (re[i] / N_FFT) * win[i];
    wsum[off + i] += win[i] * win[i];
  }
}
const dt = performance.now() - t0;

let maxDiff = 0, sumSq = 0;
const n = yRef.length;
for (let i = 0; i < n; i++) {
  const v = Math.max(-1, Math.min(1, enh[i] / Math.max(wsum[i], 1e-8)));
  const d = Math.abs(v - yRef[i]);
  maxDiff = Math.max(maxDiff, d);
  sumSq += d * d;
}
const rms = Math.sqrt(sumSq / n);
console.log(`tier=${tier} frames=${nFrames} JS RTF=${((dt / 1000) * SR / (nFrames * HOP)).toFixed(4)} (incl. JS FFT)`);
console.log(`max |wasm - onnxruntime| = ${maxDiff.toExponential(3)}`);
console.log(`rms  diff                = ${rms.toExponential(3)}`);
console.log(maxDiff < 2e-3 ? "PASS — worklet-grade equivalence" : "CHECK — larger than expected");
if (maxDiff >= 2e-3) process.exit(1);
