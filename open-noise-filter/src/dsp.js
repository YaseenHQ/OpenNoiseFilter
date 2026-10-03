// Copyright 2026 YaseenHQ
// SPDX-License-Identifier: Apache-2.0

/**
 * open-noise-filter — shared DSP: radix-2 FFT and FrameEngine, the
 * streaming STFT → FastEnhancer WASM → iSTFT → overlap-add pipeline used
 * by worklet.js and worker.js (esbuild-bundled; no runtime imports).
 *
 * Mirrors scripts/reference.py: raw complex STFT in (power compression is
 * inside the model), clamp in/out to [-1, 1], periodic hann, frames =
 * [hop f-1, hop f] with first-hop priming.
 */
export const N = 1024, H = 512;

export function makeFFT(n) {
  const rev = new Uint32Array(n);
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    rev[i] = j;
  }
  // per-stage twiddles; identical values to the inline Math.cos/sin calls
  const stages = [];
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1, c = new Float64Array(half), s = new Float64Array(half);
    const ang = (-2 * Math.PI) / len;
    for (let k = 0; k < half; k++) { c[k] = Math.cos(ang * k); s[k] = Math.sin(ang * k); }
    stages.push({ c, s });
  }
  return function fft(re, im) {
    for (let i = 1; i < n; i++) {
      const j = rev[i];
      if (i < j) { const tr = re[i]; re[i] = re[j]; re[j] = tr; const ti = im[i]; im[i] = im[j]; im[j] = ti; }
    }
    let st = 0;
    for (let len = 2; len <= n; len <<= 1, st++) {
      const half = len >> 1, tc = stages[st].c, ts = stages[st].s;
      for (let i = 0; i < n; i += len) {
        for (let k = 0; k < half; k++) {
          const c = tc[k], s = ts[k];
          const ur = re[i + k], ui = im[i + k];
          const xr = re[i + k + half], xi = im[i + k + half];
          const vr = xr * c - xi * s, vi = xr * s + xi * c;
          re[i + k] = ur + vr; im[i + k] = ui + vi;
          re[i + k + half] = ur - vr; im[i + k + half] = ui - vi;
        }
      }
    }
  };
}

/**
 * Wraps a FastEnhancer wasm instance's exports (fe_init/fe_run/fe_in_ptr/
 * fe_out_ptr) and owns all frame state: window, hop priming, OLA accumulators,
 * FFT scratch, and the cached memory view.
 */
export class FrameEngine {
  constructor(exports) {
    this.e = exports;
    this.inF = this.e.fe_in_ptr() >>> 2;
    this.outF = this.e.fe_out_ptr() >>> 2;
    this.fft = makeFFT(N);
    this.win = new Float64Array(N);
    for (let i = 0; i < N; i++) this.win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N); // periodic hann
    this.hopPrev = new Float32Array(H);
    this.hopPrimed = false; // first hop only primes hopPrev; no [0,hop0] frame
    this.acc = new Float64Array(N); this.wacc = new Float64Array(N);
    this.re = new Float64Array(N); this.im = new Float64Array(N);
    this.mem = new Float32Array(this.e.memory.buffer);
    this.e.fe_init();
  }

  reset() {
    this.hopPrimed = false;
    this.hopPrev.fill(0);
    this.acc.fill(0); this.wacc.fill(0);
    this.e.fe_init();
  }

  memF32() {
    const b = this.e.memory.buffer;
    if (this.mem.buffer !== b) this.mem = new Float32Array(b);
    return this.mem;
  }

  /**
   * Consume one 512-sample hop of raw (unclamped) 48 kHz input.
   * Returns false for the priming hop; when true, `out` holds 512 finished
   * samples (normalized, clamped to [-1, 1]). `out` may alias `hopIn`.
   */
  processHop(hopIn, out) {
    const re = this.re, win = this.win, hp = this.hopPrev;
    if (!this.hopPrimed) {
      for (let i = 0; i < H; i++) { const v = hopIn[i]; hp[i] = v > 1 ? 1 : v < -1 ? -1 : v; }
      this.hopPrimed = true;
      return false;
    }
    // frame = [hopPrev, hop]
    for (let i = 0; i < H; i++) re[i] = hp[i] * win[i];
    for (let i = 0; i < H; i++) {
      const v = hopIn[i], c = v > 1 ? 1 : v < -1 ? -1 : v;
      re[H + i] = c * win[H + i];
      hp[i] = c;
    }
    this.im.fill(0);
    this.fft(re, this.im);

    let mem = this.memF32();
    for (let b = 0; b <= H; b++) { mem[this.inF + 2 * b] = re[b]; mem[this.inF + 2 * b + 1] = this.im[b]; }
    this.e.fe_run();
    mem = this.memF32(); // fe_run may have grown memory
    for (let b = 0; b <= H; b++) { re[b] = mem[this.outF + 2 * b]; this.im[b] = mem[this.outF + 2 * b + 1]; }

    // hermitian mirror + inverse fft (conj trick: ifft = fft(conj)/N, real output)
    for (let b = H + 1; b < N; b++) { re[b] = re[N - b]; this.im[b] = -this.im[N - b]; }
    for (let b = 0; b < N; b++) this.im[b] = -this.im[b];
    this.fft(re, this.im);

    for (let i = 0; i < N; i++) {
      this.acc[i] += (re[i] / N) * win[i];
      this.wacc[i] += win[i] * win[i];
    }
    for (let i = 0; i < H; i++) {
      out[i] = Math.max(-1, Math.min(1, this.acc[i] / Math.max(this.wacc[i], 1e-8)));
    }
    this.acc.copyWithin(0, H); this.acc.fill(0, H);
    this.wacc.copyWithin(0, H); this.wacc.fill(0, H);
    return true;
  }
}
