/**
 * open-noise-filter — AudioWorkletProcessor (FastEnhancer tiers + gate).
 *
 * processorOptions: { wasm?, enabled?, channels?: 1|2,
 *                     mode: "audio" | "worker" | "gate",
 *                     gate?: { openThreshold?, closeThreshold?, holdMs? } }
 *
 * "audio":  DSP in process(); latency 896 samples. One wasm instance per
 *           channel — the shim keeps static state (one instance = one stream).
 * "worker": one Worker per channel behind an adaptive jitter buffer; latency
 *           1408 mono / 1920 stereo, +512 per underrun (max 2432).
 * "gate":   pure-JS gate, zero latency, any sample rate.
 *
 * Port protocol: 'ready' once live; 'setEnabled' = copy-through bypass
 * (re-enable resets DSP state); 'destroy' → process() returns false;
 * worker mode posts {type:"stats", underruns, overruns, skips, target}.
 */
import { N, H, FrameEngine } from "./dsp.js";

const RING = 4096, RING_MASK = RING - 1; // 48 kHz output ring (power of two)
const XIN = 8192;                        // context-rate input history cap
const POOL = 8;                          // in-flight hop buffers per channel

class FastEnhancerProcessor extends AudioWorkletProcessor {
  constructor(opts) {
    super();
    const o = (opts && opts.processorOptions) || {};
    this.ready = false;
    this.destroyed = false;
    this.enabled = o.enabled === undefined ? true : !!o.enabled;
    this.mode = o.mode === "worker" || o.mode === "gate" ? o.mode : "audio";
    this.C = o.channels === 2 ? 2 : 1;
    // per-channel state (shared cursors keep channels time-aligned)
    this.hopAcc = []; this.scratch = []; this.ring = []; this.xin = [];
    this.producedC = []; this.engines = [];
    for (let c = 0; c < this.C; c++) {
      this.hopAcc.push(new Float32Array(H));
      this.scratch.push(new Float32Array(H));
      this.ring.push(new Float32Array(RING));
      this.xin.push(new Float32Array(XIN));
      this.producedC.push(0);
    }
    this.hopFill = 0;
    this.firstFrameDone = false;
    this.consumed = 0;                     // 48k path: absolute read cursor
    // non-48 kHz contexts: Catmull-Rom resampler around the DSP
    this.ratio = sampleRate / 48000;       // context / 48000
    this.inWrite = 0;                      // input history write index
    this.inRead = 0;                       // context-domain read cursor (fractional)
    this.outRead = 0;                      // ring-domain output cursor (fractional)
    // worker-mode hop transport + jitter buffer (per channel)
    this.wports = []; this.pools = []; this.inflight = [];
    for (let c = 0; c < this.C; c++) {
      this.wports.push(null); this.inflight.push(0);
      const pool = [];
      for (let i = 0; i < POOL; i++) pool.push(new Float32Array(H));
      this.pools.push(pool);
    }
    this.gen = 0;
    this.streaming = false;                // jitter buffer: emitting vs rebuffering
    this.outStarted = false;               // resampler path (audio mode): past start lead
    // worker jitter-buffer target fill (48k samples); stereo starts deeper —
    // ready gates on the slower of two workers, 2H underrun at startup
    this.target = this.C === 2 ? 3 * H : 2 * H;
    this.underruns = 0; this.overruns = 0; this.skips = 0;
    this.statQuanta = 0;
    this.lastStats = { underruns: -1, overruns: -1, skips: -1, target: -1 };
    // resampler output start lead (audio mode only; worker mode uses target)
    this.startLead = 2 * H;
    // gate mode: hysteresis state + one-pole gain smoother
    const g = o.gate || {};
    this.openTh = g.openThreshold === undefined ? -45 : g.openThreshold;
    this.closeTh = g.closeThreshold === undefined ? -55 : g.closeThreshold;
    this.holdMs = g.holdMs === undefined ? 200 : g.holdMs;
    this.gateGain = 1;                     // starts open — no clipped first words
    this.gateTarget = 1;
    this.belowMs = 0;
    this.atk = 1 - Math.exp(-1 / (0.005 * sampleRate));
    this.rel = 1 - Math.exp(-1 / (0.05 * sampleRate));

    if (this.mode === "audio") {
      // synchronous compile+instantiate: no bypass window, and the async
      // instantiate() promise never settles in some AudioWorklet environments
      try {
        const mod = new WebAssembly.Module(o.wasm);
        for (let c = 0; c < this.C; c++) {
          this.engines.push(new FrameEngine(new WebAssembly.Instance(mod).exports));
        }
        this.ready = true;
        this.port.postMessage({ type: "ready" });
      } catch (err) {
        this.port.postMessage({ type: "error", error: String(err) });
      }
    } else if (this.mode === "gate") {
      this.ready = true;
      this.port.postMessage({ type: "ready" });
    }
    this.port.onmessage = (ev) => {
      const d = ev.data;
      if (!d) return;
      if (d.type === "setEnabled") {
        const v = !!d.value;
        if (v && !this.enabled) this.resetDsp(); // no stale audio on re-enable
        this.enabled = v;
      } else if (d.type === "destroy") {
        this.destroyed = true;
      } else if (d.type === "port" && this.mode === "worker") {
        this.setPort(0, d.port);
      } else if (d.type === "ports" && this.mode === "worker") {
        for (let c = 0; c < this.C; c++) this.setPort(c, d.ports[c]);
      }
    };
  }

  setPort(c, port) {
    if (this.wports[c]) return; // first binding wins
    this.wports[c] = port;
    port.onmessage = (e2) => this.onWorkerMessage(e2.data, c);
    if (this.wports.every(Boolean) && !this.ready) {
      this.ready = true;
      this.port.postMessage({ type: "ready" });
    }
  }

  get produced() { // committed production visible to readers: slowest channel
    return this.C === 2 ? Math.min(this.producedC[0], this.producedC[1]) : this.producedC[0];
  }

  get readCursor() {
    // Resampled output reads in fractional 48 kHz ring coordinates. Keep one
    // preceding sample for Catmull-Rom interpolation when deciding what is
    // still live in the ring.
    return this.ratio === 1 ? this.consumed : Math.max(0, Math.floor(this.outRead) - 1);
  }

  advanceReadCursor(cursor) {
    if (this.ratio === 1) this.consumed = cursor;
    else this.outRead = cursor;
  }

  onWorkerMessage(d, c) {
    if (!d || d.type !== "out") return;
    this.pools[c].push(d.buf); // buffer always returns to the pool
    if (this.inflight[c] > 0) this.inflight[c]--;
    if (d.gen === this.gen && d.produced) {
      // A write must never lap unread samples. `consumed` only advances on
      // the native 48 kHz path; resampled readers use the fractional outRead
      // cursor, so guard against the cursor that actually reads this ring.
      const writeEnd = this.producedC[c] + H;
      if (writeEnd - this.readCursor > RING) {
        // Keep the target fill plus Catmull-Rom's preceding history sample.
        // The reader cursor is shared by stereo channels to preserve alignment.
        const next = Math.max(1, writeEnd - this.target);
        this.advanceReadCursor(next);
        this.skips++;
      }
      const ring = this.ring[c];
      for (let i = 0; i < H; i++) ring[(this.producedC[c] + i) & RING_MASK] = d.buf[i];
      this.producedC[c] += H;
      this.firstFrameDone = true;
    }
  }

  resetDsp() {
    this.hopFill = 0;
    this.firstFrameDone = false;
    for (let c = 0; c < this.C; c++) {
      this.hopAcc[c].fill(0);
      this.ring[c].fill(0);
      this.producedC[c] = 0;
    }
    this.consumed = 0;
    this.inWrite = 0; this.inRead = 0; this.outRead = 0;
    this.streaming = false; this.outStarted = false;
    this.inflight.fill(0);
    this.target = this.C === 2 ? 3 * H : 2 * H;
    if (this.mode === "worker") {
      // stale replies from the old generation are ignored by gen check
      this.gen++;
      for (const p of this.wports) if (p) p.postMessage({ type: "reset", gen: this.gen });
    } else if (this.ready) {
      for (const e of this.engines) e.reset();
    }
  }

  submitHop() {
    if (this.mode === "worker") {
      for (let c = 0; c < this.C; c++) {
        if (this.pools[c].length === 0) {
          // a worker is >8 hops behind: drop everything and resync all channels
          this.overruns++;
          this.gen++;
          this.resetLocal();
          for (const p of this.wports) if (p) p.postMessage({ type: "reset", gen: this.gen });
          return;
        }
      }
      for (let c = 0; c < this.C; c++) {
        const buf = this.pools[c].pop();
        buf.set(this.hopAcc[c]); // hopAcc holds clamped samples
        this.inflight[c]++;
        this.wports[c].postMessage({ type: "hop", gen: this.gen, buf }, [buf.buffer]);
      }
      return;
    }
    let done = true;
    for (let c = 0; c < this.C; c++) {
      if (this.engines[c].processHop(this.hopAcc[c], this.scratch[c])) {
        const ring = this.ring[c];
        for (let i = 0; i < H; i++) ring[(this.producedC[c] + i) & RING_MASK] = this.scratch[c][i];
        this.producedC[c] += H;
      } else done = false;
    }
    if (done) this.firstFrameDone = true;
  }

  resetLocal() {
    this.hopFill = 0;
    this.firstFrameDone = false;
    for (let c = 0; c < this.C; c++) {
      this.hopAcc[c].fill(0);
      this.ring[c].fill(0);
      this.producedC[c] = 0;
    }
    this.consumed = 0;
    this.inWrite = 0; this.inRead = 0; this.outRead = 0;
    this.streaming = false; this.outStarted = false;
    this.inflight.fill(0);
    this.target = this.C === 2 ? 3 * H : 2 * H;
  }

  process(inputs, outputs) {
    if (this.destroyed) return false;
    const outs = outputs[0];
    const out0 = outs && outs[0];
    if (!out0) return true;
    const ins = inputs[0];
    const inp0 = ins && ins[0];

    if (!this.ready || !this.enabled || !inp0) {
      for (let c = 0; c < outs.length; c++) {
        const ic = ins ? (ins[c] || ins[0]) : null;
        if (ic) outs[c].set(ic.subarray(0, outs[c].length));
        else outs[c].fill(0);
      }
      return true;
    }

    if (this.mode === "gate") {
      this.renderGate(outs, ins);
      return true;
    }

    this.render(outs, ins);

    // worker-mode stats, once per second of context-rate audio
    if (this.mode === "worker" && ++this.statQuanta * 128 >= sampleRate) {
      this.statQuanta = 0;
      if (this.underruns !== this.lastStats.underruns || this.overruns !== this.lastStats.overruns ||
          this.skips !== this.lastStats.skips || this.target !== this.lastStats.target) {
        this.lastStats = { underruns: this.underruns, overruns: this.overruns, skips: this.skips, target: this.target };
        this.port.postMessage({ type: "stats", underruns: this.underruns, overruns: this.overruns, skips: this.skips, target: this.target });
      }
    }
    return true;
  }

  renderGate(outs, ins) {
    // level = max per-channel RMS over the quantum, in dBFS — linked gating
    let lv = -Infinity;
    const Q = outs[0].length;
    for (let c = 0; c < outs.length; c++) {
      const ic = ins[c] || ins[0];
      let e = 0;
      for (let i = 0; i < Q; i++) e += ic[i] * ic[i];
      const db = 10 * Math.log10(e / Q + 1e-12);
      if (db > lv) lv = db;
    }
    if (lv >= this.openTh) {
      this.gateTarget = 1;
      this.belowMs = 0;
    } else if (lv < this.closeTh) {
      this.belowMs += (Q * 1000) / sampleRate;
      if (this.belowMs >= this.holdMs) this.gateTarget = 0;
    } else {
      this.belowMs = 0; // between thresholds: hysteresis — keep the state
    }
    for (let i = 0; i < Q; i++) {
      const k = this.gateTarget > this.gateGain ? this.atk : this.rel;
      this.gateGain += k * (this.gateTarget - this.gateGain);
      for (let c = 0; c < outs.length; c++) {
        outs[c][i] = (ins[c] || ins[0])[i] * this.gateGain;
      }
    }
  }

  render(outs, ins) {
    const C = this.C;
    const src = (c) => ins[c] || ins[0];
    if (this.ratio === 1) {
      // native 48 kHz path: one 128-sample quantum per callback (clamped to
      // [-1, 1], matching the reference pipeline exactly)
      const qf = this.hopFill;
      for (let c = 0; c < C; c++) {
        const ic = src(c), acc = this.hopAcc[c];
        for (let i = 0; i < 128; i++) {
          const v = ic[i];
          acc[qf + i] = v > 1 ? 1 : v < -1 ? -1 : v;
        }
      }
      this.hopFill = qf + 128;
      if (this.hopFill >= H) {
        this.hopFill = 0;
        this.submitHop();
      }

      let avail = this.produced - this.consumed;
      if (avail < 0) avail = 0;
      if (this.mode === "worker") {
        // adaptive jitter buffer: resume at `target`; underruns deepen it
        // (max 4H), over-fill drops oldest — latency can't creep upward
        let minInflight = this.inflight[0];
        for (let c = 1; c < C; c++) minInflight = Math.min(minInflight, this.inflight[c]);
        if (!this.streaming && avail + minInflight * H >= this.target) this.streaming = true;
        if (this.streaming && avail > this.target + H) {
          this.consumed = this.produced - this.target;
          avail = this.target;
          this.skips++;
        }
        if (this.streaming && avail >= 128) {
          for (let c = 0; c < C; c++) {
            const ring = this.ring[c], out = outs[c];
            for (let i = 0; i < 128; i++) out[i] = ring[(this.consumed + i) & RING_MASK];
          }
          this.consumed += 128;
        } else {
          for (let c = 0; c < C; c++) outs[c].fill(0);
          if (this.streaming) {
            this.streaming = false;
            this.underruns++;
            this.target = Math.min(this.target + H, 4 * H);
          }
        }
      } else if (this.firstFrameDone && avail >= 128) {
        for (let c = 0; c < C; c++) {
          const ring = this.ring[c], out = outs[c];
          for (let i = 0; i < 128; i++) out[i] = ring[(this.consumed + i) & RING_MASK];
        }
        this.consumed += 128;
      } else {
        for (let c = 0; c < C; c++) outs[c].fill(0); // algorithmic latency (~18.7 ms)
      }
      return;
    }

    // non-48 kHz: Catmull-Rom resample around the DSP
    // 1) consume context-rate input, emit internal 48 kHz samples into the hop
    const nx = Math.min(128, XIN - this.inWrite);
    for (let i = 0; i < nx; i++) {
      for (let c = 0; c < C; c++) this.xin[c][this.inWrite + i] = src(c)[i];
    }
    this.inWrite += nx;
    while (this.inRead + 3 < this.inWrite) {
      const j = Math.floor(this.inRead), t = this.inRead - j;
      for (let c = 0; c < C; c++) {
        const xin = this.xin[c];
        const p0 = j >= 1 ? xin[j - 1] : 0, p1 = xin[j], p2 = xin[j + 1], p3 = xin[j + 2];
        this.hopAcc[c][this.hopFill] =
          p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)));
      }
      this.inRead += this.ratio;
      this.hopFill++;
      if (this.hopFill === H) {
        this.hopFill = 0;
        this.submitHop();
      }
    }
    if (this.inRead > 4096) {
      const drop = Math.floor(this.inRead) - 3;
      for (let c = 0; c < C; c++) this.xin[c].copyWithin(0, drop, this.inWrite);
      this.inWrite -= drop;
      this.inRead -= drop;
    }

    // 2) emit context-rate output by interpolating the 48 kHz ring.
    if (this.mode === "worker") {
      // start/resume at target + H so the smooth reader keeps a hop of margin
      // against blocky 512-sample production; over-fill rewinds the cursor
      const produced = this.produced;
      if (!this.streaming && produced - this.outRead >= this.target + H) this.streaming = true;
      let stalled = false;
      for (let i = 0; i < 128; i++) {
        const j = Math.floor(this.outRead), t = this.outRead - j;
        if (this.streaming && j + 2 < produced) {
          for (let c = 0; c < C; c++) {
            const ring = this.ring[c];
            const p0 = j >= 1 ? ring[(j - 1) & RING_MASK] : 0;
            const p1 = ring[j & RING_MASK], p2 = ring[(j + 1) & RING_MASK], p3 = ring[(j + 2) & RING_MASK];
            outs[c][i] = p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)));
          }
          this.outRead += 1 / this.ratio;
        } else {
          for (let c = 0; c < C; c++) outs[c][i] = 0;
          if (this.streaming) { this.streaming = false; stalled = true; }
        }
      }
      if (stalled) {
        this.underruns++;
        this.target = Math.min(this.target + H, 4 * H);
      }
      if (this.streaming && this.produced - this.outRead > this.target + 2 * H) {
        this.outRead = this.produced - (this.target + H);
        this.skips++;
      }
      return;
    }
    // audio mode: startLead keeps a frame of margin between the smooth
    // consumption cursor and blocky 512/frame production
    let stalled = false;
    for (let i = 0; i < 128; i++) {
      const j = Math.floor(this.outRead), t = this.outRead - j;
      if (this.produced >= this.startLead && j + 2 < this.produced) {
        this.outStarted = true;
        for (let c = 0; c < C; c++) {
          const ring = this.ring[c];
          const p0 = j >= 1 ? ring[(j - 1) & RING_MASK] : 0;
          const p1 = ring[j & RING_MASK], p2 = ring[(j + 1) & RING_MASK], p3 = ring[(j + 2) & RING_MASK];
          outs[c][i] = p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)));
        }
        this.outRead += 1 / this.ratio;
      } else {
        for (let c = 0; c < C; c++) outs[c][i] = 0; // startup latency / rebuffering
        if (this.outStarted) stalled = true;
      }
    }
    if (stalled) this.underruns++;
  }
}

registerProcessor("fastenhancer-worklet", FastEnhancerProcessor);
