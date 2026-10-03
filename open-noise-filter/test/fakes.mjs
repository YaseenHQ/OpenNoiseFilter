/** Minimal Web Audio / DOM fakes for testing src/index.ts + src/livekit.ts
 * in plain node. installFakes() sets the globals the library touches;
 * uninstallFakes() restores them. The bundled code only touches these at
 * call time, never at module top level. */

export function installFakes({ wasmBytes }) {
  const saved = {};
  const fakes = {
    warnings: [],
    nodes: [],
    workers: [],
    log: [],            // ordered event log: {ev: "..."} entries
    modules: [],        // addModule URLs
    fetchCalls: [],
    workletReply: "ready",  // set to "error" to make the node fail init
    workerReply: "ready",   // same for the Worker
    fetchFails: false,
    constraintsApplied: [], // [track, constraints] pairs
  };
  const save = (k) => { saved[k] = globalThis[k]; };

  save("window"); globalThis.window = globalThis;
  save("WebAssembly"); // leave real WebAssembly alone (present in node)
  save("console");
  const origWarn = console.warn;
  console.warn = (...a) => { fakes.warnings.push(a.join(" ")); };
  fakes._origWarn = origWarn;

  class FakeWorkletNode {
    constructor(ctx, name, opts) {
      this.ctx = ctx; this.name = name;
      this.options = opts || {};
      this.processorOptions = (opts && opts.processorOptions) || {};
      this.connectedTo = [];
      const events = fakes.log;
      this.port = {
        onmessage: null,
        posted: [],
        postMessage: (m) => {
          this.port.posted.push(m);
          // worker mode: "ready" follows the {type:"ports"} handoff;
          // audio mode: "ready" was already settled at construction
          if (m.type === "port" || m.type === "ports") queueMicrotask(() =>
            this.port.onmessage && this.port.onmessage({
              data: fakes.workletReply === "error"
                ? { type: "error", error: "worklet init failed (fake)" }
                : { type: fakes.workletReply },
            }));
        },
      };
      // audio and gate modes post "ready" at construction; worker mode posts
      // it in reply to the {type:"ports"} handoff (handled in postMessage)
      if (this.processorOptions.wasm || this.processorOptions.mode === "gate") queueMicrotask(() =>
        this.port.onmessage && this.port.onmessage({
          data: fakes.workletReply === "error"
            ? { type: "error", error: "worklet init failed (fake)" }
            : { type: fakes.workletReply },
        }));
      fakes.nodes.push(this);
      events.push({ ev: "node", name });
    }
    connect(dest) { this.connectedTo.push(dest); fakes.log.push({ ev: "connect" }); return dest; }
    disconnect() { fakes.log.push({ ev: "disconnect" }); }
  }

  class FakeWorker {
    constructor(url) {
      this.url = url; this.onmessage = null; this.posted = []; this.terminated = false;
      fakes.workers.push(this);
    }
    postMessage(m) {
      this.posted.push(m);
      if (m.type === "init") queueMicrotask(() =>
        this.onmessage && this.onmessage({
          data: fakes.workerReply === "error"
            ? { type: "error", error: "worker init failed (fake)" }
            : { type: fakes.workerReply },
        }));
    }
    terminate() { this.terminated = true; fakes.log.push({ ev: "terminate" }); }
  }

  class FakeOfflineAudioContext { }

  save("AudioWorkletNode"); globalThis.AudioWorkletNode = FakeWorkletNode;
  save("Worker"); globalThis.Worker = FakeWorker;
  save("OfflineAudioContext"); globalThis.OfflineAudioContext = FakeOfflineAudioContext;
  save("MediaStream");
  globalThis.MediaStream = class MediaStream {
    constructor(tracks) { this.tracks = tracks; }
    getAudioTracks() { return this.tracks; }
  };
  save("fetch");
  globalThis.fetch = async (url) => {
    fakes.fetchCalls.push(String(url));
    if (fakes.fetchFails) return { ok: false, status: 404, statusText: "Not Found" };
    return { ok: true, arrayBuffer: async () => wasmBytes.slice(0) };
  };

  fakes.makeTrack = (label = "track") => {
    const orig = { echoCancellation: true, noiseSuppression: true };
    return {
      label,
      getConstraints: () => orig,
      applyConstraints: async (c) => {
        fakes.constraintsApplied.push([label, c]);
        fakes.log.push({ ev: "applyConstraints" });
      },
    };
  };

  fakes.makeCtx = (sampleRate = 48000) => ({
    sampleRate,
    audioWorklet: { addModule: async (u) => { fakes.modules.push(String(u)); } },
    createMediaStreamSource: (stream) => {
      fakes.log.push({ ev: "createSource" });
      return {
        connect(n) { fakes.log.push({ ev: "sourceConnect" }); return n; },
        disconnect() { fakes.log.push({ ev: "sourceDisconnect" }); },
      };
    },
    createMediaStreamDestination: () => {
      const track = { stop() { fakes.log.push({ ev: "trackStop" }); } };
      return { stream: new globalThis.MediaStream([track]) };
    },
  });

  fakes.makeOfflineCtx = (sampleRate = 48000) =>
    Object.assign(new FakeOfflineAudioContext(), {
      sampleRate,
      audioWorklet: { addModule: async (u) => { fakes.modules.push(String(u)); } },
    });

  fakes.uninstall = () => {
    for (const k of Object.keys(saved)) {
      if (saved[k] === undefined) delete globalThis[k];
      else globalThis[k] = saved[k];
    }
    console.warn = origWarn;
  };
  return fakes;
}
