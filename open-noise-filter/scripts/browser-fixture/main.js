import { createNoiseFilter } from "open-noise-filter";
import { LiveKitNoiseFilter } from "open-noise-filter/livekit";
import worklet from "open-noise-filter/worklet.js?url";
import worker from "open-noise-filter/worker.js?url";
import tiny from "open-noise-filter/wasm/fastenhancer_t.wasm?url";
import base from "open-noise-filter/wasm/fastenhancer_b.wasm?url";
import small from "open-noise-filter/wasm/fastenhancer_s.wasm?url";
import tinyScalar from "open-noise-filter/wasm/fastenhancer_t_scalar.wasm?url";
import baseScalar from "open-noise-filter/wasm/fastenhancer_b_scalar.wasm?url";
import smallScalar from "open-noise-filter/wasm/fastenhancer_s_scalar.wasm?url";

const binaries = { low: tiny, medium: base, high: small };
const scalarBinaries = { low: tinyScalar, medium: baseScalar, high: smallScalar };
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const urlsFor = (quality, scalar) => ({
  worklet, worker,
  wasm: (scalar ? scalarBinaries : binaries)[quality],
  scalarWasm: scalarBinaries[quality],
});

async function sourceFor(ctx) {
  const bytes = await (await fetch("/reference.f32")).arrayBuffer();
  const samples = new Float32Array(bytes);
  const buffer = ctx.createBuffer(1, samples.length, 48000);
  buffer.copyToChannel(samples, 0);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.loop = true;
  return source;
}

async function readSignal(analyser) {
  let peak = 0;
  const samples = new Float32Array(analyser.fftSize);
  for (let i = 0; i < 8; i++) {
    await wait(50);
    analyser.getFloatTimeDomainData(samples);
    for (const value of samples) {
      if (!Number.isFinite(value)) throw new Error("Non-finite output");
      peak = Math.max(peak, Math.abs(value));
    }
  }
  if (peak === 0) throw new Error("Filter produced only silence");
  return peak;
}

window.runFilterSmoke = async ({ quality, sampleRate = 48000, maxChannels = 1, scalar = false, thread }) => {
  const ctx = new AudioContext({ sampleRate });
  let filter;
  let source;
  const stats = [];
  try {
    await ctx.resume();
    source = await sourceFor(ctx);
    filter = await createNoiseFilter(ctx, {
      quality, maxChannels, thread, urls: urlsFor(quality, scalar),
      onStats: (value) => stats.push(value),
    });
    const analyser = ctx.createAnalyser();
    const mute = ctx.createGain();
    mute.gain.value = 0;
    source.connect(filter.node);
    filter.node.connect(analyser).connect(mute).connect(ctx.destination);
    source.start();
    await wait(600);
    const peak = await readSignal(analyser);
    filter.setEnabled(false);
    await readSignal(analyser);
    filter.setEnabled(true);
    await wait(300);
    await readSignal(analyser);
    await ctx.suspend();
    await wait(50);
    await ctx.resume();
    await readSignal(analyser);
    return { quality: filter.quality, thread: filter.thread, sampleRate: ctx.sampleRate, maxChannels, scalar, peak, stats };
  } finally {
    source?.stop();
    filter?.destroy();
    filter?.destroy();
    await ctx.close();
  }
};

window.runLivekitSmoke = async () => {
  const ctx = new AudioContext({ sampleRate: 48000 });
  const processor = LiveKitNoiseFilter({ quality: "medium", enabled: false, urls: urlsFor("medium") });
  let source;
  let track;
  try {
    await ctx.resume();
    source = await sourceFor(ctx);
    const capture = ctx.createMediaStreamDestination();
    source.connect(capture);
    source.start();
    track = capture.stream.getAudioTracks()[0];
    await processor.init({ kind: "audio", track, audioContext: ctx });
    if (processor.isEnabled()) throw new Error("Initial enabled:false was ignored");
    if (processor.processedTrack?.readyState !== "live") throw new Error("No live processed track");
    await processor.setEnabled(true);
    const previous = processor.processedTrack;
    await processor.restart({ kind: "audio", track, audioContext: ctx });
    if (previous.readyState !== "ended") throw new Error("Restart leaked the old processed track");
    if (processor.processedTrack?.readyState !== "live") throw new Error("Restart did not create a live track");
    const current = processor.processedTrack;
    await processor.destroy();
    await processor.destroy();
    if (current.readyState !== "ended") throw new Error("Destroy leaked the processed track");
    if (track.readyState !== "live") throw new Error("Destroy stopped the source track");
    return { enabledOption: true, restart: true, destroy: true };
  } finally {
    await processor.destroy();
    source?.stop();
    track?.stop();
    await ctx.close();
  }
};
