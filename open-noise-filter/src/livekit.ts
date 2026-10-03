// Copyright 2026 YaseenHQ
// SPDX-License-Identifier: Apache-2.0

/**
 * open-noise-filter — LiveKit adapter.
 *
 * Implements the livekit-client TrackProcessor contract:
 *
 *   import { LiveKitNoiseFilter } from "open-noise-filter/livekit";
 *   await track.setProcessor(LiveKitNoiseFilter({ quality: "high" }));
 *
 * All DSP happens in the AudioWorklet built by createNoiseFilter(); this
 * adapter only manages the LiveKit lifecycle: track constraints, the
 * source → worklet → destination graph, and the published processedTrack.
 */
import type { AudioProcessorOptions, Track, TrackProcessor } from "livekit-client";
import { createNoiseFilter, isNoiseFilterSupported } from "./index.js";
import type { NoiseFilter, NoiseFilterOptions } from "./index.js";

export interface LiveKitNoiseFilterOptions extends Omit<NoiseFilterOptions, "onStats"> {
  debugLogs?: boolean;
}

const log = (debug: boolean, ...args: unknown[]) => {
  if (debug) console.log("[open-noise-filter]", ...args);
};

export class NoiseFilterProcessor implements TrackProcessor<Track.Kind.Audio, AudioProcessorOptions> {
  readonly name = "open-noise-filter";
  processedTrack?: MediaStreamTrack;

  private readonly opts: LiveKitNoiseFilterOptions;
  private track?: MediaStreamTrack;
  private enabled = true;
  private destroyed = false;
  private originalConstraints?: MediaTrackConstraints;
  private source?: MediaStreamAudioSourceNode;
  private dest?: MediaStreamAudioDestinationNode;
  private filter?: NoiseFilter;

  constructor(options: LiveKitNoiseFilterOptions = {}) {
    this.opts = options;
    this.enabled = options.enabled ?? true;
  }

  static isSupported(): boolean {
    return isNoiseFilterSupported();
  }

  init = async (opts: AudioProcessorOptions): Promise<void> => {
    await this.build(opts);
  };

  restart = async (opts: AudioProcessorOptions): Promise<void> => {
    const oldTrack = this.track;
    await this.teardownGraph();
    await this.restoreConstraints(oldTrack); // best effort on the old track
    await this.build(opts);
  };

  onPublish = async (): Promise<void> => {};
  onUnpublish = async (): Promise<void> => {};

  /** Bypass/enable the DSP without rebuilding the graph or the published track. */
  setEnabled = async (enable: boolean): Promise<boolean | undefined> => {
    this.enabled = enable;
    this.filter?.setEnabled(enable);
    return this.enabled;
  };

  isEnabled = (): boolean => this.enabled;

  destroy = async (): Promise<void> => {
    if (this.destroyed) return;
    await this.restoreConstraints(this.track); // before teardown clears fields
    await this.teardownGraph();
    this.destroyed = true;
  };

  private async applyConstraints(track: MediaStreamTrack): Promise<void> {
    try {
      this.originalConstraints = track.getConstraints();
      await track.applyConstraints({
        ...this.originalConstraints,
        noiseSuppression: false, // ours replaces it; keep the user's AEC/AGC
      });
    } catch (e) {
      log(!!this.opts.debugLogs, "constraint apply skipped:", e);
    }
  }

  private async restoreConstraints(track?: MediaStreamTrack): Promise<void> {
    if (!track || !this.originalConstraints) return;
    try {
      await track.applyConstraints(this.originalConstraints);
    } catch { /* best effort */ }
  }

  private async build(opts: AudioProcessorOptions): Promise<void> {
    if (this.destroyed) return;
    this.track = opts.track;
    try {
      await this.buildGraph(opts);
    } catch (e) {
      await this.teardownGraph(); // no dangling source/dest/node on failed init
      throw e;
    }
  }

  private async buildGraph(opts: AudioProcessorOptions): Promise<void> {
    const ctx = opts.audioContext;
    const debug = !!this.opts.debugLogs;
    const { urls, quality, thread, maxChannels, gate } = this.opts;

    this.source = ctx.createMediaStreamSource(new MediaStream([opts.track]));
    this.dest = ctx.createMediaStreamDestination();
    this.filter = await createNoiseFilter(ctx, {
      quality,
      thread,
      maxChannels,
      gate,
      enabled: this.enabled,
      urls,
      onStats: (s) =>
        log(debug, `worker stats: ${s.underruns} underruns, ${s.overruns} overruns, ${s.skips} skips, target ${s.target}`),
    });
    // only touch the source track once the engine is confirmed alive — a failed
    // init must leave the track's constraints (and the track itself) untouched
    await this.applyConstraints(opts.track);
    this.source.connect(this.filter.node);
    this.filter.node.connect(this.dest);
    this.processedTrack = this.dest.stream.getAudioTracks()[0];
    log(debug, `open-noise-filter ${this.filter.quality} online at ${ctx.sampleRate} Hz (${this.filter.thread} thread)`);
  }

  private async teardownGraph(): Promise<void> {
    try { this.filter?.destroy(); } catch { /* gone */ }
    this.filter = undefined;
    try { this.source?.disconnect(); } catch { /* not connected */ }
    try { this.processedTrack?.stop(); } catch { /* already stopped */ }
    this.source = undefined;
    this.dest = undefined;
    this.processedTrack = undefined;
    this.track = undefined;
  }
}

/** Factory: `setProcessor(LiveKitNoiseFilter({ quality: "high" }))`. */
export const LiveKitNoiseFilter = (options: LiveKitNoiseFilterOptions = {}) =>
  new NoiseFilterProcessor(options);
export const isLiveKitNoiseFilterSupported = NoiseFilterProcessor.isSupported;
