// Copyright 2026 YaseenHQ
// SPDX-License-Identifier: Apache-2.0

//! Multi-channel convenience: one state per channel, interleaved or
//! planar access. Mirrors the worklet's behavior — 48 kHz channels emit
//! zero-filled output for the fixed 1024-sample startup latency; resampled
//! channels have the same causal, block-independent timeline through adapter.

use crate::engine::{DenoiseState, Tier, H};
use crate::resample::ResampledDenoiseState;
use std::collections::VecDeque;

/// The frame engine needs two complete hops before it can return its first
/// output. Keeping that output queued establishes a fixed 1024-sample delay.
const STARTUP_DELAY: usize = 2 * H;

enum Channel {
    /// 48 kHz: hops are accumulated internally, so blocks can be any
    /// length; emitted output carries the engine's algorithmic delay.
    Direct {
        eng: DenoiseState,
        hop_in: Vec<f32>,
        fill: usize,
        hop_out: [f32; H],
        out_q: VecDeque<f32>,
    },
    Resampled(ResampledDenoiseState),
}

/// `channels` independent noise suppressors with a shared tier and rate.
pub struct DenoiseChannels {
    states: Vec<Channel>,
    scratch: Vec<f32>,
}

impl DenoiseChannels {
    /// `sample_rate` other than 48000 runs each channel through the
    /// Catmull-Rom adapter. Rates must be finite and in 8–192 kHz, and at
    /// least one channel is required.
    ///
    /// # Panics
    /// Panics if `channels` is zero or `sample_rate` is outside that range.
    pub fn new(tier: Tier, channels: usize, sample_rate: f64) -> Self {
        assert!(channels > 0, "channels must be greater than zero");
        assert!(
            sample_rate.is_finite() && (8_000.0..=192_000.0).contains(&sample_rate),
            "sample_rate must be finite and between 8000 and 192000 Hz"
        );
        let states = (0..channels)
            .map(|_| {
                if (sample_rate - 48000.0).abs() < f64::EPSILON {
                    Channel::Direct {
                        eng: DenoiseState::new(tier),
                        hop_in: vec![0.0; H],
                        fill: 0,
                        hop_out: [0.0; H],
                        out_q: VecDeque::from(vec![0.0; STARTUP_DELAY]),
                    }
                } else {
                    Channel::Resampled(ResampledDenoiseState::new(tier, sample_rate))
                }
            })
            .collect();
        DenoiseChannels {
            states,
            scratch: Vec::new(),
        }
    }

    pub fn channels(&self) -> usize {
        self.states.len()
    }

    pub fn reset(&mut self) {
        for s in &mut self.states {
            match s {
                Channel::Direct {
                    eng, fill, out_q, ..
                } => {
                    eng.reset();
                    *fill = 0;
                    out_q.clear();
                    out_q.extend(std::iter::repeat(0.0).take(STARTUP_DELAY));
                }
                Channel::Resampled(r) => r.reset(),
            }
        }
    }

    /// Process one channel slice in place. Length is unrestricted.
    pub fn process_channel(&mut self, chan: usize, buf: &mut [f32]) {
        match &mut self.states[chan] {
            Channel::Direct {
                eng,
                hop_in,
                fill,
                hop_out,
                out_q,
            } => {
                // Advance input and output together so a large call cannot
                // expose frames to earlier samples in that same call.
                for sample in buf.iter_mut() {
                    hop_in[*fill] = *sample;
                    *fill += 1;
                    if *fill == H {
                        *fill = 0;
                        let in_arr: &[f32; H] = hop_in.as_slice().try_into().unwrap();
                        if eng.process_frame(in_arr, hop_out) {
                            out_q.extend(hop_out.iter());
                        }
                    }
                    *sample = out_q.pop_front().unwrap_or(0.0);
                }
            }
            Channel::Resampled(r) => r.process_in_place(buf),
        }
    }

    /// Process `buf` in place, `channels`-way interleaved. Length must be a
    /// multiple of the channel count.
    pub fn process_interleaved(&mut self, buf: &mut [f32]) {
        let nch = self.states.len();
        assert_eq!(buf.len() % nch, 0);
        let n = buf.len() / nch;
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.resize(n, 0.0);
        for c in 0..nch {
            for i in 0..n {
                scratch[i] = buf[i * nch + c];
            }
            self.process_channel(c, &mut scratch);
            for i in 0..n {
                buf[i * nch + c] = scratch[i];
            }
        }
        self.scratch = scratch;
    }

    /// Process per-channel slices in place; all must share a length.
    pub fn process_planar(&mut self, chans: &mut [&mut [f32]]) {
        assert_eq!(chans.len(), self.states.len());
        if let Some(first) = chans.first() {
            assert!(
                chans.iter().all(|buf| buf.len() == first.len()),
                "all planar channel slices must have the same length"
            );
        }
        for (c, buf) in chans.iter_mut().enumerate() {
            self.process_channel(c, buf);
        }
    }
}
