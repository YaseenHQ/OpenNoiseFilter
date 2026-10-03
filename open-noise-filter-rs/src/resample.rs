//! Context-rate adapter around the 48 kHz frame engine.
//!
//! Input is interpolated at the internal rate and output is interpolated back
//! at the context rate. Samples are handled in stream order, so call/block
//! boundaries do not change the output timeline. Startup output is held until
//! two internal hops are available; this gives the resampler enough queued
//! output to cover its fixed interpolation lookahead and the engine priming.

use crate::engine::{DenoiseState, Tier, H};

// Small circular input history is enough because interpolation consumes input
// in order and waits for three future samples before each internal sample.
const XIN: usize = 16;
const XIN_MASK: usize = XIN - 1;
const RING: usize = 4096; // 48 kHz output ring (power of two)
const RING_MASK: usize = RING - 1;
const START_LEAD: usize = 2 * H;

#[inline]
fn catmull(p0: f32, p1: f32, p2: f32, p3: f32, t: f64) -> f32 {
    let t = t as f32;
    p1 + 0.5
        * t
        * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0)))
}

/// One channel of context-rate audio around a 48 kHz `DenoiseState`.
///
/// `process` consumes `input.len()` context-rate samples and writes the same
/// count to `out`. It has a fixed startup delay (about 32 ms including the
/// two-hop lead) followed by a continuous stream. Splitting the same input
/// into different block sizes produces identical output.
pub struct ResampledDenoiseState {
    eng: DenoiseState,
    ratio: f64, // context / 48000
    xin: [f32; XIN],
    in_write: usize,
    in_read: f64,
    hop_acc: [f32; H],
    hop_fill: usize,
    hop_out: [f32; H],
    ring: [f32; RING],
    produced: usize,
    out_read: f64,
}

impl ResampledDenoiseState {
    /// Create an adapter for a context rate from 8 kHz through 192 kHz.
    ///
    /// # Panics
    /// Panics if `sample_rate` is non-finite or outside the supported range.
    pub fn new(tier: Tier, sample_rate: f64) -> Self {
        assert!(
            sample_rate.is_finite() && (8_000.0..=192_000.0).contains(&sample_rate),
            "sample_rate must be finite and between 8000 and 192000 Hz"
        );
        ResampledDenoiseState {
            eng: DenoiseState::new(tier),
            ratio: sample_rate / 48000.0,
            xin: [0.0; XIN],
            in_write: 0,
            in_read: 0.0,
            hop_acc: [0.0; H],
            hop_fill: 0,
            hop_out: [0.0; H],
            ring: [0.0; RING],
            produced: 0,
            out_read: 0.0,
        }
    }

    /// Drop all stream state and restore the original startup delay.
    pub fn reset(&mut self) {
        self.eng.reset();
        self.in_write = 0;
        self.in_read = 0.0;
        self.hop_fill = 0;
        self.produced = 0;
        self.out_read = 0.0;
    }

    /// Process a block of context-rate audio. Block boundaries do not affect
    /// latency or sample values.
    pub fn process(&mut self, input: &[f32], out: &mut [f32]) {
        assert_eq!(input.len(), out.len());
        for (&sample, output) in input.iter().zip(out.iter_mut()) {
            *output = self.process_sample(sample);
        }
    }

    /// In-place variant. Each input sample is saved before its output is
    /// written, so input/output aliasing is safe.
    pub fn process_in_place(&mut self, buf: &mut [f32]) {
        for sample in buf.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    fn process_sample(&mut self, sample: f32) -> f32 {
        self.xin[self.in_write & XIN_MASK] = sample;
        self.in_write += 1;

        // Process only the input time that is available at this point in the
        // stream. The three-sample lookahead is fixed and independent of the
        // caller's block size.
        while self.in_read + 3.0 < self.in_write as f64 {
            let j = self.in_read.floor() as usize;
            let t = self.in_read - j as f64;
            let p0 = if j >= 1 { self.input_at(j - 1) } else { 0.0 };
            self.hop_acc[self.hop_fill] = catmull(
                p0,
                self.input_at(j),
                self.input_at(j + 1),
                self.input_at(j + 2),
                t,
            );
            self.in_read += self.ratio;
            self.hop_fill += 1;
            if self.hop_fill == H {
                self.hop_fill = 0;
                if self.eng.process_frame(&self.hop_acc, &mut self.hop_out) {
                    for i in 0..H {
                        self.ring[(self.produced + i) & RING_MASK] = self.hop_out[i];
                    }
                    self.produced += H;
                }
            }
        }

        self.emit_sample()
    }

    #[inline]
    fn input_at(&self, absolute_index: usize) -> f32 {
        self.xin[absolute_index & XIN_MASK]
    }

    fn emit_sample(&mut self) -> f32 {
        // Hold the output cursor at zero until the fixed startup lead exists.
        // Thereafter producer and consumer advance at the same time ratio.
        if self.produced < START_LEAD {
            return 0.0;
        }
        let j = self.out_read.floor() as usize;
        if j + 2 >= self.produced {
            // This is only reachable for an unsupported rate or corrupted
            // stream state. Preserve the timeline instead of freezing it.
            self.out_read += 1.0 / self.ratio;
            return 0.0;
        }
        let t = self.out_read - j as f64;
        let p0 = if j >= 1 {
            self.ring[(j - 1) & RING_MASK]
        } else {
            0.0
        };
        let result = catmull(
            p0,
            self.ring[j & RING_MASK],
            self.ring[(j + 1) & RING_MASK],
            self.ring[(j + 2) & RING_MASK],
            t,
        );
        self.out_read += 1.0 / self.ratio;
        result
    }
}
