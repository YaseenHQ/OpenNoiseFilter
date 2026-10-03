//! FrameEngine port: streaming STFT → FastEnhancer (onnx2c C) → iSTFT →
//! overlap-add. Mirrors the JS `FrameEngine` in `src/dsp.js` exactly:
//! raw complex STFT in (power compression is inside the model), input
//! clamped to [-1, 1], periodic Hann, frames = [hop f-1, hop f] with
//! first-hop priming, Hermitian-mirror inverse FFT, Hann OLA.

use crate::fft::Fft;
use crate::sys;
use std::sync::Mutex;

pub const N: usize = 1024;
pub const H: usize = 512;

/// Which compiled-in FastEnhancer checkpoint to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// FastEnhancer Tiny — cheapest, ~0.5 ms/frame on this dev machine.
    Tiny,
    /// FastEnhancer Base — default balance.
    Base,
    /// FastEnhancer Small — best quality, ~4.3 ms/frame.
    Small,
}

// spec is always [1,513,1,2] complex = 1026 floats; cache geometry comes
// from the checkpoint (see open-noise-filter-wasm/shims/)
const SPEC_LEN: usize = 1026;
fn cache_layout(tier: Tier) -> (usize, usize) {
    match tier {
        Tier::Tiny => (2, 24 * 20),
        Tier::Base => (3, 36 * 36),
        Tier::Small => (3, 48 * 48),
    }
}

// generated `entry` keeps scratch tensors in `static` — not thread-safe
static ENTRY_LOCK: Mutex<()> = Mutex::new(());

/// Streaming noise suppressor for one 48 kHz channel.
///
/// ```no_run
/// let mut st = open_noise_filter::DenoiseState::new(open_noise_filter::Tier::Base);
/// let mut out = [0f32; 512];
/// // first call primes the window and returns false
/// let produced = st.process_frame(&[0f32; 512], &mut out);
/// ```
pub struct DenoiseState {
    tier: Tier,
    ci: Vec<Vec<f32>>,
    co: Vec<Vec<f32>>,
    spec_in: Vec<f32>,
    spec_out: Vec<f32>,
    fft: Fft,
    win: Vec<f64>,
    hop_prev: Vec<f32>,
    hop_primed: bool,
    acc: Vec<f64>,
    wacc: Vec<f64>,
    re: Vec<f64>,
    im: Vec<f64>,
}

impl DenoiseState {
    pub fn new(tier: Tier) -> Self {
        let (n_caches, cache_len) = cache_layout(tier);
        let win: Vec<f64> = (0..N)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N as f64).cos())
            .collect();
        DenoiseState {
            tier,
            ci: vec![vec![0.0; cache_len]; n_caches],
            co: vec![vec![0.0; cache_len]; n_caches],
            spec_in: vec![0.0; SPEC_LEN],
            spec_out: vec![0.0; SPEC_LEN],
            fft: Fft::new(N),
            win,
            hop_prev: vec![0.0; H],
            hop_primed: false,
            acc: vec![0.0; N],
            wacc: vec![0.0; N],
            re: vec![0.0; N],
            im: vec![0.0; N],
        }
    }

    /// Drop all stream state (caches, OLA, hop priming) — like the worklet's
    /// re-enable reset.
    pub fn reset(&mut self) {
        self.hop_primed = false;
        for c in self.ci.iter_mut().chain(self.co.iter_mut()) {
            c.fill(0.0);
        }
        self.hop_prev.fill(0.0);
        self.acc.fill(0.0);
        self.wacc.fill(0.0);
    }

    /// Consume one 512-sample hop of 48 kHz audio. Returns false for the
    /// priming hop; when true, `out` holds 512 finished samples clamped to
    /// [-1, 1]. `out` may not alias `input`.
    pub fn process_frame(&mut self, input: &[f32; H], out: &mut [f32; H]) -> bool {
        if !self.hop_primed {
            for i in 0..H {
                self.hop_prev[i] = input[i].clamp(-1.0, 1.0);
            }
            self.hop_primed = true;
            return false;
        }
        let re = &mut self.re;
        let win = &self.win;
        for i in 0..H {
            re[i] = self.hop_prev[i] as f64 * win[i];
            let c = input[i].clamp(-1.0, 1.0);
            re[H + i] = c as f64 * win[H + i];
            self.hop_prev[i] = c;
        }
        self.im.fill(0.0);
        self.fft.run(re, &mut self.im);

        for b in 0..=H {
            self.spec_in[2 * b] = re[b] as f32;
            self.spec_in[2 * b + 1] = self.im[b] as f32;
        }
        {
            let _g = ENTRY_LOCK.lock().unwrap();
            sys::run_entry(
                self.tier,
                &self.spec_in,
                &self.ci,
                &mut self.co,
                &mut self.spec_out,
            );
        }
        // mirror the C shims: cache_out becomes next call's cache_in
        for (ci, co) in self.ci.iter_mut().zip(self.co.iter()) {
            ci.copy_from_slice(co);
        }
        for b in 0..=H {
            re[b] = self.spec_out[2 * b] as f64;
            self.im[b] = self.spec_out[2 * b + 1] as f64;
        }

        // hermitian mirror + inverse fft (conj trick: ifft = fft(conj)/N)
        for b in (H + 1)..N {
            re[b] = re[N - b];
            self.im[b] = -self.im[N - b];
        }
        for b in 0..N {
            self.im[b] = -self.im[b];
        }
        self.fft.run(re, &mut self.im);

        for i in 0..N {
            self.acc[i] += (re[i] / N as f64) * win[i];
            self.wacc[i] += win[i] * win[i];
        }
        for i in 0..H {
            out[i] = (self.acc[i] / self.wacc[i].max(1e-8)).clamp(-1.0, 1.0) as f32;
        }
        self.acc.copy_within(H.., 0);
        self.acc[H..].fill(0.0);
        self.wacc.copy_within(H.., 0);
        self.wacc[H..].fill(0.0);
        true
    }
}
