//! open-noise-filter-rs — FastEnhancer neural noise suppression, native.
//!
//! Same model and pipeline as the `open-noise-filter` Web Audio package:
//! the onnx2c-generated model C is vendored in `csrc/` (built by the
//! `open-noise-filter-wasm` tools repo) and compiled at build time; the
//! streaming STFT → model → iSTFT → overlap-add front-end is a port of the
//! library's `src/dsp.js` `FrameEngine`.
//!
//! Contract: 48 kHz, mono per `DenoiseState`, 512-sample hops. The first
//! `process_frame` call after `new()`/`reset()` only primes the window and
//! returns `false`; after that each call returns `true` with 512 finished
//! samples. Create one `DenoiseState` per channel for stereo. `Send` is
//! safe; concurrent calls across instances are serialized internally
//! (the generated C uses static scratch).
//!
//! ```no_run
//! use open_noise_filter::{DenoiseState, Tier};
//! let mut st = DenoiseState::new(Tier::Base);
//! let input = [0.0f32; 512];
//! let mut out = [0.0f32; 512];
//! assert!(!st.process_frame(&input, &mut out)); // priming hop
//! ```

mod channels;
mod engine;
mod fft;
mod resample;
mod sys;

pub use channels::DenoiseChannels;
pub use engine::{DenoiseState, Tier, H, N};
pub use resample::ResampledDenoiseState;
