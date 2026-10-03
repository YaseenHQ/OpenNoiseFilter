// Copyright 2026 YaseenHQ
// SPDX-License-Identifier: Apache-2.0

//! Golden-vector test: the native port must track the WASM build's output
//! bit-for-bit-ish. tests/data/y_{t,b,s}.f32 are produced by running
//! test/reference_x.f32 through the shipped WASM worklet (see README);
//! tolerance accounts for native vs wasm FP codegen (fast-math both sides).

use open_noise_filter::{DenoiseChannels, DenoiseState, ResampledDenoiseState, Tier, H};

fn f32s(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// the WASM reference includes the worklet's 896-sample algorithmic delay;
// the native engine emits its stream without that leading padding
const WORKLET_DELAY: usize = 896;

fn check(tier: Tier, name: &str) {
    let x = f32s("tests/data/x.f32");
    let y_ref = f32s(&format!("tests/data/y_{}.f32", name));
    let mut st = DenoiseState::new(tier);
    let mut produced = 0usize;
    let mut max_diff = 0f32;
    let mut out = [0f32; H];
    for (i, hop) in x.chunks_exact(H).enumerate() {
        let input: &[f32; H] = hop.try_into().unwrap();
        let primed = st.process_frame(input, &mut out);
        if i == 0 {
            assert!(!primed, "first hop must only prime");
            continue;
        }
        assert!(primed);
        for (k, &o) in out.iter().enumerate() {
            let idx = produced * H + k + WORKLET_DELAY;
            if idx < y_ref.len() {
                max_diff = max_diff.max((o - y_ref[idx]).abs());
            }
        }
        produced += 1;
    }
    println!("{}: max|rust-wasm| = {:.3e}", name, max_diff);
    assert!(
        max_diff < 1e-4,
        "{} diverges from the wasm build: {}",
        name,
        max_diff
    );
}

#[test]
fn golden_tiny() {
    check(Tier::Tiny, "t");
}
#[test]
fn golden_base() {
    check(Tier::Base, "b");
}
#[test]
fn golden_small() {
    check(Tier::Small, "s");
}

#[test]
fn two_instances_independent() {
    let x = f32s("tests/data/x.f32");
    let mut a = DenoiseState::new(Tier::Tiny);
    let mut b = DenoiseState::new(Tier::Tiny);
    let mut oa = [0f32; H];
    let mut ob = [0f32; H];
    for (i, hop) in x.chunks_exact(H).take(4).enumerate() {
        let input: &[f32; H] = hop.try_into().unwrap();
        let pa = a.process_frame(input, &mut oa);
        let pb = b.process_frame(input, &mut ob);
        assert_eq!(pa, pb);
        if pa {
            assert_eq!(oa, ob, "frame {}: identical inputs must match", i);
        }
    }
}

#[test]
fn reset_replays_identically() {
    let x = f32s("tests/data/x.f32");
    let mut st = DenoiseState::new(Tier::Tiny);
    let mut first = Vec::new();
    let mut out = [0f32; H];
    for hop in x.chunks_exact(H).take(8) {
        let input: &[f32; H] = hop.try_into().unwrap();
        if st.process_frame(input, &mut out) {
            first.extend_from_slice(&out);
        }
    }
    st.reset();
    let mut second = Vec::new();
    for hop in x.chunks_exact(H).take(8) {
        let input: &[f32; H] = hop.try_into().unwrap();
        if st.process_frame(input, &mut out) {
            second.extend_from_slice(&out);
        }
    }
    assert_eq!(first, second);
}

// The golden clip was captured from the 44.1 kHz worklet at 128-sample
// cadence. Causal scheduling adds a fixed five-context-sample alignment to
// that reference. Skip the 2048-sample startup/priming interval, then compare
// at the measured alignment with the original tight numeric tolerance.
#[test]
fn golden_resampled_44k() {
    let x = f32s("tests/data/x_44k.f32");
    let y_ref = f32s("tests/data/y_44k_b.f32");
    let mut st = ResampledDenoiseState::new(Tier::Base, 44100.0);
    // the worklet reference only covered complete 128-sample quanta
    let n = x.len() / 128 * 128;
    let mut y = x[..n].to_vec();
    for blk in y.chunks_mut(128) {
        st.process_in_place(blk);
    }
    const CAUSAL_ALIGNMENT: usize = 5;
    let max_diff = y
        .iter()
        .skip(2048 + CAUSAL_ALIGNMENT)
        .zip(y_ref.iter().skip(2048))
        .fold(0.0f32, |max_diff, (&actual, &expected)| {
            max_diff.max((actual - expected).abs())
        });
    println!("44k: max|rust-wasm| = {:.3e}", max_diff);
    assert!(max_diff < 1e-4, "resampled path diverges: {}", max_diff);
}

#[test]
fn channels_interleaved_matches_mono() {
    let x = f32s("tests/data/x.f32");
    let n = x.len() / 4;
    let left = x[..n].to_vec();
    let right = x[n..2 * n].to_vec();

    let mut stereo = DenoiseChannels::new(Tier::Tiny, 2, 48000.0);
    let mut interleaved = Vec::with_capacity(2 * n);
    for i in 0..n {
        interleaved.push(left[i]);
        interleaved.push(right[i]);
    }
    stereo.process_interleaved(&mut interleaved);

    let mut mono_l = DenoiseChannels::new(Tier::Tiny, 1, 48000.0);
    let mut mono_r = DenoiseChannels::new(Tier::Tiny, 1, 48000.0);
    let mut ml = left.clone();
    let mut mr = right.clone();
    mono_l.process_interleaved(&mut ml);
    mono_r.process_interleaved(&mut mr);

    for i in 0..n {
        assert_eq!(interleaved[2 * i], ml[i], "left sample {}", i);
        assert_eq!(interleaved[2 * i + 1], mr[i], "right sample {}", i);
    }
}

#[test]
fn channels_planar_and_leading_delay() {
    let x = f32s("tests/data/x.f32");
    let n = 4096;
    let mut c = DenoiseChannels::new(Tier::Tiny, 2, 48000.0);
    let mut a = x[..n].to_vec();
    let mut b = x[..n].to_vec();
    assert!(a.iter().all(|&v| v != 0.0 || a[0] == 0.0));
    let mut bufs: [&mut [f32]; 2] = [&mut a, &mut b];
    c.process_planar(&mut bufs);
    let (a, b) = (bufs[0].to_vec(), bufs[1].to_vec());
    assert!(a[..2 * H].iter().all(|&v| v == 0.0));
    assert_eq!(a, b, "identical channels must stay identical");
    assert!(
        a.iter().any(|&v| v != 0.0),
        "engine must produce non-zero output by sample {}",
        n
    );
}

fn resampled_with_block_size(input: &[f32], block_size: usize) -> Vec<f32> {
    let mut state = ResampledDenoiseState::new(Tier::Tiny, 44100.0);
    let mut output = input.to_vec();
    for block in output.chunks_mut(block_size) {
        state.process_in_place(block);
    }
    output
}

#[test]
fn resampled_stream_is_invariant_to_block_size_and_long_tail() {
    // Longer than the former 8192-sample input history cap. The one-sample
    // stream is the reference for preserving every sample and each timeline
    // position; other block sizes must produce exactly the same stream.
    let input = f32s("tests/data/x_44k.f32")[..24017].to_vec();
    let samplewise = resampled_with_block_size(&input, 1);
    for block_size in [64, 128, 512, input.len()] {
        let output = resampled_with_block_size(&input, block_size);
        if let Some((index, (expected, actual))) = samplewise
            .iter()
            .zip(output.iter())
            .enumerate()
            .find(|(_, (expected, actual))| expected != actual)
        {
            panic!("resampled output changed with block size {block_size} at sample {index}: expected {expected}, got {actual}");
        }
    }
    assert!(samplewise.iter().all(|sample| sample.is_finite()));
    assert!(samplewise[12_000..].iter().any(|&sample| sample != 0.0));
}

#[test]
fn resampler_supported_rate_endpoints_preserve_chunks_and_finite_output() {
    let input = f32s("tests/data/x_44k.f32")[..8193].to_vec();
    for sample_rate in [8000.0, 192000.0] {
        let samplewise = {
            let mut state = ResampledDenoiseState::new(Tier::Tiny, sample_rate);
            let mut output = input.clone();
            for sample in output.chunks_mut(1) {
                state.process_in_place(sample);
            }
            output
        };
        for block_size in [128, input.len()] {
            let mut state = ResampledDenoiseState::new(Tier::Tiny, sample_rate);
            let mut output = input.clone();
            for block in output.chunks_mut(block_size) {
                state.process_in_place(block);
            }
            assert_eq!(samplewise, output, "chunking changed at {sample_rate} Hz");
        }
        assert!(samplewise.iter().all(|sample| sample.is_finite()));
        assert!(samplewise.iter().any(|&sample| sample != 0.0));
    }
}

fn channels_with_block_size(input: &[f32], sample_rate: f64, block_size: usize) -> Vec<f32> {
    let mut state = DenoiseChannels::new(Tier::Tiny, 1, sample_rate);
    let mut output = input.to_vec();
    for block in output.chunks_mut(block_size) {
        state.process_channel(0, block);
    }
    output
}

#[test]
fn channel_timeline_is_invariant_to_block_size() {
    let input = f32s("tests/data/x.f32")[..8193].to_vec();
    for sample_rate in [48000.0, 44100.0] {
        let samplewise = channels_with_block_size(&input, sample_rate, 1);
        for block_size in [64, 128, 512, input.len()] {
            let output = channels_with_block_size(&input, sample_rate, block_size);
            if let Some((index, (expected, actual))) = samplewise
                .iter()
                .zip(output.iter())
                .enumerate()
                .find(|(_, (expected, actual))| expected != actual)
            {
                panic!("channel timeline changed at {sample_rate} Hz with block size {block_size} at sample {index}: expected {expected}, got {actual}");
            }
        }
    }
}

#[test]
fn direct_channels_match_golden_with_documented_128_sample_alignment() {
    let input = f32s("tests/data/x.f32");
    let reference = f32s("tests/data/y_t.f32");
    let mut output = input.clone();
    let mut channels = DenoiseChannels::new(Tier::Tiny, 1, 48000.0);
    channels.process_channel(0, &mut output);

    // The reference worklet pads 896 samples; this causal channel adapter
    // queues output after two complete hops (1024 samples), a fixed +128.
    let mut max_diff = 0.0f32;
    for (&actual, &expected) in output
        .iter()
        .skip(1024)
        .zip(reference.iter().skip(WORKLET_DELAY))
    {
        max_diff = max_diff.max((actual - expected).abs());
    }
    assert!(
        max_diff < 1e-4,
        "direct channel golden diverges: {max_diff}"
    );
}

#[test]
fn channels_reset_replays_full_startup_delay() {
    let input = f32s("tests/data/x.f32")[..4096].to_vec();
    for sample_rate in [48000.0, 44100.0] {
        let mut state = DenoiseChannels::new(Tier::Tiny, 1, sample_rate);
        let mut warm = input.clone();
        state.process_channel(0, &mut warm);
        state.reset();
        let mut after_reset = input.clone();
        state.process_channel(0, &mut after_reset);
        let fresh = channels_with_block_size(&input, sample_rate, input.len());
        assert_eq!(
            after_reset, fresh,
            "reset did not restore startup at {sample_rate} Hz"
        );
        assert!(after_reset[..896].iter().all(|&sample| sample == 0.0));
        if sample_rate == 48000.0 {
            assert!(after_reset[..1024].iter().all(|&sample| sample == 0.0));
        }
    }
}

#[test]
#[should_panic(expected = "channels must be greater than zero")]
fn channels_rejects_zero_channels() {
    let _ = DenoiseChannels::new(Tier::Tiny, 0, 48000.0);
}

#[test]
#[should_panic(expected = "sample_rate must be finite")]
fn channels_rejects_invalid_sample_rate() {
    let _ = DenoiseChannels::new(Tier::Tiny, 1, f64::NAN);
}
