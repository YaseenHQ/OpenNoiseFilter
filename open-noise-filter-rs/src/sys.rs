// Copyright 2026 YaseenHQ
// SPDX-License-Identifier: Apache-2.0

//! FFI to the per-tier onnx2c `entry` functions (symbol-renamed at build
//! time to fe_t_entry / fe_b_entry / fe_s_entry). Signature per tier:
//! entry(spec_in, cache_in_0..K, cache_out_0..K, spec_out) with K = 2 for
//! Tiny (24×20 caches) and K = 3 for Base (36×36) / Small (48×48).
//! All float arrays are flat; the C code indexes them as multidimensional
//! arrays, which decay to pointers at the ABI boundary.

use crate::engine::Tier;

extern "C" {
    fn fe_t_entry(
        spec_in: *const f32,
        ci0: *const f32,
        ci1: *const f32,
        co0: *mut f32,
        co1: *mut f32,
        spec_out: *mut f32,
    );
    fn fe_b_entry(
        spec_in: *const f32,
        ci0: *const f32,
        ci1: *const f32,
        ci2: *const f32,
        co0: *mut f32,
        co1: *mut f32,
        co2: *mut f32,
        spec_out: *mut f32,
    );
    fn fe_s_entry(
        spec_in: *const f32,
        ci0: *const f32,
        ci1: *const f32,
        ci2: *const f32,
        co0: *mut f32,
        co1: *mut f32,
        co2: *mut f32,
        spec_out: *mut f32,
    );
}

/// Run one model frame. Caller must hold ENTRY_LOCK — the generated code
/// uses static scratch tensors and is not thread-safe.
pub fn run_entry(
    tier: Tier,
    spec_in: &[f32],
    ci: &[Vec<f32>],
    co: &mut [Vec<f32>],
    spec_out: &mut [f32],
) {
    unsafe {
        match tier {
            Tier::Tiny => fe_t_entry(
                spec_in.as_ptr(),
                ci[0].as_ptr(),
                ci[1].as_ptr(),
                co[0].as_mut_ptr(),
                co[1].as_mut_ptr(),
                spec_out.as_mut_ptr(),
            ),
            Tier::Base => fe_b_entry(
                spec_in.as_ptr(),
                ci[0].as_ptr(),
                ci[1].as_ptr(),
                ci[2].as_ptr(),
                co[0].as_mut_ptr(),
                co[1].as_mut_ptr(),
                co[2].as_mut_ptr(),
                spec_out.as_mut_ptr(),
            ),
            Tier::Small => fe_s_entry(
                spec_in.as_ptr(),
                ci[0].as_ptr(),
                ci[1].as_ptr(),
                ci[2].as_ptr(),
                co[0].as_mut_ptr(),
                co[1].as_mut_ptr(),
                co[2].as_mut_ptr(),
                spec_out.as_mut_ptr(),
            ),
        }
    }
}
