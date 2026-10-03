// Copyright 2026 YaseenHQ
// SPDX-License-Identifier: Apache-2.0

/* Stateful wrapper over onnx2c entry() — Tiny tier (2 caches, 24x20).
 * NOTE: normalized graphs order outputs [cache_out_0, cache_out_1, spec_out]. */
typedef float Spec[513][1][2];
typedef float CacheT[24][20];
void entry(const Spec *, const CacheT *, const CacheT *, CacheT *, CacheT *, Spec *);
static CacheT ci0, ci1, co0, co1;

void fe_init(void) {
    for (int i = 0; i < 24 * 20; i++) {
        ((float *)ci0)[i] = 0.0f;
        ((float *)ci1)[i] = 0.0f;
    }
}

void fe_step(const float *in, float *out) {
    entry((const Spec *)in, (const CacheT *)ci0, (const CacheT *)ci1,
          (CacheT *)co0, (CacheT *)co1, (Spec *)out);
    for (int i = 0; i < 24 * 20; i++) {
        ((float *)ci0)[i] = ((float *)co0)[i];
        ((float *)ci1)[i] = ((float *)co1)[i];
    }
}

/* malloc-free JS interface: JS writes into the in-buffer, calls fe_run(),
 * reads the out-buffer via the pointer getters. */
static float fe_in_b[513 * 2], fe_out_b[513 * 2];
float *fe_in_ptr(void) { return fe_in_b; }
float *fe_out_ptr(void) { return fe_out_b; }
void fe_run(void) { fe_step(fe_in_b, fe_out_b); }
