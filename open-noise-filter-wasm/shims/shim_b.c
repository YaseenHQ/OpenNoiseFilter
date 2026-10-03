// Copyright 2026 YaseenHQ
// SPDX-License-Identifier: Apache-2.0

/* Stateful wrapper over onnx2c entry() — Base tier (3 caches, 36x36). */
typedef float Spec[513][1][2];
typedef float CacheT[36][36];
void entry(const Spec *, const CacheT *, const CacheT *, const CacheT *, CacheT *, CacheT *, CacheT *, Spec *);
static CacheT ci0, ci1, ci2, co0, co1, co2;

void fe_init(void) {
    for (int i = 0; i < 36 * 36; i++) {
        ((float *)ci0)[i] = 0.0f;
        ((float *)ci1)[i] = 0.0f;
        ((float *)ci2)[i] = 0.0f;
    }
}

void fe_step(const float *in, float *out) {
    entry((const Spec *)in, (const CacheT *)ci0, (const CacheT *)ci1, (const CacheT *)ci2,
          (CacheT *)co0, (CacheT *)co1, (CacheT *)co2, (Spec *)out);
    for (int i = 0; i < 36 * 36; i++) {
        ((float *)ci0)[i] = ((float *)co0)[i];
        ((float *)ci1)[i] = ((float *)co1)[i];
        ((float *)ci2)[i] = ((float *)co2)[i];
    }
}

static float fe_in_b[513 * 2], fe_out_b[513 * 2];
float *fe_in_ptr(void) { return fe_in_b; }
float *fe_out_ptr(void) { return fe_out_b; }
void fe_run(void) { fe_step(fe_in_b, fe_out_b); }
