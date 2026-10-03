# wasm/ — prebuilt FastEnhancer binaries

These files are the compiled FastEnhancer neural nets. They are **prebuilt
artifacts**, distributed as-is; the model build pipeline lives in the
sibling `open-noise-filter-wasm` component (`node build.mjs` there
regenerates them). No model-build toolchain is needed to consume them.

`*_scalar.wasm` are the same models built without `-msimd128`, for browsers
lacking WebAssembly SIMD; the library picks them at runtime.

## Provenance

FastEnhancer `onnx-48khz-v1` `*_spec.onnx` checkpoints
([github.com/aask1357/fastenhancer](https://github.com/aask1357/fastenhancer),
MIT, [arXiv:2509.21867](https://arxiv.org/abs/2509.21867)) →

1. GRU/Gather/Gemm/Conv lowering to MatMul/elementwise soup
   (each step ORT-validated to ~1e-6)
2. graph normalization
3. [onnx2c](https://github.com/kraiskil/onnx2c) C codegen — four upstream
   bugs were found and worked around along the way:
   <https://github.com/kraiskil/onnx2c/issues/133>
4. union→struct de-aliasing pass (onnx2c liveness-bug workaround)
5. hand-optimized ikj fast-matmul rewrite
6. `emcc -O3 -msimd128 -ffast-math` (STANDALONE_WASM) with a shim exporting
   `fe_init` / `fe_run` / `fe_in_ptr` / `fe_out_ptr` — scalar builds drop
   `-msimd128`

## Hashes (sha256)

| file | sha256 |
|------|--------|
| `fastenhancer_t.wasm` | `5ce2fbbaa37bca5a7b08338b256062fb86fed5e84effed2739f78ead0d52f1de` |
| `fastenhancer_b.wasm` | `cadee7ba6b26c2947d732dfeed0e440da07fb0a58b2d09c8893193a8ddbf038b` |
| `fastenhancer_s.wasm` | `69bd3c46481a5f1cf8d972ccc036e0d6dcb431ffbb398e149f4535d124b6e727` |
| `fastenhancer_t_scalar.wasm` | `7e5154cf7f9fe3369f9fb488650332b12d95c2031db780c54da557b9256224d8` |
| `fastenhancer_b_scalar.wasm` | `90dec6ec88ace1b8474ae6a07e9961a65d6d3dbcc1e99cd563e00780a952913e` |
| `fastenhancer_s_scalar.wasm` | `674fdec51bc2fe176585e710ba48dcac14c01d36dd51bae480eae56a39126050` |
