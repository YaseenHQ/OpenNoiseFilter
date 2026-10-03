# License scope

Original OpenNoiseFilter code is licensed under Apache-2.0, with
`Copyright 2026 YaseenHQ` attribution. The root license does not replace
the licenses of included third-party material.

| Material | License and attribution |
|---|---|
| Original Web Audio, Rust, and model-tool code | Apache-2.0; YaseenHQ |
| FastEnhancer ONNX checkpoints and generated model C/WASM assets | Upstream MIT; AHN Sung Hwan |
| `open-noise-filter-wasm/bench/dnsmos.py` | Upstream MIT; REAL-TSE |
| Emscripten runtime code linked into WASM | Upstream MIT / University of Illinois-NCSA terms |
| musl runtime/math code linked into WASM | Upstream MIT and the permissive terms of its constituent code |
| onnx2c compiler | Its own permissive custom license; compiler provenance retained in generated C |

Each component includes `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md`.
The npm tarball and Rust crate include these files too. Redistribution must
preserve the applicable notices and comply with each included license.

Build and verification dependencies are installed separately through npm,
Cargo, Python, or the toolchain. They retain their own licenses. LiveKit is
an external peer dependency, not bundled into the browser library. Benchmark
audio, external comparison engines, and downloaded DNSMOS models are not
included in the source release; their upstream terms apply when obtained.

Apache-2.0 grants relevant patent rights only from contributors who can
license them. It does not create patent grants from the authors of these
third-party components or guarantee that the complete audio pipeline is
free of third-party patent claims.

The initial GitHub release used MIT for original OpenNoiseFilter code.
Previously distributed copies retain those original license terms.
