# Repository layout

OpenNoiseFilter keeps three components in one source repository:

```text
OpenNoiseFilter/
  README.md
  .github/workflows/ci.yml
  open-noise-filter/       # browser npm package
  open-noise-filter-rs/    # native Rust crate
  open-noise-filter-wasm/  # model build and benchmark tools
```

The package names stay `open-noise-filter` on npm and
`open-noise-filter-rs` on crates.io. The Rust import name is
`open_noise_filter`. The tools component is private and ships as source
in the GitHub release rather than as another npm runtime package.

Keeping the existing directory names preserves the tools' default
`../open-noise-filter` lookup. `ONF_DIR` can override that location.
Each component retains its own license and model attribution so its package
includes the required notices.

Run npm commands inside their component directory. Run Cargo commands from
`open-noise-filter-rs/`; its golden tests load fixtures relative to that
directory. GitHub Actions lives at the repository root and selects the
appropriate working directory for each job.

Model generation produces files in `open-noise-filter-wasm/build/` and
`out/`. Copy the six tier binaries into `open-noise-filter/wasm/` and the
generated C into `open-noise-filter-rs/csrc/` using the mappings in the
component guides. Recompute the browser asset hashes and rerun equivalence
checks before updating a release.

The source repository excludes dependencies, build caches, Python virtual
environments, downloaded benchmark data, and generated benchmark output.
Model ONNX checkpoints, shipped WASM binaries, generated native C, and
small golden fixtures remain tracked so consumers can build the libraries.
