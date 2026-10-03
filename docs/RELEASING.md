# Releasing OpenNoiseFilter

The GitHub source repository contains all three components. npm and
crates.io distribute the two runtime libraries independently; the WASM
tools remain private source tooling.

## Validate

From the repository root, enter each component and run its checks:

```bash
cd open-noise-filter
npm ci
npm run typecheck
npm test
npm run test:browser
npm pack --dry-run
```

Install Chromium first with `npx playwright install chromium`, or set
`BROWSER_EXECUTABLE` to an installed Chromium-based browser. Then run
`cargo test` and `cargo package` from `open-noise-filter-rs/`, and
`npm ci` plus `npm test` from `open-noise-filter-wasm/`.

When updating a model, also rebuild and verify SIMD and scalar assets
against ONNX Runtime, refresh distribution hashes, regenerate native C,
and rerun the native golden tests. Generated caches, dependencies, and
benchmark downloads stay out of the source release.

## Publish packages

Keep the intended version in each component's manifest. Their version
numbers can diverge later if their APIs evolve independently. Check registry
ownership and authenticate through the registry's supported login flow.

From `open-noise-filter/`:

```bash
npm publish --access public
```

From `open-noise-filter-rs/`:

```bash
cargo publish
```

The tools package has `private: true`; do not publish it to npm. Keep
tokens outside the source tree. If a registry requires account verification
or two-factor authentication, complete that step before retrying publication.

## GitHub release

Commit the reviewed source and tag its exact commit, for example `v0.1.0`.
Check root-level CI and create an experimental prerelease with notes that
identify the available packages and known performance limits. GitHub's
source archive contains all components. Optional npm tarball and crate
attachments should come from that same source version, with checksums.

A source release does not imply that both registries published successfully.
Release notes should state which destinations are available and which
remain pending.
