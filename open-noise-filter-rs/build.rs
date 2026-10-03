// Compiles the vendored onnx2c-generated C for all three FastEnhancer tiers.
// Each file is compiled with -Dentry=fe_<tier>_entry so their identical
// `entry` symbols don't collide — tier selection is at runtime in Rust.
// Flags mirror the WASM build: -O3 -ffast-math (cl: /O2 /fp:fast).
fn main() {
    let msvc = cc::Build::new().get_compiler().is_like_msvc();
    for tier in ["t", "b", "s"] {
        let mut b = cc::Build::new();
        b.file(format!("csrc/fe_{}_norm.c", tier))
            .define("entry", Some(format!("fe_{}_entry", tier).as_str()))
            .opt_level(3)
            .flag_if_supported("-ffast-math")
            .flag_if_supported("/fp:fast")
            // the generated code gates `static inline` on __STDC_VERSION__ >=
            // C99 — MSVC doesn't define it without an explicit /std flag,
            // which would leave every node_* symbol external and collide
            // across tiers at link time
            .flag_if_supported("/std:c17")
            .flag_if_supported("-std=c17")
            .warnings(false);
        if msvc {
            b.define("__restrict__", Some("__restrict"));
        }
        b.compile(&format!("fastenhancer_{}", tier));
    }
    println!("cargo:rerun-if-changed=csrc");
}
