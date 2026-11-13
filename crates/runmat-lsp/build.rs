use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Honor an explicitly no-debug Cargo profile for the native DLL. Rustc
    // otherwise requests /DEBUG and MSVC still constructs a large PDB from
    // linked inputs. Keep requested debug builds and other targets unchanged.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
        && env::var("DEBUG").as_deref() == Ok("false")
    {
        println!("cargo:rustc-link-arg-cdylib=/DEBUG:NONE");
    }

    // The canonical catalog is the documentation authority. A published LSP
    // crate must not reach outside its package for repository-only assets.
    // Cargo tracks the runmat-builtins build dependency; this remains the
    // shared owner's full presentation projection, not an example receipt.
    let export = runmat_builtins::documentation_export::export_canonical_presentation()
        .unwrap_or_else(|error| panic!("cannot build authoritative LSP documentation: {error}"));
    let encoded = serde_json::to_string(&export).expect("serialize documentation presentation");
    let output = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    fs::write(output.join("builtin_documentation.json"), encoded)
        .expect("write LSP documentation presentation");
}
