use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

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
