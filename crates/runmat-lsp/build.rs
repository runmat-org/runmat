use std::{env, fs, path::PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let legacy_directory = manifest_dir.join("../../docs/builtins/reference");
    println!("cargo:rerun-if-changed={}", legacy_directory.display());

    // One shared owner enforces canonical/legacy source ownership. No nested
    // Cargo invocation, generated name map, or consumer-specific sidecar merge.
    // Transition is explicit: existing legacy prose is retained, but missing
    // package assets are an error rather than a canonical-only fallback.
    let export = runmat_builtins::documentation_export::export_presentation_from_directory(
        &legacy_directory,
        true,
    )
    .unwrap_or_else(|error| panic!("cannot build authoritative LSP documentation: {error}"));
    let encoded = serde_json::to_string(&export).expect("serialize documentation presentation");
    let output = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    fs::write(output.join("builtin_documentation.json"), encoded)
        .expect("write LSP documentation presentation");
}
