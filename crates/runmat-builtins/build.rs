use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("missing manifest dir"));
    let registry_path = manifest_dir
        .join("..")
        .join("..")
        .join("target")
        .join("runmat_wasm_registry.rs");
    if let Some(parent) = registry_path.parent() {
        fs::create_dir_all(parent).expect("failed to create registry directory");
    }
    fs::write(
        &registry_path,
        "// Auto-generated wasm registration list\n{\n}\n",
    )
    .expect("failed to initialize wasm registry file");
}
