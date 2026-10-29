use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn collect(root: &Path, dir: &Path, entries: &mut Vec<(String, PathBuf)>) {
    println!("cargo:rerun-if-changed={}", dir.display());
    for entry in fs::read_dir(dir).expect("FEA reference directory") {
        let path = entry.expect("FEA reference entry").path();
        if path.is_dir() {
            // Example packages are studies, not reference pages. Qualification records are
            // engineering evidence: they are listed on the Validation records page and kept out
            // of the agent's reading budget.
            if !matches!(
                path.file_name().and_then(|v| v.to_str()),
                Some(".artifacts" | "examples" | "qualification")
            ) {
                collect(root, &path, entries);
            }
        } else if path.extension().and_then(|v| v.to_str()) == Some("md") {
            let relative = path.strip_prefix(root).expect("reference within root");
            entries.push((
                format!("fea/{}", relative.to_string_lossy().replace('\\', "/")),
                path,
            ));
        }
    }
}

fn main() {
    let root =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("../../../docs/simulation");
    let mut entries = Vec::new();
    collect(&root, &root, &mut entries);
    // The pending-feature registry tells the agent which documented features are not in this
    // binary; docs flag them with `[not yet in the current release](pending:<id>)`.
    let registry = root.join("pending-features.json");
    println!("cargo:rerun-if-changed={}", registry.display());
    entries.push(("fea/pending-features.json".to_string(), registry));
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut source =
        String::from("pub static FEA_REFERENCE_DOCUMENTS: &[ReferenceDocument] = &[\n");
    for (id, path) in entries {
        source.push_str(&format!(
            "ReferenceDocument {{ id: {id:?}, content: include_str!({path:?}) }},\n"
        ));
    }
    source.push_str("];\n");
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("fea_reference.rs"),
        source,
    )
    .unwrap();
}
