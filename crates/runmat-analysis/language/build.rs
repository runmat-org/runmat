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
            // engineering evidence kept out of the agent's reading budget. Validation case
            // write-ups are already part of the generated per-family Validation pages.
            if !matches!(
                path.file_name().and_then(|v| v.to_str()),
                Some(".artifacts" | "examples" | "qualification" | "cases")
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
    // The capability ledger tells the agent what this binary runs and which documented features
    // are planned (docs flag them with `[not yet in the current release](pending:<id>)`). The agent
    // gets the status columns only; the requirement register stays out of its reading budget.
    let ledger = root.join("ledger.json");
    println!("cargo:rerun-if-changed={}", ledger.display());
    let ledger: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&ledger).expect("docs/simulation/ledger.json"))
            .expect("ledger.json is JSON");
    let capabilities: Vec<serde_json::Value> = ledger["capabilities"]
        .as_array()
        .expect("ledger capabilities")
        .iter()
        .map(|row| {
            let mut out = serde_json::Map::new();
            for key in [
                "id",
                "title",
                "kind",
                "area",
                "status",
                "acceptance",
                "docs",
            ] {
                if let Some(value) = row.get(key) {
                    out.insert(key.to_string(), value.clone());
                }
            }
            serde_json::Value::Object(out)
        })
        .collect();
    let status = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("capabilities.json");
    fs::write(
        &status,
        serde_json::to_string_pretty(&serde_json::json!({
            "release": ledger["release"],
            "about": "Status of every RunMat simulation capability, from docs/simulation/ledger.json. status \"pending\": documented but not in this binary; do not author it into a study, say it is planned.",
            "capabilities": capabilities,
        }))
        .unwrap(),
    )
    .unwrap();
    entries.push(("fea/capabilities.json".to_string(), status));
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
