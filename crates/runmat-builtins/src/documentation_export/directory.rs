use serde_json::Value;
use std::{collections::BTreeMap, fs, path::Path};

pub(super) fn read_legacy_documents(
    directory: &Path,
) -> Result<BTreeMap<String, Value>, Box<dyn std::error::Error>> {
    let mut documents = BTreeMap::new();
    for item in fs::read_dir(directory).map_err(|error| {
        format!("authoritative builtin documentation assets unavailable at {}: {error}; transitional consumers require these assets even in standalone package builds", directory.display())
    })? {
        let item = item?;
        let path = item.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| format!("non-UTF-8 builtin sidecar path: {}", path.display()))?;
        let key = stem.to_ascii_lowercase();
        let mut document: Value = serde_json::from_slice(&fs::read(&path)?)?;
        document
            .as_object_mut()
            .ok_or_else(|| format!("builtin sidecar is not an object: {}", path.display()))?
            .insert("module_stem".into(), Value::String(stem.into()));
        if documents.insert(key.clone(), document).is_some() {
            return Err(format!("duplicate legacy documentation identity: {key}").into());
        }
    }
    Ok(documents)
}
