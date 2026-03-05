use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct DatasetSchema {
    pub arrays: HashMap<String, usize>,
}

pub trait DatasetSchemaProvider {
    fn load_schema(&self, dataset_path: &str) -> Option<DatasetSchema>;
}

#[derive(Clone, Debug, Default)]
pub struct FsDatasetSchemaProvider;

impl DatasetSchemaProvider for FsDatasetSchemaProvider {
    fn load_schema(&self, dataset_path: &str) -> Option<DatasetSchema> {
        load_dataset_schema_from_fs(dataset_path)
    }
}

pub fn load_dataset_schema_from_fs(dataset_path: &str) -> Option<DatasetSchema> {
    let manifest_path = if dataset_path.ends_with(".json") {
        PathBuf::from(dataset_path)
    } else {
        PathBuf::from(dataset_path).join("manifest.json")
    };
    let bytes = std::fs::read(&manifest_path).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let arrays = value.get("arrays")?.as_object()?;
    let mut out = HashMap::new();
    for (name, meta) in arrays {
        let shape = meta.get("shape")?.as_array()?;
        out.insert(name.clone(), shape.len());
    }
    Some(DatasetSchema { arrays: out })
}

pub fn normalize_literal_string(text: &str) -> String {
    if text.len() >= 2 {
        let bytes = text.as_bytes();
        if (bytes[0] == b'\'' && bytes[text.len() - 1] == b'\'')
            || (bytes[0] == b'"' && bytes[text.len() - 1] == b'"')
        {
            return text[1..text.len() - 1].to_string();
        }
    }
    text.to_string()
}
