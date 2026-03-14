use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Provider-neutral dataset manifest descriptor shared by runtime and remote clients.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DataManifestDescriptor {
    pub schema_version: u32,
    pub format: String,
    pub dataset_id: String,
    pub updated_at: String,
    pub txn_sequence: u64,
}

/// Request payload for fetching dataset metadata from remote implementations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DataManifestRequest {
    pub path: String,
    pub version: Option<String>,
}

/// Chunk descriptor used by remote dataset chunk APIs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DataChunkDescriptor {
    pub key: String,
    pub object_id: String,
    pub hash: String,
    pub bytes_raw: u64,
    pub bytes_stored: u64,
}

/// Request payload for batched chunk upload target issuance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DataChunkUploadRequest {
    pub dataset_path: String,
    pub array: String,
    pub chunks: Vec<DataChunkDescriptor>,
}

/// Response payload for provider-neutral chunk upload targets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DataChunkUploadTarget {
    pub key: String,
    pub method: String,
    pub upload_url: String,
    pub headers: HashMap<String, String>,
}
