use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StoredArtifact {
    pub name: String,
    pub media_type: String,
    pub byte_len: u64,
    pub content_digest: String,
    pub store_key: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactManifest {
    pub artifacts: Vec<StoredArtifact>,
}
