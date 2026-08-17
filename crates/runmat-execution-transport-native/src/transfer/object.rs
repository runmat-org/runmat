#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpaqueObject {
    pub ciphertext_digest: String,
    pub ciphertext_size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ObjectChunk {
    pub offset: u64,
    pub bytes: Vec<u8>,
}
