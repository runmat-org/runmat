#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpaqueObject {
    pub ciphertext_digest: String,
    pub ciphertext_size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectChunk {
    pub offset: u64,
    pub bytes: Vec<u8>,
}
