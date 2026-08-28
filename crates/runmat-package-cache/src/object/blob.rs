use crate::CacheError;
use runmat_package::ContentDigest;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlobMetadata {
    pub digest: ContentDigest,
    pub byte_len: u64,
}

impl BlobMetadata {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            digest: ContentDigest::sha256(bytes),
            byte_len: bytes.len() as u64,
        }
    }

    pub fn verify(&self, bytes: &[u8]) -> Result<(), CacheError> {
        if bytes.len() as u64 != self.byte_len || ContentDigest::sha256(bytes) != self.digest {
            return Err(CacheError::DigestMismatch(self.digest.clone()));
        }
        Ok(())
    }
}
