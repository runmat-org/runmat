use sha2::{Digest, Sha256};

use super::{normalize_metadata, validate_metadata, MetadataError, NativeLibraryMetadata};

pub fn artifact_identity(metadata: &NativeLibraryMetadata) -> Result<String, MetadataError> {
    let normalized = normalize_metadata(metadata.clone());
    validate_metadata(&normalized)?;
    let bytes = serde_json::to_vec(&normalized).map_err(MetadataError::Serialize)?;
    let digest = Sha256::digest(bytes);
    Ok(format!("native-ffi-v1:{digest:x}"))
}
