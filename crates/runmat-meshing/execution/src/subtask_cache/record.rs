//! The identity is inside the retained record, not merely its storage lookup key.
use super::*;
const PREFIX: &[u8] = b"runmat-meshing-subtask-cache/v1\0";
pub(crate) const HEADER_BYTES: usize = PREFIX.len() + 32;

pub(crate) fn frame(bytes: &mut Vec<u8>, key: StableDigest) {
    // Move only the existing bounded wire buffer, never clone the mesh inventory.
    // The shared direct-encoding profile already admits four wire capacities.
    bytes.splice(0..0, PREFIX.iter().chain(key.bytes()).copied());
}

pub(crate) fn body(bytes: &[u8], key: StableDigest) -> Result<&[u8], MeshingSubtaskCacheError> {
    let rest = bytes.strip_prefix(PREFIX).ok_or_else(|| {
        MeshingSubtaskCacheError::Storage("subtask cache record prefix differs".into())
    })?;
    let identity = rest.get(..32).ok_or_else(|| {
        MeshingSubtaskCacheError::Storage("subtask cache record identity is truncated".into())
    })?;
    if identity != key.bytes() {
        return Err(MeshingSubtaskCacheError::Storage(
            "subtask cache record is for another numerical input".into(),
        ));
    }
    Ok(&rest[32..])
}
