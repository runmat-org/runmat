use runmat_execution::Digest;

use crate::{ArtifactError, ArtifactResult, LogicalObject};

pub trait CacheImport {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>>;
}

pub fn import_verified_object(
    source: &impl CacheImport,
    expected: &LogicalObject,
) -> ArtifactResult<Option<LogicalObject>> {
    let Some(bytes) = source.read_verified(expected.descriptor.digest)? else {
        return Ok(None);
    };
    if Digest::sha256(&bytes) != expected.descriptor.digest {
        return Err(ArtifactError::Identity(
            "cache import returned bytes under the wrong digest".into(),
        ));
    }
    Ok(Some(LogicalObject {
        descriptor: expected.descriptor.clone(),
        bytes,
    }))
}
