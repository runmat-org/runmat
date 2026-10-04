use runmat_execution::Digest;

use crate::{ArtifactError, ArtifactResult, LogicalObject};

pub trait CacheImport {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>>;

    /// Enforce this caller's byte bound before allocating or reading the payload.
    /// Providers must opt in; an unbounded read followed by a length check is insufficient.
    fn read_verified_bounded(
        &self,
        _digest: Digest,
        _maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        Err(ArtifactError::Limit(
            "cache provider cannot enforce bounded verified reads".into(),
        ))
    }
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
