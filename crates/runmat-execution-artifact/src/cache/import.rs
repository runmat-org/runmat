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
    /// Admit the returned buffer before allocation. Capable providers report its
    /// actual capacity once, after bounding storage metadata and before reading.
    /// The default retains the bounded-read contract conservatively by admitting
    /// the complete ceiling. It never allocates through an unbounded fallback.
    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum_bytes: u64,
        admit: &mut dyn FnMut(u64) -> ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        admit(maximum_bytes)?;
        self.read_verified_bounded(digest, maximum_bytes)
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

/// Copy a resident verified object only after bounding and admitting its actual buffer.
pub fn copy_verified_bounded_with_admission(
    digest: Digest,
    bytes: Option<&[u8]>,
    maximum_bytes: u64,
    admit: &mut dyn FnMut(u64) -> ArtifactResult<()>,
) -> ArtifactResult<Option<Vec<u8>>> {
    if maximum_bytes == 0 {
        return Err(ArtifactError::Limit(
            "object read has no byte allowance".into(),
        ));
    }
    let Some(bytes) = bytes else { return Ok(None) };
    if bytes.len() as u64 > maximum_bytes {
        return Err(ArtifactError::Limit(
            "resident object exceeds admitted read bound".into(),
        ));
    }
    if Digest::sha256(bytes) != digest {
        return Err(ArtifactError::Identity(
            "resident object content identity differs".into(),
        ));
    }
    admit(bytes.len() as u64)?;
    Ok(Some(bytes.to_vec()))
}
