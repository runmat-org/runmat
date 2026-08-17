use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactError, LogicalObject, ObjectNamespace};

use crate::{MeshingExecutionError, MeshingExecutionResult};

pub(crate) fn logical_object(
    logical_prefix: &str,
    namespace: ObjectNamespace,
    media_type: &str,
    bytes: Vec<u8>,
) -> MeshingExecutionResult<LogicalObject> {
    let digest = Digest::sha256(&bytes);
    LogicalObject::new(
        namespace,
        format!("{logical_prefix}/{}", digest_hex(digest)),
        media_type,
        bytes,
    )
    .map_err(Into::into)
}

pub(crate) fn read_exact(
    source: &impl CacheImport,
    digest: Digest,
    encoded_length: u64,
) -> MeshingExecutionResult<Vec<u8>> {
    let bytes = read_verified(source, digest)?;
    if bytes.len() as u64 != encoded_length {
        return Err(MeshingExecutionError::Identity(
            "object length differs from its descriptor",
        ));
    }
    Ok(bytes)
}

pub(crate) fn read_verified(
    source: &impl CacheImport,
    digest: Digest,
) -> MeshingExecutionResult<Vec<u8>> {
    let bytes = source
        .read_verified(digest)?
        .ok_or(MeshingExecutionError::MissingObject(digest))?;
    if Digest::sha256(&bytes) != digest {
        return Err(MeshingExecutionError::Identity(
            "cache returned bytes under the wrong digest",
        ));
    }
    Ok(bytes)
}

pub(crate) fn enforce_object_length(
    class: &str,
    encoded_length: u64,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<()> {
    if encoded_length > limits.max_object_bytes {
        return Err(ArtifactError::Limit(format!("{class} object is too large")).into());
    }
    Ok(())
}

pub(crate) fn add_inventory_bytes(
    domain: &str,
    total: &mut u64,
    encoded_length: u64,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<()> {
    *total = total
        .checked_add(encoded_length)
        .ok_or_else(|| ArtifactError::Limit(format!("{domain} object inventory size overflow")))?;
    if *total > limits.max_total_bytes {
        return Err(ArtifactError::Limit(format!("{domain} object inventory is too large")).into());
    }
    Ok(())
}

fn digest_hex(digest: Digest) -> String {
    digest
        .bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
