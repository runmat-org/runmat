use runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION;
use runmat_execution::value::{ValueLimits, ValueRef, ValueRefKind};
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactError, LogicalObject, ObjectNamespace};

use crate::{ArtifactAccess, MeshingExecutionError, MeshingExecutionResult};

pub(crate) fn logical_object(
    logical_prefix: &str,
    namespace: ObjectNamespace,
    media_type: &str,
    bytes: Vec<u8>,
) -> MeshingExecutionResult<LogicalObject> {
    let digest = Digest::sha256(&bytes);
    let descriptor = runmat_execution_artifact::ObjectDescriptor::content_named(
        namespace,
        logical_prefix,
        media_type,
        digest,
        bytes.len() as u64,
    )?;
    Ok(LogicalObject { descriptor, bytes })
}

pub(crate) fn read_exact(
    source: &impl CacheImport,
    digest: Digest,
    encoded_length: u64,
) -> MeshingExecutionResult<Vec<u8>> {
    let bytes = source
        .read_verified_bounded(digest, encoded_length)?
        .ok_or(MeshingExecutionError::MissingObject(digest))?;
    if Digest::sha256(&bytes) != digest {
        return Err(MeshingExecutionError::Identity(
            "cache returned bytes under the wrong digest",
        ));
    }
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
    maximum_bytes: u64,
) -> MeshingExecutionResult<Vec<u8>> {
    let bytes = source
        .read_verified_bounded(digest, maximum_bytes)?
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

pub(crate) fn input_object_reference(
    object: &LogicalObject,
    access: &ArtifactAccess,
    value_schema: &str,
    domain: &'static str,
) -> MeshingExecutionResult<ValueRef> {
    object.validate()?;
    let reference = ValueRef {
        schema_version: VALUE_PAYLOAD_SCHEMA_VERSION,
        id: access.value_id(object.descriptor.digest),
        logical_digest: object.descriptor.digest,
        encoded_length: object.descriptor.encoded_length,
        media_type: object.descriptor.media_type.clone(),
        value_schema: value_schema.into(),
        encryption_context: access.encryption_context,
        kind: ValueRefKind::DriverObject,
        authorization_scope: access.authorization_scope.clone(),
        resident_fence: None,
    };
    reference
        .validate(ValueLimits::default())
        .map_err(|_| MeshingExecutionError::Identity(domain))?;
    Ok(reference)
}

pub(crate) fn validate_input_root(
    root: &ValueRef,
    access: &ArtifactAccess,
    media_type: &str,
    value_schema: &str,
    domain: &'static str,
) -> MeshingExecutionResult<()> {
    root.validate(ValueLimits::default())
        .map_err(|_| MeshingExecutionError::Identity(domain))?;
    if root.kind != ValueRefKind::DriverObject
        || root.authorization_scope != access.authorization_scope
        || root.encryption_context != access.encryption_context
        || root.media_type != media_type
        || root.value_schema != value_schema
        || root.id != access.value_id(root.logical_digest)
    {
        return Err(MeshingExecutionError::Identity(domain));
    }
    Ok(())
}

pub(crate) fn digest_hex(digest: Digest) -> String {
    digest
        .bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Check canonical name and original bytes without constructing another owned
/// object. Inventory/content verification remains the caller's explicit gate.
pub(crate) fn check_canonical_object(
    object: &LogicalObject,
    namespace: ObjectNamespace,
    prefix: &str,
    media: &str,
    expected: &[u8],
) -> MeshingExecutionResult<()> {
    if object.bytes != expected {
        return Err(MeshingExecutionError::Identity(
            "object bytes differ from their canonical typed contract",
        ));
    }
    let descriptor = runmat_execution_artifact::ObjectDescriptor::content_named(
        namespace,
        prefix,
        media,
        object.descriptor.digest,
        expected.len() as u64,
    )?;
    if object.descriptor != descriptor {
        return Err(MeshingExecutionError::Identity(
            "object descriptor differs from its canonical typed contract",
        ));
    }
    Ok(())
}

/// Optional only for legacy raw APIs; governed stage import always supplies it.
pub(crate) fn admit_import_storage(
    control: Option<&dyn runmat_geometry_core::GeometryCanonicalControl>,
    bytes: u64,
) -> MeshingExecutionResult<()> {
    if let Some(control) = control {
        control
            .consume_allocation_bytes(bytes)
            .map_err(|error| MeshingExecutionError::Invalid(error.to_string()))?;
    }
    Ok(())
}
pub(crate) fn import_slots<T>(count: usize) -> MeshingExecutionResult<u64> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| ArtifactError::Limit("input metadata storage overflow".into()).into())
}
