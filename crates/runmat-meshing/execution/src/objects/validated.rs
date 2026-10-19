//! Construction-only authority for an immutable, already closed object set.
//! Raw import/public APIs retain their independent full revalidation.
use super::*;
pub(crate) struct ValidatedStageObjects(PreparedMeshingStageObjects);
impl ValidatedStageObjects {
    // Only the parent object's raw revalidation gate can construct this authority.
    pub(super) fn from_revalidated(objects: PreparedMeshingStageObjects) -> Self {
        Self(objects)
    }
    pub(super) fn as_raw(&self) -> &PreparedMeshingStageObjects {
        &self.0
    }
    pub(crate) fn into_raw(self) -> PreparedMeshingStageObjects {
        self.0
    }
}
pub(crate) fn prepare_closed_stage_objects(
    closed: runmat_meshing_core::ClosedMeshingStage,
    limits: ObjectInventoryLimits,
    mut admit: impl FnMut(u64) -> MeshingExecutionResult<()>,
) -> MeshingExecutionResult<ValidatedStageObjects> {
    let (identity, manifest, chunks) = closed.into_parts();
    assemble(identity, manifest, chunks, limits, &mut admit)
}
pub(super) fn assemble(
    result_identity: MeshingStageResultIdentity,
    manifest: MeshingStageManifest,
    chunks: Vec<EncodedMeshingChunk>,
    limits: ObjectInventoryLimits,
    admit: &mut impl FnMut(u64) -> MeshingExecutionResult<()>,
) -> MeshingExecutionResult<ValidatedStageObjects> {
    assemble_with_metadata(result_identity, manifest, chunks, limits, admit, None)
}
pub(super) fn assemble_with_metadata(
    result_identity: MeshingStageResultIdentity,
    manifest: MeshingStageManifest,
    chunks: Vec<EncodedMeshingChunk>,
    limits: ObjectInventoryLimits,
    admit: &mut impl FnMut(u64) -> MeshingExecutionResult<()>,
    metadata: Option<(Vec<u8>, Vec<u8>)>,
) -> MeshingExecutionResult<ValidatedStageObjects> {
    let (original_identity, original_manifest) = match metadata {
        Some((identity, manifest)) => (Some(identity), Some(manifest)),
        None => (None, None),
    };
    let count = chunks
        .len()
        .checked_add(2)
        .ok_or_else(|| limit("meshing object count overflow"))?;
    if count > limits.max_objects {
        return Err(limit("too many meshing stage objects"));
    }
    admit(slots::<LogicalObject>(count)?)?;
    let mut objects = Vec::with_capacity(count);
    let mut total = 0u64;
    let identity = if let Some(bytes) = original_identity {
        bytes
    } else {
        admit(
            result_identity
                .canonical_encoding_profile()?
                .resident_workspace_bytes_estimate,
        )?;
        result_identity.canonical_encode()?
    };
    let object = named_object(
        "meshing/v2/identities",
        MESHING_RESULT_IDENTITY_MEDIA_TYPE,
        identity,
        admit,
    )?;
    check_limit(&object, &mut total, limits)?;
    objects.push(object);
    for chunk in chunks {
        let media = chunk.descriptor.media_type.media_type();
        admit(ObjectDescriptor::content_named_text_bytes(
            "meshing/v2/chunks",
            media,
        )?)?;
        // The consumed core closure has verified these exact immutable bytes.
        let descriptor = ObjectDescriptor::content_named(
            ObjectNamespace::ResultValue,
            "meshing/v2/chunks",
            media,
            execution_digest(chunk.descriptor.digest),
            chunk.descriptor.encoded_length,
        )?;
        let object = LogicalObject {
            descriptor,
            bytes: chunk.bytes,
        };
        check_limit(&object, &mut total, limits)?;
        objects.push(object);
    }
    let bytes = if let Some(bytes) = original_manifest {
        bytes
    } else {
        admit(
            manifest
                .canonical_encoding_profile()?
                .resident_workspace_bytes_estimate,
        )?;
        admit(MeshingStageManifest::validation_scratch_bytes(
            manifest.chunks.len(),
        )?)?;
        manifest.canonical_encode()?
    };
    let object = named_object(
        "meshing/v2/manifests",
        MESHING_STAGE_MANIFEST_MEDIA_TYPE,
        bytes,
        admit,
    )?;
    check_limit(&object, &mut total, limits)?;
    admit((object.descriptor.logical_name.len() + object.descriptor.media_type.len()) as u64)?;
    let root = object.descriptor.clone();
    objects.push(object);
    Ok(ValidatedStageObjects(PreparedMeshingStageObjects {
        result_identity,
        manifest,
        root,
        objects,
    }))
}
fn named_object(
    prefix: &str,
    media: &str,
    bytes: Vec<u8>,
    admit: &mut impl FnMut(u64) -> MeshingExecutionResult<()>,
) -> MeshingExecutionResult<LogicalObject> {
    admit(ObjectDescriptor::content_named_text_bytes(prefix, media)?)?;
    logical_object(prefix, ObjectNamespace::ResultValue, media, bytes)
}
fn check_limit(
    object: &LogicalObject,
    total: &mut u64,
    limits: ObjectInventoryLimits,
) -> MeshingExecutionResult<()> {
    enforce_object_length("meshing object", object.descriptor.encoded_length, limits)?;
    add_inventory_bytes(
        "meshing stage",
        total,
        object.descriptor.encoded_length,
        limits,
    )
}
fn slots<T>(count: usize) -> MeshingExecutionResult<u64> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| limit("meshing object storage overflow"))
}
fn limit(reason: &str) -> MeshingExecutionError {
    ArtifactError::Limit(reason.into()).into()
}
