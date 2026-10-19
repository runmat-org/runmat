//! Independently verified cold objects retain their original canonical metadata
//! buffers. No geometry result is trusted merely because a digest was serialized.
use super::*;
use runmat_meshing_core::{
    verify_stage_manifest_closure_borrowed_with_admission, MeshingCanonicalCodecControl,
};
pub(crate) fn import_stage_objects_with_admission(
    source: &impl CacheImport,
    root: MeshingStageObjectRoot,
    limits: ObjectInventoryLimits,
    cancellation: &dyn MeshingCancellationSignal,
    admit: &mut impl FnMut(u64) -> MeshingExecutionResult<()>,
) -> MeshingExecutionResult<ValidatedStageObjects> {
    ensure_active(cancellation)?;
    enforce_object_length("manifest", root.encoded_length, limits)?;
    let manifest_bytes = read_exact(source, root.digest, root.encoded_length)?;
    let codec = MeshingCanonicalCodecControl::new(cancellation);
    let manifest = MeshingStageManifest::canonical_decode_with_admission(
        &manifest_bytes,
        &codec,
        |bytes| admit_contract(admit, bytes),
        |manifest| MeshingStageManifest::validation_scratch_bytes(manifest.chunks.len()),
    )?;
    let count = manifest
        .chunks
        .len()
        .checked_add(2)
        .ok_or_else(|| limit("meshing stage object inventory count overflow"))?;
    if count > limits.max_objects {
        return Err(limit("too many meshing stage objects"));
    }
    let mut total = root.encoded_length;
    let identity_digest = execution_digest(manifest.logical_result_identity);
    let identity_bytes = read_verified(source, identity_digest, limits.max_object_bytes.min(limits.max_total_bytes.saturating_sub(total)).min(MeshingStageResultIdentity::LIMITS.maximum_encoded_bytes as u64))
        .map_err(|error| match error { MeshingExecutionError::Artifact(ArtifactError::Limit(_)) => limit("meshing stage object inventory is too large or provider cannot enforce its admitted bound"), other => other })?;
    enforce_object_length(
        "meshing result identity",
        identity_bytes.len() as u64,
        limits,
    )?;
    add_inventory_bytes(
        "meshing stage",
        &mut total,
        identity_bytes.len() as u64,
        limits,
    )?;
    let identity = MeshingStageResultIdentity::canonical_decode_with_admission(
        &identity_bytes,
        &codec,
        |bytes| admit_contract(admit, bytes),
        |_| Ok(0),
    )?;
    admit(
        identity
            .canonical_encoding_profile()?
            .resident_workspace_bytes_estimate,
    )?;
    if identity.canonical_digest()? != manifest.logical_result_identity {
        return Err(MeshingExecutionError::Identity(
            "result identity bytes differ from manifest identity",
        ));
    }
    admit(slots::<EncodedMeshingChunk>(manifest.chunks.len())?)?;
    let mut chunks = Vec::with_capacity(manifest.chunks.len());
    for descriptor in &manifest.chunks {
        ensure_active(cancellation)?;
        enforce_object_length("meshing chunk", descriptor.encoded_length, limits)?;
        add_inventory_bytes(
            "meshing stage",
            &mut total,
            descriptor.encoded_length,
            limits,
        )?;
        let bytes = read_exact(
            source,
            execution_digest(descriptor.digest),
            descriptor.encoded_length,
        )?;
        chunks.push(EncodedMeshingChunk {
            descriptor: descriptor.clone(),
            bytes,
        });
    }
    admit(slots::<runmat_meshing_core::EncodedMeshingChunkRef<'_>>(
        chunks.len(),
    )?)?;
    let borrowed = chunks.iter().map(Into::into).collect::<Vec<_>>();
    verify_stage_manifest_closure_borrowed_with_admission(
        &manifest,
        &identity,
        &borrowed,
        cancellation,
        |profile| admit_contract(admit, profile.requested_bytes()?),
    )?;
    drop(borrowed);
    let prepared = validated::assemble_with_metadata(
        identity,
        manifest,
        chunks,
        limits,
        admit,
        Some((identity_bytes, manifest_bytes)),
    )?;
    if prepared.as_raw().root.digest != root.digest
        || prepared.as_raw().root.encoded_length != root.encoded_length
    {
        return Err(MeshingExecutionError::Identity(
            "imported manifest differs from requested root",
        ));
    }
    Ok(prepared)
}
fn admit_contract(
    admit: &mut impl FnMut(u64) -> MeshingExecutionResult<()>,
    bytes: u64,
) -> Result<(), runmat_meshing_core::MeshingContractError> {
    admit(bytes).map_err(|error| {
        runmat_meshing_core::MeshingContractError::invalid(
            "cold stage metadata admission",
            error.to_string(),
        )
    })
}
fn ensure_active(cancellation: &dyn MeshingCancellationSignal) -> MeshingExecutionResult<()> {
    if cancellation.is_cancelled() {
        Err(MeshingExecutionError::Invalid(
            "cold stage import cancelled".into(),
        ))
    } else {
        Ok(())
    }
}
fn slots<T>(count: usize) -> MeshingExecutionResult<u64> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| limit("cold stage metadata storage overflow"))
}
fn limit(reason: &str) -> MeshingExecutionError {
    ArtifactError::Limit(reason.into()).into()
}
