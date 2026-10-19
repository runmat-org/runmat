//! Authenticate original manifest and chunk occurrences; optionally admit each allocation.
use super::*;

pub(crate) struct AnalysisObjectImport<'a> {
    pub root: &'a ValueRef,
    pub expected_kind: AnalysisNumericalObjectKind,
    pub expected_logical_identity: Digest,
    pub expected_payload_digest: Digest,
    pub access: &'a ArtifactAccess,
    pub limits: ObjectInventoryLimits,
}

pub fn import_analysis_objects(
    source: &impl CacheImport,
    root: &ValueRef,
    expected_kind: AnalysisNumericalObjectKind,
    expected_logical_identity: Digest,
    expected_payload_digest: Digest,
    access: &ArtifactAccess,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    import_analysis_objects_admitted(
        source,
        AnalysisObjectImport {
            root,
            expected_kind,
            expected_logical_identity,
            expected_payload_digest,
            access,
            limits,
        },
        None,
    )
}

pub(crate) fn import_analysis_objects_admitted(
    source: &impl CacheImport,
    request: AnalysisObjectImport<'_>,
    admission: Option<&AnalysisInputAdmission>,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    let root = request.root;
    let limits = request.limits;
    let ImportedManifest {
        manifest,
        manifest_object,
        namespace,
    } = import_manifest(source, request, admission)?;
    if manifest.chunks.len().saturating_add(1) > limits.max_objects
        || manifest.encoded_length > limits.max_total_bytes
        || root
            .encoded_length
            .checked_add(manifest.encoded_length)
            .is_none_or(|total| total > limits.max_total_bytes)
    {
        return Err(AnalysisBridgeError::Invalid(
            "analysis object closure exceeds its inventory limits".into(),
        ));
    }
    if let Some(admission) = admission {
        let count = manifest
            .chunks
            .len()
            .checked_add(1)
            .ok_or(AnalysisBridgeError::Identity(
                "analysis input count overflow",
            ))?;
        let metadata = conversion::import_transport_metadata_bytes(count)?;
        admission.charge(metadata)?;
        admission.require_additional(manifest.encoded_length)?;
    }
    let mut objects = Vec::with_capacity(manifest.chunks.len() + 1);
    objects.push(manifest_object);
    for chunk in &manifest.chunks {
        if chunk.encoded_length > limits.max_object_bytes {
            return Err(AnalysisBridgeError::Invalid(
                "analysis chunk exceeds its object limit".into(),
            ));
        }
        let bytes = read_input(source, chunk.digest, chunk.encoded_length, admission)?.ok_or_else(
            || {
                AnalysisBridgeError::Invalid(
                    "analysis chunk is absent from the object store".into(),
                )
            },
        )?;
        if bytes.len() as u64 != chunk.encoded_length || Digest::sha256(&bytes) != chunk.digest {
            return Err(AnalysisBridgeError::Identity(
                "analysis chunk bytes differ from their descriptor",
            ));
        }
        objects.push(logical_object(
            namespace,
            &format!("analysis/chunks/{:08}", chunk.ordinal),
            ANALYSIS_CHUNK_MEDIA_TYPE,
            bytes,
        )?);
    }
    // Every chunk length was checked above; the manifest validates their total.
    if Digest::sha256_parts(objects.iter().skip(1).map(|object| &object.bytes))
        != manifest.payload_digest
    {
        return Err(AnalysisBridgeError::Identity(
            "analysis object closure differs from its payload identity",
        ));
    }
    validate_partition_payloads(&manifest, &objects)?;
    validate_inventory(&objects, limits)?;
    Ok(PreparedAnalysisObjects {
        manifest,
        root: objects[0].descriptor.clone(),
        objects,
    })
}

/// Loads and verifies one semantic payload without reading unrelated chunks.
pub fn import_analysis_partition(
    source: &impl CacheImport,
    request: AnalysisPartitionImport<'_>,
) -> AnalysisBridgeResult<ImportedAnalysisPartition> {
    let ImportedManifest {
        manifest,
        manifest_object,
        namespace,
    } = import_manifest(
        source,
        AnalysisObjectImport {
            root: request.root,
            expected_kind: request.expected_kind,
            expected_logical_identity: request.expected_logical_identity,
            expected_payload_digest: request.expected_payload_digest,
            access: request.access,
            limits: request.limits,
        },
        None,
    )?;
    let partition = manifest
        .partitions
        .iter()
        .find(|partition| partition.logical_identity == request.partition_identity)
        .cloned()
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid(
                "requested semantic payload is absent from the numerical object".into(),
            )
        })?;
    let selected_object_count = partition.chunk_count as usize + 1;
    if selected_object_count > request.limits.max_objects
        || partition.encoded_length > request.limits.max_total_bytes
        || request
            .root
            .encoded_length
            .checked_add(partition.encoded_length)
            .is_none_or(|total| total > request.limits.max_total_bytes)
    {
        return Err(AnalysisBridgeError::Invalid(
            "selected analysis payload exceeds its inventory limits".into(),
        ));
    }
    let start = partition.first_chunk_ordinal as usize;
    let end = start + partition.chunk_count as usize;
    let mut objects = Vec::with_capacity(selected_object_count);
    objects.push(manifest_object);
    for chunk in &manifest.chunks[start..end] {
        if chunk.encoded_length > request.limits.max_object_bytes {
            return Err(AnalysisBridgeError::Invalid(
                "selected analysis chunk exceeds its object limit".into(),
            ));
        }
        let bytes = source.read_verified(chunk.digest)?.ok_or_else(|| {
            AnalysisBridgeError::Invalid(
                "selected analysis chunk is absent from the object store".into(),
            )
        })?;
        if bytes.len() as u64 != chunk.encoded_length || Digest::sha256(&bytes) != chunk.digest {
            return Err(AnalysisBridgeError::Identity(
                "selected analysis chunk differs from its descriptor",
            ));
        }
        objects.push(logical_object(
            namespace,
            &format!("analysis/chunks/{:08}", chunk.ordinal),
            ANALYSIS_CHUNK_MEDIA_TYPE,
            bytes,
        )?);
    }
    // The selected interval and exact chunk lengths prove the partition byte count.
    if Digest::sha256_parts(objects.iter().skip(1).map(|object| &object.bytes))
        != partition.payload_digest
    {
        return Err(AnalysisBridgeError::Identity(
            "selected analysis payload differs from its semantic descriptor",
        ));
    }
    validate_inventory(&objects, request.limits)?;
    Ok(ImportedAnalysisPartition {
        root: objects[0].descriptor.clone(),
        manifest,
        partition,
        objects,
    })
}

struct ImportedManifest {
    manifest: AnalysisObjectManifest,
    manifest_object: LogicalObject,
    namespace: ObjectNamespace,
}

fn import_manifest(
    source: &impl CacheImport,
    request: AnalysisObjectImport<'_>,
    admission: Option<&AnalysisInputAdmission>,
) -> AnalysisBridgeResult<ImportedManifest> {
    let AnalysisObjectImport {
        root,
        expected_kind,
        expected_logical_identity,
        expected_payload_digest,
        access,
        limits,
    } = request;
    access.validate_reference(
        root,
        root.kind,
        ANALYSIS_MANIFEST_MEDIA_TYPE,
        ANALYSIS_MANIFEST_VALUE_SCHEMA,
    )?;
    if matches!(
        root.kind,
        ValueRefKind::ResidentObject | ValueRefKind::SlicedObject
    ) || root.encoded_length == 0
        || root.encoded_length > limits.max_object_bytes
    {
        return Err(AnalysisBridgeError::Invalid(
            "analysis input root is not a portable bounded manifest".into(),
        ));
    }
    let manifest_bytes = read_input(source, root.logical_digest, root.encoded_length, admission)?
        .ok_or_else(|| {
        AnalysisBridgeError::Invalid("analysis manifest is absent from the object store".into())
    })?;
    if manifest_bytes.len() as u64 != root.encoded_length
        || Digest::sha256(&manifest_bytes) != root.logical_digest
    {
        return Err(AnalysisBridgeError::Identity(
            "analysis manifest bytes differ from their reference",
        ));
    }
    if let Some(admission) = admission {
        admission.admit_canonical_decode(&manifest_bytes)?;
    }
    let manifest = AnalysisObjectManifest::from_canonical_bytes(&manifest_bytes)?;
    if manifest.kind != expected_kind
        || manifest.logical_identity != expected_logical_identity
        || manifest.payload_digest != expected_payload_digest
    {
        return Err(AnalysisBridgeError::Identity(
            "analysis manifest semantics differ from the workload input",
        ));
    }
    let namespace = match root.kind {
        ValueRefKind::ResultObject | ValueRefKind::WorkerObject => ObjectNamespace::ResultValue,
        ValueRefKind::CheckpointObject => ObjectNamespace::Checkpoint,
        ValueRefKind::DriverObject
        | ValueRefKind::ProjectObject
        | ValueRefKind::BroadcastObject => ObjectNamespace::InputValue,
        ValueRefKind::ResidentObject | ValueRefKind::SlicedObject => unreachable!(),
    };
    let manifest_object = logical_object(
        namespace,
        "analysis/manifests",
        ANALYSIS_MANIFEST_MEDIA_TYPE,
        manifest_bytes,
    )?;
    Ok(ImportedManifest {
        manifest,
        manifest_object,
        namespace,
    })
}
