use runmat_analysis_fea::execution::{
    chunk_numerical_payload, chunk_partitioned_numerical_payload, AnalysisNumericalObjectKind,
    AnalysisObjectManifest, AnalysisPayloadPartition, AnalysisPayloadPartitionDescriptor,
    ChunkedAnalysisPayload, ANALYSIS_CHUNK_MEDIA_TYPE, ANALYSIS_CHUNK_VALUE_SCHEMA,
    ANALYSIS_MANIFEST_MEDIA_TYPE, ANALYSIS_MANIFEST_VALUE_SCHEMA,
};
use runmat_analysis_fea::planning::resource::{
    ByteCount, PreparedInputResourceFacts, PreparedOutputResourceFacts,
};
use runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_V1;
use runmat_execution::value::{ValueRef, ValueRefKind};
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::{validate_inventory, ObjectInventoryLimits};
use runmat_execution_artifact::{ArtifactAccess, LogicalObject, ObjectDescriptor, ObjectNamespace};

use crate::{AnalysisBridgeError, AnalysisBridgeResult};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedAnalysisObjects {
    pub manifest: AnalysisObjectManifest,
    pub root: ObjectDescriptor,
    pub objects: Vec<LogicalObject>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportedAnalysisPartition {
    pub manifest: AnalysisObjectManifest,
    pub partition: AnalysisPayloadPartitionDescriptor,
    pub root: ObjectDescriptor,
    pub objects: Vec<LogicalObject>,
}

pub struct AnalysisPartitionImport<'a> {
    pub root: &'a ValueRef,
    pub expected_kind: AnalysisNumericalObjectKind,
    pub expected_logical_identity: Digest,
    pub expected_payload_digest: Digest,
    pub partition_identity: Digest,
    pub access: &'a ArtifactAccess,
    pub limits: ObjectInventoryLimits,
}

impl ImportedAnalysisPartition {
    pub fn payload(&self) -> Vec<u8> {
        self.objects
            .iter()
            .skip(1)
            .flat_map(|object| object.bytes.iter().copied())
            .collect()
    }
}

impl PreparedAnalysisObjects {
    pub fn payload(&self) -> Vec<u8> {
        self.objects
            .iter()
            .skip(1)
            .flat_map(|object| object.bytes.iter().copied())
            .collect()
    }

    pub fn partition_payload(&self, identity: Digest) -> AnalysisBridgeResult<Vec<u8>> {
        let partition = self
            .manifest
            .partitions
            .iter()
            .find(|partition| partition.logical_identity == identity)
            .ok_or_else(|| {
                AnalysisBridgeError::Invalid(
                    "semantic payload is absent from the prepared numerical object".into(),
                )
            })?;
        let start = partition.first_chunk_ordinal as usize + 1;
        let end = start + partition.chunk_count as usize;
        if end > self.objects.len() {
            return Err(AnalysisBridgeError::Identity(
                "prepared numerical object omits semantic payload chunks",
            ));
        }
        let payload = self.objects[start..end]
            .iter()
            .flat_map(|object| object.bytes.iter().copied())
            .collect::<Vec<_>>();
        if payload.len() as u64 != partition.encoded_length
            || Digest::sha256(&payload) != partition.payload_digest
        {
            return Err(AnalysisBridgeError::Identity(
                "prepared semantic payload differs from its descriptor",
            ));
        }
        Ok(payload)
    }

    pub fn references(
        &self,
        access: &ArtifactAccess,
        kind: ValueRefKind,
    ) -> AnalysisBridgeResult<Vec<ValueRef>> {
        self.objects
            .iter()
            .enumerate()
            .map(|(index, object)| {
                let schema = if index == 0 {
                    ANALYSIS_MANIFEST_VALUE_SCHEMA
                } else {
                    ANALYSIS_CHUNK_VALUE_SCHEMA
                };
                access.reference(object, kind, schema).map_err(Into::into)
            })
            .collect()
    }

    /// Returns the content-addressed transport inventory for a result closure.
    ///
    /// A numerical manifest may refer to the same immutable chunk at multiple semantic ordinals.
    /// The result protocol carries that content object once, while the manifest retains every
    /// occurrence.
    pub fn unique_references(
        &self,
        access: &ArtifactAccess,
        kind: ValueRefKind,
    ) -> AnalysisBridgeResult<Vec<ValueRef>> {
        let mut references = self.references(access, kind)?;
        retain_unique_references(&mut references);
        Ok(references)
    }

    pub fn root_reference(
        &self,
        access: &ArtifactAccess,
        kind: ValueRefKind,
    ) -> AnalysisBridgeResult<ValueRef> {
        Ok(access.reference(
            self.objects.first().ok_or(AnalysisBridgeError::Identity(
                "analysis object has no manifest",
            ))?,
            kind,
            ANALYSIS_MANIFEST_VALUE_SCHEMA,
        )?)
    }
}

/// Measures the unique content-addressed input closure and structurally profiles canonical value
/// nodes, maps, and strings for the decoded-residency admission estimate. Numerical domains add
/// their own typed workspaces, and scheduler accounting remains authoritative for actual usage.
pub fn prepared_input_resource_facts(
    inputs: &[PreparedAnalysisObjects],
) -> AnalysisBridgeResult<PreparedInputResourceFacts> {
    let encoded_payload_bytes = inputs.iter().try_fold(0_u64, |total, input| {
        total
            .checked_add(input.manifest.encoded_length)
            .ok_or_else(|| AnalysisBridgeError::Invalid("input payload bytes overflow".into()))
    })?;
    let mut unique = std::collections::BTreeMap::new();
    for input in inputs {
        for object in &input.objects {
            unique
                .entry(object.descriptor.digest)
                .or_insert_with(|| object.bytes.len());
        }
    }
    let transport_closure_bytes = unique.values().try_fold(0_u64, |total, bytes| {
        total
            .checked_add(u64::try_from(*bytes).map_err(|_| {
                AnalysisBridgeError::Invalid("input object bytes are not representable".into())
            })?)
            .ok_or_else(|| AnalysisBridgeError::Invalid("input closure bytes overflow".into()))
    })?;
    let decoded_resident_bytes_admission_budget =
        inputs.iter().try_fold(0_u64, |total, input| {
            input
                .manifest
                .partitions
                .iter()
                .try_fold(total, |partition_total, partition| {
                    let payload = input.partition_payload(partition.logical_identity)?;
                    let estimate = runmat_canonical_codec::canonical_payload_materialization_resident_bytes_estimate(
                        &payload,
                    )
                    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
                    partition_total.checked_add(estimate).ok_or_else(|| {
                        AnalysisBridgeError::Invalid("decoded input residency overflows".into())
                    })
                })
        })?;
    Ok(PreparedInputResourceFacts {
        encoded_payload_bytes: ByteCount(encoded_payload_bytes),
        transport_closure_bytes: ByteCount(transport_closure_bytes),
        decoded_resident_bytes_admission_budget: ByteCount(decoded_resident_bytes_admission_budget),
    })
}

/// Projects a chunk-policy-independent output envelope before runtime-varying provenance exists.
/// Both manifest allowances come from canonical contract byte limits, so the bound remains valid
/// for every admitted chunk count and output-set reference inventory.
pub fn projected_output_resource_facts(
    encoded_payload_bytes_upper_bound: u64,
) -> AnalysisBridgeResult<PreparedOutputResourceFacts> {
    if encoded_payload_bytes_upper_bound == 0 {
        return Err(AnalysisBridgeError::Invalid(
            "projected output payload must be nonzero".into(),
        ));
    }
    let transport_closure_bytes_upper_bound = encoded_payload_bytes_upper_bound
        .checked_add(runmat_analysis_fea::execution::ANALYSIS_MANIFEST_MAX_ENCODED_BYTES)
        .ok_or_else(|| AnalysisBridgeError::Invalid("projected output closure overflows".into()))?;
    let construction_resident_bytes_admission_budget =
        runmat_canonical_codec::projected_canonical_materialization_resident_bytes_estimate(
            encoded_payload_bytes_upper_bound,
            1,
        )
        .and_then(|bytes| bytes.checked_add(transport_closure_bytes_upper_bound))
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("projected output construction overflows".into())
        })?;
    Ok(PreparedOutputResourceFacts {
        encoded_payload_bytes_upper_bound: ByteCount(encoded_payload_bytes_upper_bound),
        transport_closure_bytes_upper_bound: ByteCount(transport_closure_bytes_upper_bound),
        construction_resident_bytes_admission_budget: ByteCount(
            construction_resident_bytes_admission_budget,
        ),
    })
}

pub fn projected_output_set_resource_facts(
) -> runmat_analysis_fea::planning::resource::PreparedOutputSetResourceFacts {
    runmat_analysis_fea::planning::resource::PreparedOutputSetResourceFacts {
        encoded_bytes_upper_bound: ByteCount(
            crate::response::ANALYSIS_OUTPUT_SET_MAX_ENCODED_BYTES,
        ),
    }
}

pub(crate) fn validate_prepared_output_resource_facts(
    output: &PreparedAnalysisObjects,
    admitted: &PreparedOutputResourceFacts,
) -> AnalysisBridgeResult<()> {
    let mut unique = std::collections::BTreeMap::new();
    for object in &output.objects {
        unique
            .entry(object.descriptor.digest)
            .or_insert_with(|| object.bytes.len());
    }
    let actual_closure = unique.values().try_fold(0_u64, |total, bytes| {
        total
            .checked_add(u64::try_from(*bytes).map_err(|_| {
                AnalysisBridgeError::Invalid("output object bytes are not representable".into())
            })?)
            .ok_or_else(|| AnalysisBridgeError::Invalid("output closure bytes overflow".into()))
    })?;
    if output.manifest.encoded_length > admitted.encoded_payload_bytes_upper_bound.0
        || actual_closure > admitted.transport_closure_bytes_upper_bound.0
    {
        return Err(AnalysisBridgeError::Invalid(
            "prepared output exceeds its admitted semantic or transport envelope".into(),
        ));
    }
    Ok(())
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
    let ImportedManifest {
        manifest,
        manifest_object,
        namespace,
    } = import_manifest(
        source,
        root,
        expected_kind,
        expected_logical_identity,
        expected_payload_digest,
        access,
        limits,
    )?;
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
    let mut objects = Vec::with_capacity(manifest.chunks.len() + 1);
    objects.push(manifest_object);
    let mut payload =
        Vec::with_capacity(usize::try_from(manifest.encoded_length).map_err(|_| {
            AnalysisBridgeError::Invalid("analysis payload length is not representable".into())
        })?);
    for chunk in &manifest.chunks {
        if chunk.encoded_length > limits.max_object_bytes {
            return Err(AnalysisBridgeError::Invalid(
                "analysis chunk exceeds its object limit".into(),
            ));
        }
        let bytes = source.read_verified(chunk.digest)?.ok_or_else(|| {
            AnalysisBridgeError::Invalid("analysis chunk is absent from the object store".into())
        })?;
        if bytes.len() as u64 != chunk.encoded_length || Digest::sha256(&bytes) != chunk.digest {
            return Err(AnalysisBridgeError::Identity(
                "analysis chunk bytes differ from their descriptor",
            ));
        }
        payload.extend_from_slice(&bytes);
        objects.push(logical_object(
            namespace,
            &format!("analysis/chunks/{:08}", chunk.ordinal),
            ANALYSIS_CHUNK_MEDIA_TYPE,
            bytes,
        )?);
    }
    if payload.len() as u64 != manifest.encoded_length
        || Digest::sha256(&payload) != manifest.payload_digest
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
        request.root,
        request.expected_kind,
        request.expected_logical_identity,
        request.expected_payload_digest,
        request.access,
        request.limits,
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
    let payload = objects
        .iter()
        .skip(1)
        .flat_map(|object| object.bytes.iter().copied())
        .collect::<Vec<_>>();
    if payload.len() as u64 != partition.encoded_length
        || Digest::sha256(&payload) != partition.payload_digest
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

pub fn analysis_object_references(
    manifest: &AnalysisObjectManifest,
    root: &ValueRef,
    access: &ArtifactAccess,
    kind: ValueRefKind,
) -> AnalysisBridgeResult<Vec<ValueRef>> {
    manifest.validate()?;
    access.validate_reference(
        root,
        kind,
        ANALYSIS_MANIFEST_MEDIA_TYPE,
        ANALYSIS_MANIFEST_VALUE_SCHEMA,
    )?;
    let mut references = Vec::with_capacity(manifest.chunks.len() + 1);
    references.push(root.clone());
    for chunk in &manifest.chunks {
        references.push(ValueRef {
            schema_version: VALUE_PAYLOAD_SCHEMA_V1,
            id: access.value_id(chunk.digest),
            logical_digest: chunk.digest,
            encoded_length: chunk.encoded_length,
            media_type: ANALYSIS_CHUNK_MEDIA_TYPE.into(),
            value_schema: ANALYSIS_CHUNK_VALUE_SCHEMA.into(),
            encryption_context: access.encryption_context,
            kind,
            authorization_scope: access.authorization_scope.clone(),
            resident_fence: None,
        });
    }
    retain_unique_references(&mut references);
    for reference in &references {
        access.validate_reference(
            reference,
            kind,
            if reference == root {
                ANALYSIS_MANIFEST_MEDIA_TYPE
            } else {
                ANALYSIS_CHUNK_MEDIA_TYPE
            },
            if reference == root {
                ANALYSIS_MANIFEST_VALUE_SCHEMA
            } else {
                ANALYSIS_CHUNK_VALUE_SCHEMA
            },
        )?;
    }
    Ok(references)
}

pub(crate) fn retain_unique_references(references: &mut Vec<ValueRef>) {
    let mut identities = std::collections::BTreeSet::new();
    references.retain(|reference| identities.insert(reference.id));
}

pub fn prepare_analysis_objects(
    kind: AnalysisNumericalObjectKind,
    logical_identity: Digest,
    payload: &[u8],
    chunk_bytes: usize,
    namespace: ObjectNamespace,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    let chunked = chunk_numerical_payload(kind, logical_identity, payload, chunk_bytes)?;
    prepare_chunked_analysis_objects(chunked, namespace, limits)
}

/// Rebinds an unpartitioned prepared input to an independently derived numerical identity.
/// Chunk content is immutable and transferred without copying; only the manifest changes.
pub(crate) fn rebind_unpartitioned_analysis_identity(
    mut prepared: PreparedAnalysisObjects,
    logical_identity: Digest,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    if prepared.manifest.partitions.len() != 1
        || prepared.manifest.partitions[0].logical_identity != prepared.manifest.logical_identity
        || prepared.objects.len() != prepared.manifest.chunks.len() + 1
        || prepared.objects.first().map(|object| &object.descriptor) != Some(&prepared.root)
    {
        return Err(AnalysisBridgeError::Identity(
            "analysis input cannot be rebound from a partitioned or invalid manifest",
        ));
    }
    validate_inventory(&prepared.objects, limits)?;
    if prepared.manifest.canonical_bytes()? != prepared.objects[0].bytes {
        return Err(AnalysisBridgeError::Identity(
            "analysis input manifest differs from its prepared root",
        ));
    }
    if prepared
        .manifest
        .chunks
        .iter()
        .zip(prepared.objects.iter().skip(1))
        .any(|(chunk, object)| {
            object.descriptor.digest != chunk.digest
                || object.descriptor.encoded_length != chunk.encoded_length
                || object.descriptor.namespace != prepared.root.namespace
        })
    {
        return Err(AnalysisBridgeError::Identity(
            "analysis input chunks differ from their prepared manifest",
        ));
    }
    prepared.manifest.logical_identity = logical_identity;
    prepared.manifest.partitions[0].logical_identity = logical_identity;
    let manifest_object = logical_object(
        prepared.root.namespace,
        "analysis/manifests",
        ANALYSIS_MANIFEST_MEDIA_TYPE,
        prepared.manifest.canonical_bytes()?,
    )?;
    prepared.root = manifest_object.descriptor.clone();
    prepared.objects[0] = manifest_object;
    validate_inventory(&prepared.objects, limits)?;
    Ok(prepared)
}

pub fn prepare_partitioned_analysis_objects(
    kind: AnalysisNumericalObjectKind,
    logical_identity: Digest,
    partitions: &[AnalysisPayloadPartition<'_>],
    chunk_bytes: usize,
    namespace: ObjectNamespace,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    let chunked =
        chunk_partitioned_numerical_payload(kind, logical_identity, partitions, chunk_bytes)?;
    prepare_chunked_analysis_objects(chunked, namespace, limits)
}

fn prepare_chunked_analysis_objects(
    chunked: ChunkedAnalysisPayload,
    namespace: ObjectNamespace,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    let mut objects = Vec::with_capacity(chunked.chunks.len() + 1);
    let manifest_object = logical_object(
        namespace,
        "analysis/manifests",
        ANALYSIS_MANIFEST_MEDIA_TYPE,
        chunked.manifest.canonical_bytes()?,
    )?;
    let root = manifest_object.descriptor.clone();
    objects.push(manifest_object);
    for (descriptor, bytes) in chunked.manifest.chunks.iter().zip(chunked.chunks) {
        if descriptor.digest != Digest::sha256(&bytes)
            || descriptor.encoded_length != bytes.len() as u64
        {
            return Err(AnalysisBridgeError::Identity(
                "analysis chunk differs from its manifest descriptor",
            ));
        }
        objects.push(logical_object(
            namespace,
            &format!("analysis/chunks/{:08}", descriptor.ordinal),
            ANALYSIS_CHUNK_MEDIA_TYPE,
            bytes,
        )?);
    }
    validate_inventory(&objects, limits)?;
    Ok(PreparedAnalysisObjects {
        manifest: chunked.manifest,
        root,
        objects,
    })
}

fn logical_object(
    namespace: ObjectNamespace,
    prefix: &str,
    media_type: &str,
    bytes: Vec<u8>,
) -> AnalysisBridgeResult<LogicalObject> {
    let digest = Digest::sha256(&bytes);
    let suffix = digest
        .bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(LogicalObject::new(
        namespace,
        format!("{prefix}/{suffix}"),
        media_type,
        bytes,
    )?)
}

struct ImportedManifest {
    manifest: AnalysisObjectManifest,
    manifest_object: LogicalObject,
    namespace: ObjectNamespace,
}

fn import_manifest(
    source: &impl CacheImport,
    root: &ValueRef,
    expected_kind: AnalysisNumericalObjectKind,
    expected_logical_identity: Digest,
    expected_payload_digest: Digest,
    access: &ArtifactAccess,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedManifest> {
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
    let manifest_bytes = source.read_verified(root.logical_digest)?.ok_or_else(|| {
        AnalysisBridgeError::Invalid("analysis manifest is absent from the object store".into())
    })?;
    if manifest_bytes.len() as u64 != root.encoded_length
        || Digest::sha256(&manifest_bytes) != root.logical_digest
    {
        return Err(AnalysisBridgeError::Identity(
            "analysis manifest bytes differ from their reference",
        ));
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

fn validate_partition_payloads(
    manifest: &AnalysisObjectManifest,
    objects: &[LogicalObject],
) -> AnalysisBridgeResult<()> {
    for partition in &manifest.partitions {
        let start = partition.first_chunk_ordinal as usize + 1;
        let end = start + partition.chunk_count as usize;
        let payload = objects[start..end]
            .iter()
            .flat_map(|object| object.bytes.iter().copied())
            .collect::<Vec<_>>();
        if payload.len() as u64 != partition.encoded_length
            || Digest::sha256(&payload) != partition.payload_digest
        {
            return Err(AnalysisBridgeError::Identity(
                "analysis payload partition differs from its descriptor",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod resource_tests {
    use super::*;

    #[test]
    fn rebinding_unpartitioned_identity_preserves_payload_chunks() {
        let limits = ObjectInventoryLimits::default();
        let payload = vec![7_u8; 129];
        let prepared = prepare_analysis_objects(
            AnalysisNumericalObjectKind::StructuralTransientWorkload,
            Digest::sha256("scope-workload"),
            &payload,
            32,
            ObjectNamespace::InputValue,
            limits,
        )
        .unwrap();
        let old_root = prepared.root.digest;
        let chunk_addresses = prepared
            .objects
            .iter()
            .skip(1)
            .map(|object| object.bytes.as_ptr())
            .collect::<Vec<_>>();
        let rebound = rebind_unpartitioned_analysis_identity(
            prepared,
            Digest::sha256("assembled-workload"),
            limits,
        )
        .unwrap();
        assert_ne!(rebound.root.digest, old_root);
        assert_eq!(
            rebound.manifest.logical_identity,
            Digest::sha256("assembled-workload")
        );
        assert_eq!(
            rebound.manifest.partitions[0].logical_identity,
            rebound.manifest.logical_identity
        );
        assert_eq!(rebound.payload(), payload);
        assert_eq!(
            rebound
                .objects
                .iter()
                .skip(1)
                .map(|object| object.bytes.as_ptr())
                .collect::<Vec<_>>(),
            chunk_addresses
        );
        assert_eq!(
            rebound,
            prepare_analysis_objects(
                AnalysisNumericalObjectKind::StructuralTransientWorkload,
                Digest::sha256("assembled-workload"),
                &payload,
                32,
                ObjectNamespace::InputValue,
                limits,
            )
            .unwrap()
        );

        let partitioned = prepare_partitioned_analysis_objects(
            AnalysisNumericalObjectKind::StructuralTransientWorkload,
            Digest::sha256("partitioned-workload"),
            &[
                AnalysisPayloadPartition {
                    logical_identity: Digest::sha256("left"),
                    payload: &[1_u8; 8],
                },
                AnalysisPayloadPartition {
                    logical_identity: Digest::sha256("right"),
                    payload: &[2_u8; 8],
                },
            ],
            32,
            ObjectNamespace::InputValue,
            limits,
        )
        .unwrap();
        assert!(rebind_unpartitioned_analysis_identity(
            partitioned,
            Digest::sha256("rebinding-must-fail"),
            limits,
        )
        .is_err());

        let mut substituted = prepare_analysis_objects(
            AnalysisNumericalObjectKind::StructuralTransientWorkload,
            Digest::sha256("scope-workload"),
            &payload,
            32,
            ObjectNamespace::InputValue,
            limits,
        )
        .unwrap();
        let original = &substituted.objects[1];
        substituted.objects[1] = LogicalObject::new(
            original.descriptor.namespace,
            original.descriptor.logical_name.clone(),
            original.descriptor.media_type.clone(),
            vec![9_u8; original.bytes.len()],
        )
        .unwrap();
        assert!(rebind_unpartitioned_analysis_identity(
            substituted,
            Digest::sha256("rebinding-must-fail"),
            limits,
        )
        .is_err());
    }

    #[test]
    fn prepared_output_must_fit_semantic_and_transport_envelopes_independently() {
        let prepared = prepare_analysis_objects(
            AnalysisNumericalObjectKind::ObservationResult,
            Digest::sha256("output"),
            &[7_u8; 128],
            32,
            ObjectNamespace::ResultValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        let actual_closure = prepared
            .objects
            .iter()
            .map(|object| (object.descriptor.digest, object.bytes.len() as u64))
            .collect::<std::collections::BTreeMap<_, _>>()
            .into_values()
            .sum::<u64>();
        let admitted = PreparedOutputResourceFacts {
            encoded_payload_bytes_upper_bound: ByteCount(128),
            transport_closure_bytes_upper_bound: ByteCount(actual_closure),
            construction_resident_bytes_admission_budget: ByteCount(128),
        };
        validate_prepared_output_resource_facts(&prepared, &admitted).unwrap();

        let mut understated_payload = admitted.clone();
        understated_payload.encoded_payload_bytes_upper_bound.0 -= 1;
        assert!(validate_prepared_output_resource_facts(&prepared, &understated_payload).is_err());

        let mut understated_transport = admitted;
        understated_transport.transport_closure_bytes_upper_bound.0 -= 1;
        assert!(
            validate_prepared_output_resource_facts(&prepared, &understated_transport).is_err()
        );
    }
}
