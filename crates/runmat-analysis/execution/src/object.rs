use runmat_analysis_fea::execution::{
    chunk_numerical_payload, chunk_partitioned_numerical_payload, AnalysisNumericalObjectKind,
    AnalysisObjectManifest, AnalysisPayloadPartition, AnalysisPayloadPartitionDescriptor,
    ChunkedAnalysisPayload, ANALYSIS_CHUNK_MEDIA_TYPE, ANALYSIS_CHUNK_VALUE_SCHEMA,
    ANALYSIS_MANIFEST_MEDIA_TYPE, ANALYSIS_MANIFEST_VALUE_SCHEMA,
};
use runmat_analysis_fea::planning::resource::{
    ByteCount, PreparedInputResourceFacts, PreparedOutputResourceFacts,
};
use runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION;
use runmat_execution::value::{ValueRef, ValueRefKind};
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::{validate_inventory, ObjectInventoryLimits};
use runmat_execution_artifact::{ArtifactAccess, LogicalObject, ObjectDescriptor, ObjectNamespace};

use crate::{AnalysisBridgeError, AnalysisBridgeResult};

mod import;
pub use import::{import_analysis_objects, import_analysis_partition};
pub(crate) use import::{import_analysis_objects_admitted, AnalysisObjectImport};
mod admission;
pub(crate) use admission::{read_input, AnalysisInputAdmission};
mod conversion;
mod retained_storage;
mod validation;
use conversion::logical_object;
pub(crate) use conversion::prepare_chunked_analysis_objects;
pub use conversion::{
    projected_chunked_output_resource_facts, projected_transport_output_resource_facts,
};
pub(crate) use retained_storage::logical_objects_storage;

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
        Ok(self
            .partition_chunks(identity)?
            .iter()
            .flat_map(|object| object.bytes.iter().copied())
            .collect())
    }

    /// Authenticate one semantic payload without constructing a joined buffer.
    fn partition_chunks(&self, identity: Digest) -> AnalysisBridgeResult<&[LogicalObject]> {
        let partition = self
            .manifest
            .partitions
            .iter()
            .find(|p| p.logical_identity == identity)
            .ok_or_else(|| {
                AnalysisBridgeError::Invalid(
                    "semantic payload is absent from the prepared numerical object".into(),
                )
            })?;
        let start = usize::try_from(partition.first_chunk_ordinal)
            .ok()
            .and_then(|n| n.checked_add(1));
        let end = start.and_then(|n| n.checked_add(partition.chunk_count as usize));
        let chunks = start
            .zip(end)
            .and_then(|(start, end)| self.objects.get(start..end))
            .ok_or(AnalysisBridgeError::Identity(
                "prepared numerical object omits semantic payload chunks",
            ))?;
        let bytes = chunks
            .iter()
            .try_fold(0u64, |total, object| {
                total.checked_add(object.bytes.len() as u64)
            })
            .ok_or(AnalysisBridgeError::Identity(
                "prepared semantic payload length overflows",
            ))?;
        if bytes != partition.encoded_length
            || Digest::sha256_parts(chunks.iter().map(|o| &o.bytes)) != partition.payload_digest
        {
            return Err(AnalysisBridgeError::Identity(
                "prepared semantic payload differs from its descriptor",
            ));
        }
        Ok(chunks)
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
    prepared_input_resource_facts_from_refs(inputs.iter())
}

/// Reprojects numerical inputs without cloning their potentially large prepared payloads.
pub fn prepared_input_resource_facts_without_resource_plan(
    inputs: &[PreparedAnalysisObjects],
) -> AnalysisBridgeResult<PreparedInputResourceFacts> {
    prepared_input_resource_facts_from_refs(
        inputs
            .iter()
            .filter(|input| input.manifest.kind != AnalysisNumericalObjectKind::ResourcePlan),
    )
}

struct ResourceProfileControl;
impl runmat_canonical_codec::CanonicalCodecControl for ResourceProfileControl {
    fn is_cancelled(&self) -> bool {
        runmat_analysis_fea::is_fea_cancelled()
    }
}

/// Borrowed input inventory profiling without copying retained payloads.
/// Producing-contract and complete source authentication remain the consuming phase's responsibility.
pub fn prepared_input_resource_facts_from_refs<'a>(
    inputs: impl Iterator<Item = &'a PreparedAnalysisObjects>,
) -> AnalysisBridgeResult<PreparedInputResourceFacts> {
    let inputs = inputs.collect::<Vec<_>>();
    let encoded_payload_bytes = inputs.iter().try_fold(0_u64, |total, input| {
        total
            .checked_add(input.manifest.encoded_length)
            .ok_or_else(|| AnalysisBridgeError::Invalid("input payload bytes overflow".into()))
    })?;
    let mut unique = std::collections::BTreeMap::new();
    for input in &inputs {
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
                    let chunks = input.partition_chunks(partition.logical_identity)?;
                    // Inputs decode directly into typed values; no value tree is built.
                    let estimate = runmat_canonical_codec::canonical_chunked_payload_typed_resident_bytes_estimate(
                        chunks.iter().map(|object| object.bytes.as_slice()), &ResourceProfileControl,
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
    runmat_analysis_fea::planning::resource::projected_canonical_output_resource_facts(
        encoded_payload_bytes_upper_bound,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
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
        return Err(AnalysisBridgeError::Invalid(format!(
            "prepared output exceeds its admitted semantic or transport envelope: encoded {} of {}, closure {actual_closure} of {}",
            output.manifest.encoded_length,
            admitted.encoded_payload_bytes_upper_bound.0,
            admitted.transport_closure_bytes_upper_bound.0,
        )));
    }
    Ok(())
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
            schema_version: VALUE_PAYLOAD_SCHEMA_VERSION,
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

fn validate_partition_payloads(
    manifest: &AnalysisObjectManifest,
    objects: &[LogicalObject],
) -> AnalysisBridgeResult<()> {
    for partition in &manifest.partitions {
        let start = partition.first_chunk_ordinal as usize + 1;
        let end = start + partition.chunk_count as usize;
        if Digest::sha256_parts(objects[start..end].iter().map(|object| &object.bytes))
            != partition.payload_digest
        {
            return Err(AnalysisBridgeError::Identity(
                "analysis payload partition differs from its descriptor",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod resource_tests;
