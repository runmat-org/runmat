use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisPayloadPartition, AnalysisWorkloadIdentity,
    StructuralTransientBlockRequest, StructuralTransientCheckpoint,
    StructuralTransientCheckpointScope, StructuralTransientPhysicalSample,
    StructuralTransientSegment, StructuralTransientSegmentManifest,
    StructuralTransientSegmentSample, StructuralTransientSegmentSampleIdentity,
    StructuralTransientWorkload,
};

pub fn prepare_structural_transient_workload_objects(
    workload: &StructuralTransientWorkload,
    identity: AnalysisWorkloadIdentity,
    chunk_bytes: usize,
    namespace: ObjectNamespace,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    let bytes = workload.canonical_bytes()?;
    prepare_analysis_objects(
        AnalysisNumericalObjectKind::StructuralTransientWorkload,
        identity.digest(),
        &bytes,
        chunk_bytes,
        namespace,
        limits,
    )
}

pub fn decode_prepared_structural_transient_workload(
    prepared: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StructuralTransientWorkload> {
    if prepared.manifest.kind != AnalysisNumericalObjectKind::StructuralTransientWorkload {
        return Err(AnalysisBridgeError::Invalid(
            "expected a structural transient workload object".into(),
        ));
    }
    Ok(StructuralTransientWorkload::from_canonical_bytes(
        &prepared.payload(),
    )?)
}
use runmat_execution::value::ValueRef;
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};

use crate::{
    import_analysis_partition, prepare_analysis_objects, prepare_partitioned_analysis_objects,
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisPartitionImport, PreparedAnalysisObjects,
};

pub fn prepare_structural_transient_block_request_objects(
    request: &StructuralTransientBlockRequest,
    chunk_bytes: usize,
    namespace: ObjectNamespace,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    let bytes = request.canonical_bytes()?;
    prepare_analysis_objects(
        AnalysisNumericalObjectKind::StructuralTransientBlockRequest,
        request.identity.digest(),
        &bytes,
        chunk_bytes,
        namespace,
        limits,
    )
}

pub fn prepare_structural_transient_scope_objects(
    scope: &StructuralTransientCheckpointScope,
    chunk_bytes: usize,
    namespace: ObjectNamespace,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    let bytes = scope.canonical_bytes()?;
    prepare_analysis_objects(
        AnalysisNumericalObjectKind::StructuralTransientCheckpointScope,
        scope.identity()?.digest(),
        &bytes,
        chunk_bytes,
        namespace,
        limits,
    )
}

pub fn decode_prepared_structural_transient_scope(
    prepared: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StructuralTransientCheckpointScope> {
    if prepared.manifest.kind != AnalysisNumericalObjectKind::StructuralTransientCheckpointScope {
        return Err(AnalysisBridgeError::Invalid(
            "expected a structural transient checkpoint scope object".into(),
        ));
    }
    let scope = StructuralTransientCheckpointScope::from_canonical_bytes(&prepared.payload())?;
    if prepared.manifest.logical_identity != scope.identity()?.digest() {
        return Err(AnalysisBridgeError::Identity(
            "structural transient checkpoint scope object has the wrong logical identity",
        ));
    }
    Ok(scope)
}

pub fn decode_prepared_structural_transient_block_request(
    prepared: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StructuralTransientBlockRequest> {
    if prepared.manifest.kind != AnalysisNumericalObjectKind::StructuralTransientBlockRequest {
        return Err(AnalysisBridgeError::Invalid(
            "expected a structural transient block request object".into(),
        ));
    }
    let request = StructuralTransientBlockRequest::from_canonical_bytes(&prepared.payload())?;
    if prepared.manifest.logical_identity != request.identity.digest() {
        return Err(AnalysisBridgeError::Identity(
            "structural transient block request object has the wrong logical identity",
        ));
    }
    Ok(request)
}

pub fn prepare_structural_transient_checkpoint_objects(
    checkpoint: &StructuralTransientCheckpoint,
    chunk_bytes: usize,
    namespace: ObjectNamespace,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    let bytes = checkpoint.canonical_bytes()?;
    prepare_analysis_objects(
        AnalysisNumericalObjectKind::StructuralTransientCheckpoint,
        checkpoint.identity.digest(),
        &bytes,
        chunk_bytes,
        namespace,
        limits,
    )
}

pub fn decode_prepared_structural_transient_checkpoint(
    prepared: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StructuralTransientCheckpoint> {
    if prepared.manifest.kind != AnalysisNumericalObjectKind::StructuralTransientCheckpoint {
        return Err(AnalysisBridgeError::Invalid(
            "expected a structural transient checkpoint object".into(),
        ));
    }
    let checkpoint = StructuralTransientCheckpoint::from_canonical_bytes(&prepared.payload())?;
    if prepared.manifest.logical_identity != checkpoint.identity.digest() {
        return Err(AnalysisBridgeError::Identity(
            "structural transient checkpoint object has the wrong logical identity",
        ));
    }
    Ok(checkpoint)
}

pub fn prepare_structural_transient_segment_objects(
    segment: &StructuralTransientSegment,
    chunk_bytes: usize,
    namespace: ObjectNamespace,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    segment.validate()?;
    let manifest_bytes = segment.manifest().canonical_bytes()?;
    let initial_bytes = segment
        .initial_sample()
        .map(StructuralTransientPhysicalSample::canonical_bytes)
        .transpose()?;
    let sample_bytes = segment
        .samples()
        .iter()
        .map(StructuralTransientSegmentSample::canonical_bytes)
        .collect::<Result<Vec<_>, _>>()?;
    let mut partitions =
        Vec::with_capacity(sample_bytes.len() + 1 + usize::from(initial_bytes.is_some()));
    partitions.push(AnalysisPayloadPartition {
        logical_identity: segment.manifest().identity.digest(),
        payload: &manifest_bytes,
    });
    if let (Some(initial), Some(payload)) = (segment.initial_sample(), initial_bytes.as_ref()) {
        partitions.push(AnalysisPayloadPartition {
            logical_identity: initial.identity.digest(),
            payload,
        });
    }
    partitions.extend(
        segment
            .samples()
            .iter()
            .zip(&sample_bytes)
            .map(|(sample, payload)| AnalysisPayloadPartition {
                logical_identity: sample.identity.digest(),
                payload,
            }),
    );
    prepare_partitioned_analysis_objects(
        AnalysisNumericalObjectKind::StructuralTransientSegment,
        segment.manifest().identity.digest(),
        &partitions,
        chunk_bytes,
        namespace,
        limits,
    )
}

pub fn decode_prepared_structural_transient_segment(
    prepared: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StructuralTransientSegment> {
    if prepared.manifest.kind != AnalysisNumericalObjectKind::StructuralTransientSegment {
        return Err(AnalysisBridgeError::Invalid(
            "expected a structural transient segment object".into(),
        ));
    }
    let manifest = StructuralTransientSegmentManifest::from_canonical_bytes(
        &prepared.partition_payload(prepared.manifest.logical_identity)?,
    )?;
    let expected_partitions = std::iter::once(manifest.identity.digest())
        .chain(manifest.initial_sample.map(|identity| identity.digest()))
        .chain(manifest.samples.iter().map(|entry| entry.sample.digest()))
        .collect::<Vec<_>>();
    if prepared.manifest.logical_identity != manifest.identity.digest()
        || prepared.manifest.partitions.len() != expected_partitions.len()
        || prepared
            .manifest
            .partitions
            .iter()
            .zip(expected_partitions)
            .any(|(partition, expected)| partition.logical_identity != expected)
    {
        return Err(AnalysisBridgeError::Identity(
            "structural transient segment partitions differ from its semantic manifest",
        ));
    }
    let initial_sample = manifest
        .initial_sample
        .map(|identity| {
            StructuralTransientPhysicalSample::from_canonical_bytes(
                &prepared.partition_payload(identity.digest())?,
            )
            .map_err(AnalysisBridgeError::from)
        })
        .transpose()?;
    let samples = manifest
        .samples
        .iter()
        .map(|entry| {
            StructuralTransientSegmentSample::from_canonical_bytes(
                &prepared.partition_payload(entry.sample.digest())?,
            )
            .map_err(AnalysisBridgeError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(StructuralTransientSegment::from_manifest_and_samples(
        manifest,
        initial_sample,
        samples,
    )?)
}

pub struct StructuralTransientSegmentSampleImport<'a> {
    pub root: &'a ValueRef,
    pub expected_segment: Digest,
    pub expected_payload_digest: Digest,
    pub sample: StructuralTransientSegmentSampleIdentity,
    pub access: &'a ArtifactAccess,
    pub limits: ObjectInventoryLimits,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedStructuralTransientSegmentSample {
    pub manifest: StructuralTransientSegmentManifest,
    pub sample: StructuralTransientSegmentSample,
}

pub fn import_structural_transient_segment_sample(
    source: &impl CacheImport,
    import: StructuralTransientSegmentSampleImport<'_>,
) -> AnalysisBridgeResult<ImportedStructuralTransientSegmentSample> {
    let manifest_partition = import_analysis_partition(
        source,
        AnalysisPartitionImport {
            root: import.root,
            expected_kind: AnalysisNumericalObjectKind::StructuralTransientSegment,
            expected_logical_identity: import.expected_segment,
            expected_payload_digest: import.expected_payload_digest,
            partition_identity: import.expected_segment,
            access: import.access,
            limits: import.limits,
        },
    )?;
    let manifest =
        StructuralTransientSegmentManifest::from_canonical_bytes(&manifest_partition.payload())?;
    let expected_partitions = std::iter::once(manifest.identity.digest())
        .chain(manifest.initial_sample.map(|identity| identity.digest()))
        .chain(manifest.samples.iter().map(|entry| entry.sample.digest()))
        .collect::<Vec<_>>();
    if manifest.identity.digest() != import.expected_segment
        || manifest_partition.manifest.partitions.len() != expected_partitions.len()
        || manifest_partition
            .manifest
            .partitions
            .iter()
            .zip(expected_partitions)
            .any(|(partition, expected)| partition.logical_identity != expected)
        || !manifest
            .samples
            .iter()
            .any(|entry| entry.sample == import.sample)
    {
        return Err(AnalysisBridgeError::Identity(
            "requested transient segment sample is absent from the exact manifest",
        ));
    }
    let sample_partition = import_analysis_partition(
        source,
        AnalysisPartitionImport {
            root: import.root,
            expected_kind: AnalysisNumericalObjectKind::StructuralTransientSegment,
            expected_logical_identity: import.expected_segment,
            expected_payload_digest: import.expected_payload_digest,
            partition_identity: import.sample.digest(),
            access: import.access,
            limits: import.limits,
        },
    )?;
    let sample =
        StructuralTransientSegmentSample::from_canonical_bytes(&sample_partition.payload())?;
    sample.validate_against(&manifest.request)?;
    if sample.identity != import.sample {
        return Err(AnalysisBridgeError::Identity(
            "imported transient segment sample differs from its requested identity",
        ));
    }
    Ok(ImportedStructuralTransientSegmentSample { manifest, sample })
}
