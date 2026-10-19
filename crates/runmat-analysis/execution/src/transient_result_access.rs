use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisPayloadPartition, SolverResultIdentity,
    StructuralTransientNumericalReceipt, StructuralTransientPhysicalSample,
    StructuralTransientPhysicalSampleIdentity, StructuralTransientResultAssemblyRequest,
    StructuralTransientSolverResult, StructuralTransientSolverResultManifest,
};
use runmat_execution::value::ValueRef;
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};

use crate::{
    import_analysis_partition, prepare_analysis_objects, prepare_partitioned_analysis_objects,
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisPartitionImport, PreparedAnalysisObjects,
};

pub fn prepare_structural_transient_result_assembly_request_objects(
    request: &StructuralTransientResultAssemblyRequest,
    chunk_bytes: usize,
    namespace: ObjectNamespace,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    prepare_analysis_objects(
        AnalysisNumericalObjectKind::StructuralTransientResultAssemblyRequest,
        request.identity.digest(),
        &request.canonical_bytes()?,
        chunk_bytes,
        namespace,
        limits,
    )
}

pub fn decode_prepared_structural_transient_result_assembly_request(
    prepared: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StructuralTransientResultAssemblyRequest> {
    if prepared.manifest.kind
        != AnalysisNumericalObjectKind::StructuralTransientResultAssemblyRequest
    {
        return Err(AnalysisBridgeError::Invalid(
            "expected a structural transient result assembly request object".into(),
        ));
    }
    let request =
        StructuralTransientResultAssemblyRequest::from_canonical_bytes(&prepared.payload())?;
    if prepared.manifest.logical_identity != request.identity.digest() {
        return Err(AnalysisBridgeError::Identity(
            "structural transient result assembly request has the wrong logical identity",
        ));
    }
    Ok(request)
}

pub fn prepare_structural_transient_solver_result_objects(
    result: &StructuralTransientSolverResult,
    chunk_bytes: usize,
    namespace: ObjectNamespace,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    result.validate()?;
    let manifest_bytes = result.manifest().canonical_bytes()?;
    let sample_bytes = result
        .samples()
        .iter()
        .map(StructuralTransientPhysicalSample::canonical_bytes)
        .collect::<Result<Vec<_>, _>>()?;
    let mut partitions = Vec::with_capacity(sample_bytes.len() + 1);
    partitions.push(AnalysisPayloadPartition {
        logical_identity: result.manifest().identity.digest(),
        payload: &manifest_bytes,
    });
    partitions.extend(
        result
            .samples()
            .iter()
            .zip(&sample_bytes)
            .map(|(sample, payload)| AnalysisPayloadPartition {
                logical_identity: sample.identity.digest(),
                payload,
            }),
    );
    prepare_partitioned_analysis_objects(
        AnalysisNumericalObjectKind::SolverResult,
        result.manifest().identity.digest(),
        &partitions,
        chunk_bytes,
        namespace,
        limits,
    )
}

pub fn decode_prepared_structural_transient_solver_result(
    prepared: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StructuralTransientSolverResult> {
    if prepared.manifest.kind != AnalysisNumericalObjectKind::SolverResult {
        return Err(AnalysisBridgeError::Invalid(
            "expected a structural transient solver-result object".into(),
        ));
    }
    let manifest = StructuralTransientSolverResultManifest::from_canonical_bytes(
        &prepared.partition_payload(prepared.manifest.logical_identity)?,
    )?;
    validate_result_partitions(prepared, &manifest)?;
    let samples = manifest
        .samples
        .iter()
        .map(|entry| {
            StructuralTransientPhysicalSample::from_canonical_bytes(
                &prepared.partition_payload(entry.sample.digest())?,
            )
            .map_err(AnalysisBridgeError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(StructuralTransientSolverResult::from_manifest_and_samples(
        manifest, samples,
    )?)
}

pub fn prepare_structural_transient_numerical_receipt_objects(
    receipt: &StructuralTransientNumericalReceipt,
    chunk_bytes: usize,
    namespace: ObjectNamespace,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    prepare_analysis_objects(
        AnalysisNumericalObjectKind::NumericalReceipt,
        receipt.identity.digest(),
        &receipt.canonical_bytes()?,
        chunk_bytes,
        namespace,
        limits,
    )
}

pub fn decode_prepared_structural_transient_numerical_receipt(
    prepared: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<StructuralTransientNumericalReceipt> {
    if prepared.manifest.kind != AnalysisNumericalObjectKind::NumericalReceipt {
        return Err(AnalysisBridgeError::Invalid(
            "expected a structural transient numerical receipt object".into(),
        ));
    }
    let receipt = StructuralTransientNumericalReceipt::from_canonical_bytes(&prepared.payload())?;
    if prepared.manifest.logical_identity != receipt.identity.digest() {
        return Err(AnalysisBridgeError::Identity(
            "structural transient numerical receipt has the wrong logical identity",
        ));
    }
    Ok(receipt)
}

pub struct StructuralTransientResultSampleImport<'a> {
    pub root: &'a ValueRef,
    pub expected_result: SolverResultIdentity,
    pub expected_payload_digest: Digest,
    pub sample: StructuralTransientPhysicalSampleIdentity,
    pub access: &'a ArtifactAccess,
    pub limits: ObjectInventoryLimits,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedStructuralTransientResultSample {
    pub manifest: StructuralTransientSolverResultManifest,
    pub sample: StructuralTransientPhysicalSample,
}

pub fn import_structural_transient_result_sample(
    source: &impl CacheImport,
    import: StructuralTransientResultSampleImport<'_>,
) -> AnalysisBridgeResult<ImportedStructuralTransientResultSample> {
    let manifest_partition = import_analysis_partition(
        source,
        AnalysisPartitionImport {
            root: import.root,
            expected_kind: AnalysisNumericalObjectKind::SolverResult,
            expected_logical_identity: import.expected_result.digest(),
            expected_payload_digest: import.expected_payload_digest,
            partition_identity: import.expected_result.digest(),
            access: import.access,
            limits: import.limits,
        },
    )?;
    let manifest = StructuralTransientSolverResultManifest::from_canonical_bytes(
        &manifest_partition.payload(),
    )?;
    validate_result_partition_manifest(&manifest_partition.manifest, &manifest)?;
    if manifest.identity != import.expected_result
        || !manifest
            .samples
            .iter()
            .any(|entry| entry.sample == import.sample)
    {
        return Err(AnalysisBridgeError::Identity(
            "requested structural transient result sample is absent",
        ));
    }
    let sample_partition = import_analysis_partition(
        source,
        AnalysisPartitionImport {
            root: import.root,
            expected_kind: AnalysisNumericalObjectKind::SolverResult,
            expected_logical_identity: import.expected_result.digest(),
            expected_payload_digest: import.expected_payload_digest,
            partition_identity: import.sample.digest(),
            access: import.access,
            limits: import.limits,
        },
    )?;
    let sample =
        StructuralTransientPhysicalSample::from_canonical_bytes(&sample_partition.payload())?;
    sample.validate_against(&manifest.scope)?;
    if sample.identity != import.sample {
        return Err(AnalysisBridgeError::Identity(
            "imported structural transient result sample has the wrong identity",
        ));
    }
    Ok(ImportedStructuralTransientResultSample { manifest, sample })
}

fn validate_result_partitions(
    prepared: &PreparedAnalysisObjects,
    manifest: &StructuralTransientSolverResultManifest,
) -> AnalysisBridgeResult<()> {
    validate_result_partition_manifest(&prepared.manifest, manifest)
}

fn validate_result_partition_manifest(
    object: &runmat_analysis_fea::execution::AnalysisObjectManifest,
    manifest: &StructuralTransientSolverResultManifest,
) -> AnalysisBridgeResult<()> {
    let expected = std::iter::once(manifest.identity.digest())
        .chain(manifest.samples.iter().map(|entry| entry.sample.digest()))
        .collect::<Vec<_>>();
    if object.logical_identity != manifest.identity.digest()
        || object.partitions.len() != expected.len()
        || object
            .partitions
            .iter()
            .zip(expected)
            .any(|(partition, expected)| partition.logical_identity != expected)
    {
        return Err(AnalysisBridgeError::Identity(
            "structural transient result partitions differ from its manifest",
        ));
    }
    Ok(())
}
