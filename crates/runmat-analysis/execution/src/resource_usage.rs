use runmat_analysis_fea::execution::AnalysisNumericalObjectKind;
use runmat_analysis_fea::planning::resource::{
    record_resource_usage, AnalysisResourceUsage, ByteCount, ResourceUsageEvidence,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
use runmat_execution_runner::AttemptSuccess;

use crate::{
    prepare_analysis_objects, AnalysisBridgeError, AnalysisBridgeResult,
    ImportedStructuralLinearResult, PreparedAnalysisObjects, PreparedStructuralLinearProgram,
};

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedStructuralResourceUsage {
    pub evidence: ResourceUsageEvidence,
    pub objects: PreparedAnalysisObjects,
}

/// Adapts scheduler-owned accounting into source-bound FEA evidence without giving FEA a second
/// worker-metering implementation.
pub fn prepare_structural_resource_usage(
    prepared: &PreparedStructuralLinearProgram,
    result: &ImportedStructuralLinearResult,
    success: &AttemptSuccess,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedStructuralResourceUsage> {
    success.validate_accounting().map_err(|error| {
        AnalysisBridgeError::Invalid(format!("invalid scheduler resource accounting: {error}"))
    })?;
    let accounting = success.resources.as_ref().ok_or_else(|| {
        AnalysisBridgeError::Invalid(
            "structural result has no scheduler-bound resource accounting".into(),
        )
    })?;
    if accounting.artifact_bytes != result.result_object_bytes {
        return Err(AnalysisBridgeError::Identity(
            "resource accounting artifact bytes differ from the imported structural result",
        ));
    }
    let actual = AnalysisResourceUsage {
        scheduler: accounting.usage.clone(),
        peak_cpu_millicores: accounting.allocation.cpu_millicores,
        peak_memory_bytes: ByteCount(accounting.allocation.memory_bytes),
        peak_scratch_bytes: ByteCount(accounting.allocation.scratch_bytes),
        artifact_bytes: ByteCount(accounting.artifact_bytes),
    };
    let evidence = record_resource_usage(
        &prepared.resource_plan,
        prepared.host.workload.identity.digest(),
        result.identity.digest(),
        accounting.attempt_id,
        accounting.worker_id,
        accounting.allocation.clone(),
        actual,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourceUsage,
        evidence.identity().digest(),
        &evidence
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        chunk_bytes,
        ObjectNamespace::ResultValue,
        limits,
    )?;
    // Keep the value role explicit at the boundary used by publication preparation.
    objects.references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    Ok(PreparedStructuralResourceUsage { evidence, objects })
}

pub fn validate_structural_resource_usage(
    prepared: &PreparedStructuralLinearProgram,
    result: &ImportedStructuralLinearResult,
    usage: &PreparedStructuralResourceUsage,
    access: &ArtifactAccess,
) -> AnalysisBridgeResult<()> {
    let decoded = ResourceUsageEvidence::from_canonical_bytes(&usage.objects.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    usage
        .objects
        .references(access, ValueRefKind::ResultObject)?;
    if decoded != usage.evidence
        || decoded.identity().digest() != usage.objects.manifest.logical_identity
        || decoded.plan != prepared.resource_plan.identity
        || decoded.workload_digest != prepared.host.workload.identity.digest()
        || decoded.result_digest != result.identity.digest()
    {
        return Err(AnalysisBridgeError::Identity(
            "resource usage evidence differs from its structural sources",
        ));
    }
    Ok(())
}
