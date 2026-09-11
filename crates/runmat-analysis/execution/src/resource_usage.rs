use runmat_analysis_fea::execution::AnalysisNumericalObjectKind;
use runmat_analysis_fea::planning::resource::{
    record_resource_usage, ResourceUsageAttempt, ResourceUsageEvidence,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
use runmat_execution_runner::{
    AttemptResourceAccounting, AttemptResourceAggregate, AttemptSuccess,
};

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
    if accounting.resources.artifact_bytes != result.result_object_bytes {
        return Err(AnalysisBridgeError::Identity(
            "resource accounting artifact bytes differ from the imported structural result",
        ));
    }
    let evidence = record_resource_usage(
        &prepared.resource_plan,
        prepared.host.workload.identity.digest(),
        result.identity.digest(),
        vec![usage_attempt(accounting)],
        accounting.resources.usage.wall_millis,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    prepare_usage_objects(prepared, evidence, chunk_bytes, limits)
}

/// Extends solve accounting with a measured downstream execution graph while retaining one
/// canonical FEA evidence object. Generic execution owns interval measurement and aggregation;
/// this bridge only binds those records to the scientific workload and result identities.
pub fn extend_structural_resource_usage(
    prepared: &PreparedStructuralLinearProgram,
    result: &ImportedStructuralLinearResult,
    baseline: &PreparedStructuralResourceUsage,
    additional: &AttemptResourceAggregate,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedStructuralResourceUsage> {
    validate_structural_resource_usage(prepared, result, baseline, &prepared.host.artifact_access)?;
    additional.validate().map_err(|error| {
        AnalysisBridgeError::Invalid(format!("invalid execution-graph accounting: {error}"))
    })?;
    let mut attempts = baseline.evidence.attempts.clone();
    attempts.extend(additional.attempts.iter().map(usage_attempt));
    let graph_wall_millis = baseline
        .evidence
        .actual
        .scheduler
        .wall_millis
        .checked_add(additional.summary.usage.wall_millis)
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("resource graph wall time overflowed".into())
        })?;
    let evidence = record_resource_usage(
        &prepared.resource_plan,
        prepared.host.workload.identity.digest(),
        result.identity.digest(),
        attempts,
        graph_wall_millis,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    prepare_usage_objects(prepared, evidence, chunk_bytes, limits)
}

fn usage_attempt(accounting: &AttemptResourceAccounting) -> ResourceUsageAttempt {
    ResourceUsageAttempt {
        attempt_id: accounting.attempt_id,
        worker_id: accounting.worker_id,
        resources: accounting.resources.clone(),
    }
}

fn prepare_usage_objects(
    prepared: &PreparedStructuralLinearProgram,
    evidence: ResourceUsageEvidence,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedStructuralResourceUsage> {
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
