use std::collections::BTreeMap;

use runmat_analysis_fea::execution::AnalysisNumericalObjectKind;
use runmat_analysis_fea::planning::resource::{
    record_resource_graph_usage, record_resource_usage, FeaResourcePlan, ResourceUsageAttempt,
    ResourceUsageEvidence,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
use runmat_execution_runner::{AttemptResourceAccounting, AttemptSuccess};

use crate::{
    prepare_analysis_objects, AnalysisBridgeError, AnalysisBridgeResult,
    ImportedStructuralLinearResult, PreparedAnalysisObjects, PreparedStructuralLinearProgram,
};

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedResourceUsage {
    pub evidence: ResourceUsageEvidence,
    pub objects: PreparedAnalysisObjects,
}

/// Publishes one scheduler-measured task graph against its terminal numerical result.
///
/// The caller supplies the terminal task's plan as the primary plan; every distinct phase plan is
/// recovered from the supplied measured attempts. The caller must supply every executed phase
/// in the claimed scope; this function cannot detect omitted attempts or unplanned downstream
/// work. It rejects a primary plan that has no supplied scheduler accounting.
pub fn prepare_planned_resource_graph_usage(
    primary_plan: &FeaResourcePlan,
    workload_digest: runmat_execution::Digest,
    result_digest: runmat_execution::Digest,
    graph: &PlannedResourceUsageGraph,
    access: &ArtifactAccess,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedResourceUsage> {
    let mut plans = BTreeMap::new();
    let mut attempts = Vec::with_capacity(graph.attempts.len());
    for attempt in &graph.attempts {
        if let Some(existing) = plans.insert(attempt.plan.identity, attempt.plan.clone()) {
            if existing != attempt.plan {
                return Err(AnalysisBridgeError::Identity(
                    "one resource-plan identity names distinct phase plans",
                ));
            }
        }
        attempts.push(usage_attempt(attempt.plan.identity, &attempt.accounting));
    }
    if !plans.contains_key(&primary_plan.identity) {
        return Err(AnalysisBridgeError::Identity(
            "resource graph omits its terminal primary plan",
        ));
    }
    plans.remove(&primary_plan.identity);
    let evidence = record_resource_graph_usage(
        primary_plan,
        plans.into_values().collect(),
        workload_digest,
        result_digest,
        attempts,
        graph.wall_millis,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    prepare_usage_objects(evidence, access, chunk_bytes, limits)
}

/// Adapts scheduler-owned accounting into source-bound FEA evidence without giving FEA a second
/// worker-metering implementation.
pub fn prepare_structural_resource_usage(
    prepared: &PreparedStructuralLinearProgram,
    result: &ImportedStructuralLinearResult,
    success: &AttemptSuccess,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedResourceUsage> {
    success.validate_accounting().map_err(|error| {
        AnalysisBridgeError::Invalid(format!("invalid scheduler resource accounting: {error}"))
    })?;
    let accounting = success.resources().ok_or_else(|| {
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
        vec![usage_attempt(prepared.resource_plan.identity, accounting)],
        accounting.resources.usage.wall_millis,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    prepare_usage_objects(
        evidence,
        &prepared.host.artifact_access,
        chunk_bytes,
        limits,
    )
}

/// Extends solve accounting with a measured downstream execution graph while retaining one
/// canonical FEA evidence object. Generic execution owns interval measurement and aggregation;
/// this bridge only binds those records to the workload and numerical-result identities.
pub fn extend_structural_resource_usage(
    prepared: &PreparedStructuralLinearProgram,
    result: &ImportedStructuralLinearResult,
    baseline: &PreparedResourceUsage,
    additional: &PlannedResourceUsageGraph,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedResourceUsage> {
    validate_structural_resource_usage(prepared, result, baseline, &prepared.host.artifact_access)?;
    extend_resource_usage(
        baseline,
        additional,
        &prepared.host.artifact_access,
        crate::AnalysisOutputPolicy::new(chunk_bytes, limits)?,
    )
}

/// Extend a source-bound baseline with measured sequential downstream work. Preserve every
/// prior phase plan; repeated extensions must not drop earlier observation or accuracy plans.
pub fn extend_resource_usage(
    baseline: &PreparedResourceUsage,
    additional: &PlannedResourceUsageGraph,
    access: &ArtifactAccess,
    policy: crate::AnalysisOutputPolicy,
) -> AnalysisBridgeResult<PreparedResourceUsage> {
    let primary = baseline
        .evidence
        .plans
        .iter()
        .find(|plan| plan.identity == baseline.evidence.primary_plan)
        .ok_or(AnalysisBridgeError::Identity(
            "resource usage baseline omits primary plan",
        ))?;
    runmat_analysis_fea::planning::resource::validate_resource_usage_evidence(
        primary,
        &baseline.evidence,
    )?;
    if baseline.objects.manifest.kind != AnalysisNumericalObjectKind::ResourceUsage
        || baseline.objects.manifest.logical_identity != baseline.evidence.identity().digest()
        || baseline.objects.payload() != baseline.evidence.canonical_bytes()?
    {
        return Err(AnalysisBridgeError::Identity(
            "resource usage baseline differs from canonical objects",
        ));
    }
    let mut attempts = baseline.evidence.attempts.clone();
    attempts.extend(
        additional
            .attempts
            .iter()
            .map(|attempt| usage_attempt(attempt.plan.identity, &attempt.accounting)),
    );
    let mut plans = BTreeMap::new();
    for plan in &baseline.evidence.plans {
        plans.insert(plan.identity, plan.clone());
    }
    for attempt in &additional.attempts {
        if let Some(existing) = plans.insert(attempt.plan.identity, attempt.plan.clone()) {
            if existing != attempt.plan {
                return Err(AnalysisBridgeError::Identity(
                    "one resource-plan identity names distinct phase plans",
                ));
            }
        }
    }
    let graph_wall_millis = baseline
        .evidence
        .actual
        .scheduler
        .wall_millis
        .checked_add(additional.wall_millis)
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("resource graph wall time overflowed".into())
        })?;
    plans.remove(&primary.identity);
    let evidence = record_resource_graph_usage(
        primary,
        plans.into_values().collect(),
        baseline.evidence.workload_digest,
        baseline.evidence.result_digest,
        attempts,
        graph_wall_millis,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    prepare_usage_objects(
        evidence,
        access,
        policy.chunk_bytes(),
        policy.inventory_limits(),
    )
}

fn usage_attempt(
    plan: runmat_analysis_fea::planning::resource::FeaResourcePlanIdentity,
    accounting: &AttemptResourceAccounting,
) -> ResourceUsageAttempt {
    ResourceUsageAttempt {
        attempt_id: accounting.attempt_id,
        worker_id: accounting.worker_id,
        plan,
        resources: accounting.resources.clone(),
    }
}

fn prepare_usage_objects(
    evidence: ResourceUsageEvidence,
    access: &ArtifactAccess,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedResourceUsage> {
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
    objects.references(access, ValueRefKind::ResultObject)?;
    Ok(PreparedResourceUsage { evidence, objects })
}

pub fn validate_structural_resource_usage(
    prepared: &PreparedStructuralLinearProgram,
    result: &ImportedStructuralLinearResult,
    usage: &PreparedResourceUsage,
    access: &ArtifactAccess,
) -> AnalysisBridgeResult<()> {
    let decoded = ResourceUsageEvidence::from_canonical_bytes(&usage.objects.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    usage
        .objects
        .references(access, ValueRefKind::ResultObject)?;
    if decoded != usage.evidence
        || decoded.identity().digest() != usage.objects.manifest.logical_identity
        || decoded.primary_plan != prepared.resource_plan.identity
        || decoded.workload_digest != prepared.host.workload.identity.digest()
        || decoded.result_digest != result.identity.digest()
    {
        return Err(AnalysisBridgeError::Identity(
            "resource usage evidence differs from its structural sources",
        ));
    }
    Ok(())
}

mod graph;
mod observation;
pub use graph::{PlannedResourceUsageAttempt, PlannedResourceUsageGraph};
pub use observation::validate_observation_resource_usage;
