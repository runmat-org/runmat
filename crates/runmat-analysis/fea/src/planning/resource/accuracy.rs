//! Shared receipt-based accuracy evaluation admission, independent of physics family.
use super::{
    build_resource_plan, AccuracyEvaluationResourceDemand, ByteCount, Count, FeaResourceDemand,
    FeaResourcePhase, FeaResourcePlan, FeaResourcePlanningInput, PreparedPhaseResourceFacts,
    ResourcePlanningError, RESOURCE_ESTIMATION_MODEL_REVISION,
};
use crate::{
    execution::AccuracyEvaluationRequest,
    results::accuracy::{accuracy_evaluation_resource_profile, AccuracyEvidence},
};

pub fn build_accuracy_resource_plan(
    request: &AccuracyEvaluationRequest,
    evidence: &[AccuracyEvidence],
    facts: PreparedPhaseResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    request
        .validate()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let profile = accuracy_evaluation_resource_profile(request, evidence)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    super::post_process::validate_projected_output(
        &facts.output,
        profile.output_encoded_bytes_upper_bound,
        "accuracy evaluation",
    )?;
    // Source verification may reconstruct mesh projections and authored observation plans.
    // Reserve that transient workspace separately from resident decoded transport inputs.
    let context_workspace = facts
        .inputs
        .decoded_resident_bytes_admission_budget
        .0
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(profile.workspace_bytes_upper_bound))
        .ok_or(ResourcePlanningError::Overflow(
            "accuracy source verification workspace",
        ))?;
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::AccuracyEvaluation(request.identity()),
        demand: FeaResourceDemand::AccuracyEvaluation(AccuracyEvaluationResourceDemand {
            request: request.identity(),
            source_result: request.source_result,
            criterion_count: Count(
                u64::try_from(request.plan.criteria.len())
                    .map_err(|_| ResourcePlanningError::Overflow("accuracy criteria"))?,
            ),
            evidence_count: Count(
                u64::try_from(evidence.len())
                    .map_err(|_| ResourcePlanningError::Overflow("accuracy evidence"))?,
            ),
            inputs: facts.inputs,
            evaluator_workspace_bytes_upper_bound: ByteCount(context_workspace),
            output: facts.output,
        }),
        output_set: Some(facts.output_set),
        worker: super::post_process::post_process_worker(),
        accuracy_ceiling: super::post_process::post_process_ceiling(),
    })
}
