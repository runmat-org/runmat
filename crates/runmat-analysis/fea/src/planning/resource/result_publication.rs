//! Final receipt admission derived from immutable accuracy and measured source evidence.
use super::*;
use crate::execution::ResultPublicationRequest;
use crate::results::observation::ObservationPublicationCatalog;
use crate::results::{accuracy::AccuracyArtifact, publication::published_result_encoding_profile};
use runmat_canonical_codec::profile_serialized;

#[cfg(test)]
mod tests;

pub fn build_result_publication_resource_plan(
    request: &ResultPublicationRequest,
    accuracy: &AccuracyArtifact,
    usage: &ResourceUsageEvidence,
    observations: Option<&ObservationPublicationCatalog>,
    facts: PreparedPhaseResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let invalid = |error: &dyn std::fmt::Display| ResourcePlanningError::Invalid(error.to_string());
    if ResultPublicationRequest::new(accuracy, observations, usage).map_err(|e| invalid(&e))?
        != *request
    {
        return Err(ResourcePlanningError::Invalid(
            "publication admission sources differ from the source-bound request".into(),
        ));
    }
    let observation_execution_workspace = observations
        .map(ObservationPublicationCatalog::execution_workspace_bytes_estimate)
        .transpose()
        .map_err(|e| invalid(&e))?
        .unwrap_or(0);
    let profile =
        published_result_encoding_profile(request, accuracy, usage).map_err(|e| invalid(&e))?;
    // validate_identity rebuild retains cloned plans and attempts while aggregating a separate
    // Vec<ResourceAccounting>. Profiling complete attempts covers that second vector's fields,
    // accelerator strings and collection metadata (attempt IDs add only conservative overhead).
    let interval_workspace =
        profile_serialized(&usage.attempts, super::usage::RESOURCE_USAGE_LIMITS)
            .map_err(|e| invalid(&e))?
            .materialization_resident_bytes_estimate;
    // Each cloned plan is revalidated in sequence, reconstructing one planning input/estimate
    // and encoding its identity. No sum over plans: their validation workspaces do not overlap.
    let nested_plan_workspace = usage.plans.iter().try_fold(0_u64, |peak, plan| {
        let profile = profile_serialized(plan, super::usage::RESOURCE_USAGE_LIMITS)
            .map_err(|e| invalid(&e))?;
        Ok::<u64, ResourcePlanningError>(peak.max(profile.materialization_resident_bytes_estimate))
    })?;
    // The source materialization envelope covers the first cloned inventory, identity trees,
    // reconciliation records and scalar/map aggregators while the original decode stays live.
    let observation_index_workspace =
        crate::results::publication::observation_usage_index_bytes(usage.plans.len())
            .ok_or(ResourcePlanningError::Overflow("observation task index"))?;
    let validation = facts
        .inputs
        .decoded_resident_bytes_admission_budget
        .0
        .checked_add(interval_workspace)
        .and_then(|v| v.checked_add(nested_plan_workspace))
        .and_then(|v| v.checked_add(observation_index_workspace))
        .and_then(|v| v.checked_add(observation_execution_workspace))
        .ok_or(ResourcePlanningError::Overflow(
            "result publication validation workspace",
        ))?;
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::ResultPublication(request.identity()),
        demand: FeaResourceDemand::ResultPublication(ResultPublicationResourceDemand {
            request: request.identity(),
            source_result: request.source_result,
            source_accuracy: accuracy.identity(),
            source_usage: usage.identity(),
            receipt_encoded_bytes_upper_bound: ByteCount(profile.encoded_bytes_upper_bound),
            // The receipt is encoded directly, as every published output.
            receipt_materialization_bytes: ByteCount(
                runmat_canonical_codec::projected_canonical_encoding_resident_bytes_estimate(
                    profile.encoded_bytes_upper_bound,
                    1,
                )
                .ok_or(ResourcePlanningError::Overflow(
                    "result publication receipt",
                ))?,
            ),
            accounting_interval_workspace_bytes: ByteCount(interval_workspace),
            nested_plan_workspace_bytes: ByteCount(nested_plan_workspace),
            observation_task_index_workspace_bytes: ByteCount(observation_index_workspace),
            observation_execution_workspace_bytes: ByteCount(observation_execution_workspace),
            source_validation_workspace_bytes: ByteCount(validation),
            inputs: facts.inputs,
            output: facts.output,
        }),
        output_set: Some(facts.output_set),
        worker: super::post_process::post_process_worker(),
        accuracy_ceiling: super::post_process::post_process_ceiling(),
    })
}
