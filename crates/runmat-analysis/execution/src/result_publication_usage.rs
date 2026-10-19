//! Terminal accounting extends, but never rewrites, the baseline named by a receipt.
use runmat_analysis_fea::execution::{AnalysisNumericalObjectKind, AnalysisResultIdentity};
use runmat_analysis_fea::results::publication::validate_result_publication_resource_usage;
use runmat_execution::value::ValueRefKind;

use crate::{
    extend_resource_usage, AnalysisBridgeError, AnalysisBridgeResult, AnalysisOutputPolicy,
    AnalysisPhaseExecutionEvidence, ImportedPublishedResult, PreparedResourceUsage,
    PreparedResultPublication,
};

#[cfg(test)]
mod tests;

pub fn complete_result_publication_resource_usage(
    prepared: &PreparedResultPublication,
    published: &ImportedPublishedResult,
    baseline: &PreparedResourceUsage,
    execution: &AnalysisPhaseExecutionEvidence,
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<PreparedResourceUsage> {
    prepared.request.validate()?;
    let accounting = execution.winning_accounting();
    let graph = execution.resource_graph(&prepared.resource_plan)?;
    published.result.validate()?;
    if !runmat_analysis_fea::planning::resource::allocation_matches_plan(
        &accounting.resources.allocation,
        &prepared.resource_plan.request,
    ) || !prepared
        .host
        .workload
        .generic_capabilities()?
        .is_subset(&accounting.resources.allocation.required_capabilities)
        || published.result.request != prepared.request.identity()
        || published.result.producer != prepared.request.producer
        || published.result.source_result != prepared.request.source_result
        || published.result.source_accuracy != prepared.request.source_accuracy
        || published.result.source_observations != prepared.request.source_observations
        || published.result.source_resource_plan != prepared.request.source_resource_plan
        || published.result.source_resource_usage != prepared.request.source_resource_usage
        || published.result.source_resource_usage != baseline.evidence.identity()
        || published.objects.manifest.kind != AnalysisNumericalObjectKind::PublishedResult
        || published.objects.manifest.logical_identity != published.result.identity().digest()
        || published.objects.payload() != published.result.canonical_bytes()?
    {
        return Err(AnalysisBridgeError::Identity(
            "publication accounting differs from its prepared sources",
        ));
    }
    published
        .objects
        .validate_result_output(policy.chunk_bytes(), policy.inventory_limits())?;
    let output = prepared
        .resource_plan
        .input
        .demand
        .output_facts(AnalysisNumericalObjectKind::PublishedResult)
        .ok_or(AnalysisBridgeError::Identity(
            "publication plan omits its receipt envelope",
        ))?;
    crate::object::validate_prepared_output_resource_facts(&published.objects, output)?;
    let access = &prepared.host.artifact_access;
    let mut references = published
        .objects
        .unique_references(access, ValueRefKind::ResultObject)?;
    let root = published
        .objects
        .root_reference(access, ValueRefKind::ResultObject)?;
    let result = AnalysisResultIdentity::new(
        prepared.host.workload.identity,
        vec![published.objects.manifest.payload_digest],
    )?;
    let output_set = crate::response::prepare_analysis_output_set(
        &prepared.host,
        result,
        std::slice::from_ref(&published.objects),
        &[root],
    )?;
    output_set.validate_resource_facts(prepared.resource_plan.input.output_set.as_ref().ok_or(
        AnalysisBridgeError::Identity("publication plan omits its output-set envelope"),
    )?)?;
    let output_set_root = output_set.root.clone();
    references.push(output_set.root);
    crate::object::retain_unique_references(&mut references);
    let artifact_bytes = references.iter().try_fold(0_u64, |sum, reference| {
        sum.checked_add(reference.encoded_length).ok_or_else(|| {
            AnalysisBridgeError::Invalid("publication artifact bytes overflow".into())
        })
    })?;
    if accounting.resources.artifact_bytes != artifact_bytes {
        return Err(AnalysisBridgeError::Identity(
            "publication accounting differs from its complete output inventory",
        ));
    }
    // Bind the exact canonical publication closure to the original committed
    // task response, even when this API is called without a study coordinator.
    execution.validate_for(
        &prepared.host,
        &prepared.resource_plan,
        &runmat_execution_artifact::ProgramExecutionResponse::ExternalizedSuccess {
            outputs: vec![runmat_execution::value::ValuePayload::Object(Box::new(
                output_set_root,
            ))],
            result_objects: references,
        },
    )?;
    let terminal = extend_resource_usage(baseline, &graph, access, policy)?;
    validate_result_publication_resource_usage(
        &published.result,
        &baseline.evidence,
        &terminal.evidence,
        execution.graph_wall_millis(),
    )?;
    Ok(terminal)
}
