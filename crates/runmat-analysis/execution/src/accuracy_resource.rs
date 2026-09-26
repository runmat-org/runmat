//! Pre-execution accuracy sizing reused by task preparation and native admission.
use crate::{
    prepared_input_resource_facts_without_resource_plan, projected_output_resource_facts,
    projected_output_set_resource_facts, AnalysisBridgeError, AnalysisBridgeResult,
    PreparedAnalysisObjects,
};
use runmat_analysis_fea::{
    execution::{AccuracyEvaluationRequest, AccuracySourceReceipt},
    planning::resource::{
        build_accuracy_resource_plan, FeaResourcePlan, PreparedPhaseResourceFacts,
    },
    results::accuracy::accuracy_evaluation_resource_profile,
};

pub fn build_prepared_accuracy_resource_plan(
    request: &AccuracyEvaluationRequest,
    receipt: &AccuracySourceReceipt,
    inputs: &[PreparedAnalysisObjects],
) -> AnalysisBridgeResult<FeaResourcePlan> {
    if receipt.source_result() != request.source_result {
        return Err(AnalysisBridgeError::Identity(
            "accuracy resource source differs from request",
        ));
    }
    let evidence = receipt.evidence(request)?;
    let profile = accuracy_evaluation_resource_profile(request, &evidence)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    Ok(build_accuracy_resource_plan(
        request,
        &evidence,
        PreparedPhaseResourceFacts {
            inputs: prepared_input_resource_facts_without_resource_plan(inputs)?,
            output: projected_output_resource_facts(profile.output_encoded_bytes_upper_bound)?,
            output_set: projected_output_set_resource_facts(),
        },
    )?)
}
