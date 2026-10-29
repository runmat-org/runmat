//! Governed copy of an exact statistical result's fields (random-response RMS, response-spectrum
//! combined peak), not a numerical solve.
use super::*;
use runmat_analysis_execution::{
    build_prepared_statistical_field_recovery_resource_plan, decode_statistical_recovery_source,
    statistical_recovery_execution_plan, statistical_recovery_request,
};

pub(super) fn is_statistical(invocation: &AnalysisKernelInvocation<'_>) -> bool {
    invocation.inputs.iter().any(|input| {
        matches!(
            input.manifest.kind,
            AnalysisNumericalObjectKind::StructuralRandomResponseWorkload
                | AnalysisNumericalObjectKind::StructuralModalResponseSpectrumWorkload
        )
    })
}

pub(super) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    use AnalysisNumericalObjectKind::{FieldRecoveryRequest, ResourcePlan};
    if invocation.host.workload.outputs != [AnalysisNumericalObjectKind::Field] {
        return Err(invalid("statistical field task inventory differs"));
    }
    let request_objects = unique_input(&invocation, FieldRecoveryRequest)?;
    let request = runmat_analysis_fea::execution::FieldRecoveryRequest::from_canonical_bytes(
        &request_objects.payload(),
    )?;
    let admitted = unique_input(&invocation, ResourcePlan)?;
    let plan = FeaResourcePlan::from_canonical_bytes(&admitted.payload())?;
    // The rederived plan also checks the exact immutable input inventory.
    if plan.identity.digest() != admitted.manifest.logical_identity
        || plan
            != build_prepared_statistical_field_recovery_resource_plan(
                &request,
                invocation.inputs,
                invocation.output_policy,
                crate::analysis_kernel::admitted_local_execution(&plan)?,
            )?
    {
        return Err(invalid(
            "statistical recovery admission differs before source decoding",
        ));
    }
    let source_identity = request
        .source
        .solver_result()
        .ok_or_else(|| invalid("statistical recovery requires its solver result"))?;
    // Recovery sources are selected from the inputs by kind.
    let sources = invocation.inputs;
    if invocation.host.workload.execution_plan
        != statistical_recovery_execution_plan(&request, plan.identity.digest())?
        || request.producer != invocation.host.workload.step
        || request != statistical_recovery_request(request.producer, source_identity, sources)?
    {
        return Err(invalid(
            "statistical field task original numerical source differs",
        ));
    }
    let fields = decode_statistical_recovery_source(sources)?.recover(&request)?;
    recovered_field_output(&fields)
}

fn invalid(error: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(error.to_string())
}
