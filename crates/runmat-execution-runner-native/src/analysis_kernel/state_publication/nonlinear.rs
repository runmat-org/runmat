//! Original terminal MaterialHistory through shared portable publication admission.
use super::*;
use runmat_analysis_execution::{
    build_prepared_nonlinear_state_publication_resource_plan, decode_admitted_resource_plan,
    decode_nonlinear_state_publication, nonlinear_state_host_input,
    require_nonlinear_state_decode_budget,
};

pub(super) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    invocation.host.validate()?;
    if invocation.kind != AnalysisWorkKind::PublishState
        || invocation.host.workload.kind != invocation.kind
        || invocation.host.workload.outputs != [AnalysisNumericalObjectKind::SemanticState]
        || invocation.host.workload.inputs
            != invocation
                .inputs
                .iter()
                .map(nonlinear_state_host_input)
                .collect::<Vec<_>>()
    {
        return Err(AnalysisBridgeError::Identity(
            "nonlinear state immutable host closure differs",
        ));
    }
    let plan = decode_admitted_resource_plan(
        unique_input(&invocation, AnalysisNumericalObjectKind::ResourcePlan)?,
        invocation.output_policy.inventory_limits(),
    )?;
    require_nonlinear_state_decode_budget(invocation.inputs, &plan, invocation.output_policy)?;
    let request = SemanticStatePublicationRequest::from_canonical_bytes(
        &unique_input(
            &invocation,
            AnalysisNumericalObjectKind::StatePublicationRequest,
        )?
        .payload(),
    )?;
    if request.producer != invocation.host.workload.step
        || plan
            != build_prepared_nonlinear_state_publication_resource_plan(
                &request,
                invocation.inputs,
                invocation.output_policy,
            )?
    {
        return Err(AnalysisBridgeError::Identity(
            "nonlinear state actual resource/producer differs",
        ));
    }
    let (workload, block) =
        decode_nonlinear_state_publication(invocation.inputs, &plan, invocation.output_policy)?;
    let state =
        CanonicalSemanticState::from_nonlinear_solid_publication(&request, &workload, &block)?;
    Ok(vec![AnalysisKernelOutput::whole(
        AnalysisNumericalObjectKind::SemanticState,
        state.identity.digest(),
        state.canonical_bytes()?,
    )])
}
