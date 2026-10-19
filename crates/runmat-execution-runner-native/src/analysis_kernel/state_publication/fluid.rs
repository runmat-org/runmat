//! Separate original finite-volume state publication; no numerical solve is invoked.
use super::*;
use runmat_analysis_execution::{
    build_prepared_fluid_state_publication_resource_plan, decode_admitted_resource_plan,
    require_fluid_state_source_decode_budget, AnalysisWorkloadInput,
};
use runmat_analysis_fea::execution::{FluidSteadySolverResult, FluidSteadyWorkload};
use runmat_execution::value::ValueRefKind;

pub(super) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    invocation.host.validate()?;
    if invocation.kind != AnalysisWorkKind::PublishState
        || invocation.host.workload.kind != invocation.kind
        || invocation.inputs.len() != 6
        || invocation.host.workload.outputs != [AnalysisNumericalObjectKind::SemanticState]
        || invocation.host.workload.inputs
            != invocation
                .inputs
                .iter()
                .map(|input| AnalysisWorkloadInput {
                    kind: input.manifest.kind,
                    logical_identity: input.manifest.logical_identity,
                    payload_digest: input.manifest.payload_digest,
                    manifest_digest: input.root.digest,
                    manifest_encoded_length: input.root.encoded_length,
                    reference_kind: if input.manifest.kind
                        == AnalysisNumericalObjectKind::SolverResult
                    {
                        ValueRefKind::ResultObject
                    } else {
                        ValueRefKind::DriverObject
                    },
                })
                .collect::<Vec<_>>()
    {
        return Err(AnalysisBridgeError::Identity(
            "fluid state immutable task closure differs",
        ));
    }
    let mut plans = invocation
        .inputs
        .iter()
        .filter(|input| input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .map(|input| {
            decode_admitted_resource_plan(input, invocation.output_policy.inventory_limits())
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    if plans.len() != 2 {
        return Err(AnalysisBridgeError::Identity(
            "fluid state requires original solve and publication plans",
        ));
    }
    let position = plans
        .iter()
        .position(|plan| {
            matches!(
                plan.input.demand,
                FeaResourceDemand::SemanticStatePublication(_)
            )
        })
        .ok_or(AnalysisBridgeError::Identity(
            "fluid state publication resource plan absent",
        ))?;
    let plan = plans.swap_remove(position);
    require_fluid_state_source_decode_budget(
        invocation.inputs,
        &plan,
        invocation.output_policy.inventory_limits(),
    )?;
    let request_objects = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::StatePublicationRequest,
    )?;
    let request =
        SemanticStatePublicationRequest::from_canonical_bytes(&request_objects.payload())?;
    if request.producer != invocation.host.workload.step
        || plan
            != build_prepared_fluid_state_publication_resource_plan(
                &request,
                invocation.inputs,
                invocation.output_policy,
            )?
    {
        return Err(AnalysisBridgeError::Identity(
            "fluid state resource or producer authority differs",
        ));
    }
    let workload_objects = unique_input(
        &invocation,
        AnalysisNumericalObjectKind::FluidSteadyWorkload,
    )?;
    let mesh_objects = unique_input(&invocation, AnalysisNumericalObjectKind::Mesh)?;
    let result_objects = unique_input(&invocation, AnalysisNumericalObjectKind::SolverResult)?;
    let workload = FluidSteadyWorkload::from_canonical_bytes(&workload_objects.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_objects.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let result = FluidSteadySolverResult::from_canonical_bytes_for(
        &result_objects.payload(),
        result_objects.manifest.payload_digest,
        &workload,
        &mesh,
    )?;
    let state =
        CanonicalSemanticState::from_fluid_publication(&request, &workload, &mesh, &result)?;
    Ok(vec![AnalysisKernelOutput::whole(
        AnalysisNumericalObjectKind::SemanticState,
        state.identity.digest(),
        state.canonical_bytes()?,
    )])
}
