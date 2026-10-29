//! Separate admitted original cell/all-face publication; no fluid solve is called.
use super::*;
use runmat_analysis_execution::{
    build_prepared_fluid_field_recovery_resource_plan, decode_admitted_resource_plan,
    decode_fluid_recovery_source, require_fluid_recovery_source_decode_budget,
    AnalysisWorkloadInput,
};
use runmat_analysis_fea::execution::{
    ExecutionPlanIdentity, FluidSteadyWorkload, NumericalPlanIdentity,
};
use runmat_execution::{value::ValueRefKind, Digest};

pub(super) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    use AnalysisNumericalObjectKind::{
        FieldRecoveryRequest as Request, FluidSteadyWorkload as Workload, Mesh, ResourcePlan,
        SolverResult,
    };
    invocation.host.validate()?;
    if invocation.kind != AnalysisWorkKind::RecoverField
        || invocation.host.workload.kind != invocation.kind
        || invocation.inputs.len() != 5
        || invocation.host.workload.outputs != [AnalysisNumericalObjectKind::Field]
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
                    reference_kind: if input.manifest.kind == SolverResult {
                        ValueRefKind::ResultObject
                    } else {
                        ValueRefKind::DriverObject
                    },
                })
                .collect::<Vec<_>>()
    {
        return Err(invalid("fluid field task immutable closure differs"));
    }
    let plan_objects = unique_input(&invocation, ResourcePlan)?;
    let plan =
        decode_admitted_resource_plan(plan_objects, invocation.output_policy.inventory_limits())?;
    require_fluid_recovery_source_decode_budget(
        invocation.inputs,
        &plan,
        invocation.output_policy.inventory_limits(),
    )?;
    let request_objects = unique_input(&invocation, Request)?;
    let request = FieldRecoveryRequest::from_canonical_bytes(&request_objects.payload())?;
    if request_objects.manifest.logical_identity != request.identity().digest()
        || plan
            != build_prepared_fluid_field_recovery_resource_plan(
                &request,
                invocation.inputs,
                invocation.output_policy,
            )?
    {
        return Err(invalid(
            "fluid field request/resource plan differs from original source",
        ));
    }
    let workload_objects = unique_input(&invocation, Workload)?;
    let mesh_objects = unique_input(&invocation, Mesh)?;
    let result_objects = unique_input(&invocation, SolverResult)?;
    let result = decode_fluid_recovery_source(workload_objects, mesh_objects, result_objects)?;
    let workload = FluidSteadyWorkload::from_canonical_bytes(&workload_objects.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_objects.payload()).map_err(invalid)?;
    let numerical = NumericalPlanIdentity::from_fluid_steady_workload(
        workload.identity()?,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
    )?;
    if invocation.host.workload.execution_plan
        != ExecutionPlanIdentity::new(numerical, plan.identity.digest())?
        || request.producer != invocation.host.workload.step
        || request != result.native_field_request(&workload, &mesh)?
    {
        return Err(invalid(
            "fluid field task differs from original numerical intent/layout",
        ));
    }
    let fields = result.recover_native_fields(&workload, &mesh, &request)?;
    recovered_field_output(&fields)
}

fn invalid(error: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(error.to_string())
}
