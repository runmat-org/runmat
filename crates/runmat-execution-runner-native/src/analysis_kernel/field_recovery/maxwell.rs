//! Governed publication of original complex Maxwell arrays; no numerical solve.
use super::*;
use runmat_analysis_execution::{
    build_maxwell_recovery_metadata_resource_plan,
    build_prepared_maxwell_field_recovery_resource_plan, decode_maxwell_recovery_source,
    AnalysisWorkloadInput,
};
use runmat_analysis_fea::execution::{
    ExecutionPlanIdentity, MaxwellHarmonicWorkload, NumericalPlanIdentity,
};
use runmat_execution::Digest;

#[cfg(test)]
mod tests;

pub(super) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    use AnalysisNumericalObjectKind::{
        FieldRecoveryRequest, MaxwellHarmonicWorkload as Workload, Mesh, ResourcePlan, SolverResult,
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
                        runmat_execution::value::ValueRefKind::ResultObject
                    } else {
                        runmat_execution::value::ValueRefKind::DriverObject
                    },
                })
                .collect::<Vec<_>>()
    {
        return Err(invalid("Maxwell field task inventory differs"));
    }
    let request_objects = unique_input(&invocation, FieldRecoveryRequest)?;
    let request = runmat_analysis_fea::execution::FieldRecoveryRequest::from_canonical_bytes(
        &request_objects.payload(),
    )?;
    let admitted = unique_input(&invocation, ResourcePlan)?;
    let plan = FeaResourcePlan::from_canonical_bytes(&admitted.payload())?;
    let runmat_analysis_fea::planning::resource::FeaResourceDemand::MaxwellFieldRecovery(demand) =
        &plan.input.demand
    else {
        return Err(invalid("Maxwell recovery requires typed admission demand"));
    };
    // Count/profile-only checks admit the exact resource envelope before result decode
    // or retained-source recompilation. They do not execute Maxwell assembly/GMRES.
    if plan.identity.digest() != admitted.manifest.logical_identity
        || plan
            != build_maxwell_recovery_metadata_resource_plan(
                &request,
                invocation.inputs,
                invocation.output_policy,
                demand.source_validation_workspace_bytes,
                crate::analysis_kernel::admitted_local_execution(&plan)?,
            )?
    {
        return Err(invalid(
            "Maxwell recovery admission differs before source decoding",
        ));
    }
    if plan
        != build_prepared_maxwell_field_recovery_resource_plan(
            &request,
            invocation.inputs,
            invocation.output_policy,
            crate::analysis_kernel::admitted_local_execution(&plan)?,
        )?
    {
        return Err(invalid(
            "Maxwell recovery source validation reservation differs before derived maps",
        ));
    }
    let workload_objects = unique_input(&invocation, Workload)?;
    let mesh_objects = unique_input(&invocation, Mesh)?;
    let source = unique_input(&invocation, SolverResult)?;
    let result = decode_maxwell_recovery_source(workload_objects, mesh_objects, source)?;
    let workload = MaxwellHarmonicWorkload::from_canonical_bytes(&workload_objects.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_objects.payload()).map_err(invalid)?;
    let numerical = NumericalPlanIdentity::from_maxwell_harmonic_workload(
        workload.identity()?,
        Digest::from_bytes(*result.response().mesh_digest.bytes()),
    )?;
    if invocation.host.workload.execution_plan
        != ExecutionPlanIdentity::new(numerical, plan.identity.digest())?
        || request.producer != invocation.host.workload.step
        || request != result.native_field_request(&workload, &mesh)?
    {
        return Err(invalid(
            "Maxwell field task original numerical source differs",
        ));
    }
    let fields = result.recover_native_fields(&workload, &mesh, &request)?;
    recovered_field_output(&fields)
}

fn invalid(error: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(error.to_string())
}
