//! Governed publication of original retained native arrays, not a numerical solve.
use super::*;
use runmat_analysis_execution::{
    build_magnetic_recovery_metadata_resource_plan,
    build_prepared_magnetic_field_recovery_resource_plan, decode_magnetic_recovery_source,
};
use runmat_analysis_fea::execution::{
    ExecutionPlanIdentity, MagnetostaticWorkload, NumericalPlanIdentity,
};

pub(super) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    use AnalysisNumericalObjectKind::{
        FieldRecoveryRequest, MagnetostaticWorkload as Workload, Mesh, ResourcePlan, SolverResult,
    };
    if invocation.inputs.len() != 5
        || invocation.host.workload.outputs != [AnalysisNumericalObjectKind::Field]
    {
        return Err(invalid("magnetic field task inventory differs"));
    }
    let request_objects = unique_input(&invocation, FieldRecoveryRequest)?;
    let request = runmat_analysis_fea::execution::FieldRecoveryRequest::from_canonical_bytes(
        &request_objects.payload(),
    )?;
    let admitted = unique_input(&invocation, ResourcePlan)?;
    let plan = FeaResourcePlan::from_canonical_bytes(&admitted.payload())?;
    let runmat_analysis_fea::planning::resource::FeaResourceDemand::MagneticFieldRecovery(demand) =
        &plan.input.demand
    else {
        return Err(invalid("magnetic recovery requires typed admission demand"));
    };
    if plan.identity.digest() != admitted.manifest.logical_identity
        || plan
            != build_magnetic_recovery_metadata_resource_plan(
                &request,
                invocation.inputs,
                invocation.output_policy,
                demand.source_validation_workspace_bytes,
                crate::analysis_kernel::admitted_local_execution(&plan)?,
            )?
    {
        return Err(invalid(
            "magnetic recovery admission differs before source decoding",
        ));
    }
    let workload = unique_input(&invocation, Workload)?;
    let mesh = unique_input(&invocation, Mesh)?;
    let source = unique_input(&invocation, SolverResult)?;
    if plan
        != build_prepared_magnetic_field_recovery_resource_plan(
            &request,
            invocation.inputs,
            invocation.output_policy,
            crate::analysis_kernel::admitted_local_execution(&plan)?,
        )?
    {
        return Err(invalid(
            "magnetic recovery source validation reservation differs before derived maps",
        ));
    }
    let result = decode_magnetic_recovery_source(workload, mesh, source)?;
    let workload = MagnetostaticWorkload::from_canonical_bytes(&workload.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh.payload()).map_err(invalid)?;
    let numerical = NumericalPlanIdentity::from_magnetostatic_workload(
        workload.identity()?,
        result.mesh_digest(),
    )?;
    if invocation.host.workload.execution_plan
        != ExecutionPlanIdentity::new(numerical, plan.identity.digest())?
        || request.producer != invocation.host.workload.step
        || request != result.native_field_request(&workload, &mesh)?
    {
        return Err(invalid(
            "magnetic field task original numerical source differs",
        ));
    }
    let fields = result.recover_native_fields(&workload, &mesh, &request)?;
    recovered_field_output(&fields)
}
fn invalid(error: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(error.to_string())
}
