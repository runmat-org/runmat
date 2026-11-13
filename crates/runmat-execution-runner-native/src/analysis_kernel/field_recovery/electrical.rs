//! Governed publication of original retained native electrical arrays, not a numerical solve.
use super::*;
use runmat_analysis_execution::{
    build_prepared_electrical_field_recovery_resource_plan, decode_electrical_recovery_source,
};
use runmat_analysis_fea::execution::{ExecutionPlanIdentity, NumericalPlanIdentity};

pub(super) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    use AnalysisNumericalObjectKind::{
        ElectricalStaticWorkload as Workload, FieldRecoveryRequest, Mesh, ResourcePlan,
        SolverResult,
    };
    if invocation.inputs.len() != 5
        || invocation.host.workload.outputs != [AnalysisNumericalObjectKind::Field]
    {
        return Err(invalid("electrical field task inventory differs"));
    }
    let request_objects = unique_input(&invocation, FieldRecoveryRequest)?;
    let request = runmat_analysis_fea::execution::FieldRecoveryRequest::from_canonical_bytes(
        &request_objects.payload(),
    )?;
    let admitted = unique_input(&invocation, ResourcePlan)?;
    let plan = FeaResourcePlan::from_canonical_bytes(&admitted.payload())?;
    if !matches!(
        plan.input.demand,
        runmat_analysis_fea::planning::resource::FeaResourceDemand::ElectricalFieldRecovery(_)
    ) || plan.identity.digest() != admitted.manifest.logical_identity
        || plan
            != build_prepared_electrical_field_recovery_resource_plan(
                &request,
                invocation.inputs,
                invocation.output_policy,
                crate::analysis_kernel::admitted_local_execution(&plan)?,
            )?
    {
        return Err(invalid(
            "electrical recovery admission differs before source decoding",
        ));
    }
    let source = decode_electrical_recovery_source(
        unique_input(&invocation, Workload)?,
        unique_input(&invocation, Mesh)?,
        unique_input(&invocation, SolverResult)?,
    )?;
    let (workload, mesh, result) = (source.workload(), source.mesh(), source.result());
    let numerical = NumericalPlanIdentity::from_electrical_static_workload(
        workload.identity()?,
        runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes()),
    )?;
    if invocation.host.workload.execution_plan
        != ExecutionPlanIdentity::new(numerical, plan.identity.digest())?
        || request.producer != invocation.host.workload.step
        || request != source.native_field_request()?
    {
        return Err(invalid(
            "electrical field task original numerical source differs",
        ));
    }
    let fields = result.recover_native_fields(workload, mesh, &request)?;
    recovered_field_output(&fields)
}
fn invalid(error: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(error.to_string())
}
