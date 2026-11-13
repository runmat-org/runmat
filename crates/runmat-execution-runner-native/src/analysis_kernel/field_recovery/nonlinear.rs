//! Publish original accepted J2 fields under independently rederived admission.
use super::*;
use runmat_analysis_execution::{
    build_nonlinear_recovery_metadata_resource_plan,
    build_prepared_nonlinear_recovery_resource_plan, decode_nonlinear_recovery_source,
    require_nonlinear_recovery_decode_budget,
};
use runmat_analysis_fea::execution::{ExecutionPlanIdentity, NumericalPlanIdentity};

pub(super) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    use AnalysisNumericalObjectKind::{FieldRecoveryRequest, ResourcePlan};
    let request_objects = unique_input(&invocation, FieldRecoveryRequest)?;
    let request = runmat_analysis_fea::execution::FieldRecoveryRequest::from_canonical_bytes(
        &request_objects.payload(),
    )?;
    let admitted = unique_input(&invocation, ResourcePlan)?;
    let plan = runmat_analysis_execution::decode_admitted_resource_plan(
        admitted,
        invocation.output_policy.inventory_limits(),
    )?;
    require_nonlinear_recovery_decode_budget(
        invocation.inputs,
        &plan,
        invocation.output_policy.inventory_limits(),
    )?;
    let runmat_analysis_fea::planning::resource::FeaResourceDemand::NonlinearFieldRecovery(demand) =
        &plan.input.demand
    else {
        return Err(invalid(
            "nonlinear recovery requires its typed resource demand",
        ));
    };
    if invocation.host.workload.outputs != [AnalysisNumericalObjectKind::Field]
        || plan
            != build_nonlinear_recovery_metadata_resource_plan(
                &request,
                invocation.inputs,
                invocation.output_policy,
                demand.recovery_workspace_bytes,
                crate::analysis_kernel::admitted_local_execution(&plan)?,
            )?
        || plan
            != build_prepared_nonlinear_recovery_resource_plan(
                &request,
                invocation.inputs,
                invocation.output_policy,
                crate::analysis_kernel::admitted_local_execution(&plan)?,
            )?
    {
        return Err(invalid(
            "nonlinear recovery original resource authority differs",
        ));
    }
    let (workload, mesh, block) = decode_nonlinear_recovery_source(invocation.inputs)?;
    if request.producer != invocation.host.workload.step
        || invocation.host.workload.execution_plan
            != ExecutionPlanIdentity::new(
                NumericalPlanIdentity::from_nonlinear_solid_block(&block.request)?,
                plan.identity.digest(),
            )?
    {
        return Err(invalid(
            "nonlinear recovery original numerical lineage differs",
        ));
    }
    let retained = demand
        .inputs
        .decoded_resident_bytes_admission_budget
        .0
        .checked_add(demand.inputs.transport_closure_bytes.0)
        .ok_or_else(|| invalid("nonlinear recovery retained source overflow"))?;
    let fields = block.recover_native_fields(&workload, &mesh, &request, retained)?;
    recovered_field_output(&fields)
}
fn invalid(e: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(e.to_string())
}
