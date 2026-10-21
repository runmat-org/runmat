//! Governed copy of an exact random-response result's RMS fields, not a numerical solve.
use super::*;
use runmat_analysis_execution::{
    build_prepared_random_response_field_recovery_resource_plan,
    decode_random_response_recovery_source, random_response_recovery_request,
};
use runmat_analysis_fea::execution::{ExecutionPlanIdentity, NumericalPlanIdentity};

pub(super) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    use AnalysisNumericalObjectKind::{
        FieldRecoveryRequest, Mesh, ResourcePlan, SolverResult,
        StructuralRandomResponseWorkload as Workload,
    };
    if invocation.inputs.len() != 5
        || invocation.host.workload.outputs != [AnalysisNumericalObjectKind::Field]
    {
        return Err(invalid("random-response field task inventory differs"));
    }
    let request_objects = unique_input(&invocation, FieldRecoveryRequest)?;
    let request = runmat_analysis_fea::execution::FieldRecoveryRequest::from_canonical_bytes(
        &request_objects.payload(),
    )?;
    let admitted = unique_input(&invocation, ResourcePlan)?;
    let plan = FeaResourcePlan::from_canonical_bytes(&admitted.payload())?;
    if plan.identity.digest() != admitted.manifest.logical_identity
        || plan
            != build_prepared_random_response_field_recovery_resource_plan(
                &request,
                invocation.inputs,
                invocation.output_policy,
            )?
    {
        return Err(invalid(
            "random-response recovery admission differs before source decoding",
        ));
    }
    let workload = unique_input(&invocation, Workload)?;
    let mesh = unique_input(&invocation, Mesh)?;
    let source = unique_input(&invocation, SolverResult)?;
    let source_identity = request
        .source
        .solver_result()
        .ok_or_else(|| invalid("random-response recovery requires its solver result"))?;
    let numerical = NumericalPlanIdentity::from_statistical_field_recovery(
        source_identity,
        request.topology.discretization_digest(),
    )?;
    if invocation.host.workload.execution_plan
        != ExecutionPlanIdentity::new(numerical, plan.identity.digest())?
        || request.producer != invocation.host.workload.step
        || request
            != random_response_recovery_request(
                request.producer,
                source_identity,
                workload,
                mesh,
                source,
            )?
    {
        return Err(invalid(
            "random-response field task original numerical source differs",
        ));
    }
    let (_, mesh, result) = decode_random_response_recovery_source(workload, mesh, source)?;
    let fields = runmat_analysis_fea::execution::statistical_recovered_fields(
        &request,
        &mesh,
        &runmat_analysis_fea::execution::random_response_statistical_fields(result.numerical()),
    )?;
    recovered_field_output(&fields)
}

fn invalid(error: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(error.to_string())
}
