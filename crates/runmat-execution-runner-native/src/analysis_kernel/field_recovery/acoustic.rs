//! Governed publication of the accepted acoustic pressure/velocity arrays; no numerical solve.
use super::*;
use runmat_analysis_execution::{
    build_prepared_acoustic_field_recovery_resource_plan, decode_acoustic_recovery_source,
    AnalysisWorkloadInput,
};
use runmat_analysis_fea::execution::{ExecutionPlanIdentity, NumericalPlanIdentity};
use runmat_execution::Digest;

pub(super) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    use AnalysisNumericalObjectKind::{
        AcousticHarmonicWorkload as Workload, FieldRecoveryRequest, Mesh, ResourcePlan,
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
                        runmat_execution::value::ValueRefKind::ResultObject
                    } else {
                        runmat_execution::value::ValueRefKind::DriverObject
                    },
                })
                .collect::<Vec<_>>()
    {
        return Err(invalid("acoustic field task inventory differs"));
    }
    let request_objects = unique_input(&invocation, FieldRecoveryRequest)?;
    let request = runmat_analysis_fea::execution::FieldRecoveryRequest::from_canonical_bytes(
        &request_objects.payload(),
    )?;
    let admitted = unique_input(&invocation, ResourcePlan)?;
    let plan = FeaResourcePlan::from_canonical_bytes(&admitted.payload())?;
    if plan.identity.digest() != admitted.manifest.logical_identity
        || plan
            != build_prepared_acoustic_field_recovery_resource_plan(
                &request,
                invocation.inputs,
                invocation.output_policy,
                crate::analysis_kernel::admitted_local_execution(&plan)?,
            )?
    {
        return Err(invalid(
            "acoustic recovery admission differs before source decoding",
        ));
    }
    let (workload, mesh, result) = decode_acoustic_recovery_source(
        unique_input(&invocation, Workload)?,
        unique_input(&invocation, Mesh)?,
        unique_input(&invocation, SolverResult)?,
    )?;
    let numerical = NumericalPlanIdentity::from_acoustic_workload(
        workload.identity()?,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
    )?;
    if invocation.host.workload.execution_plan
        != ExecutionPlanIdentity::new(numerical, plan.identity.digest())?
        || request.producer != invocation.host.workload.step
        || request != result.native_field_request(&workload, &mesh)?
    {
        return Err(invalid(
            "acoustic field task original numerical source differs",
        ));
    }
    let fields = result.recover_native_fields(&workload, &mesh, &request)?;
    recovered_field_output(&fields)
}

fn invalid(error: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(error.to_string())
}
