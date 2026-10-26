//! Governed original finite-volume CPU solve using shared source admission.
use super::{
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisKernelInvocation, AnalysisKernelOutput,
};
use runmat_analysis_execution::{decode_fluid_transient_task_source, prepare_analysis_objects};
use runmat_analysis_fea::execution::{AnalysisNumericalObjectKind, AnalysisWorkKind};
use runmat_execution_artifact::ObjectNamespace;

#[cfg(test)]
pub(crate) mod tests;

pub(super) fn execute(
    invocation: AnalysisKernelInvocation<'_>,
) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    if invocation.kind != AnalysisWorkKind::SolveFluidTransientBlock {
        return Err(invalid("fluid transient kernel work kind differs"));
    }
    let original = decode_fluid_transient_task_source(
        invocation.host,
        invocation.inputs,
        invocation.output_policy,
    )?;
    let prepared = original.prepare(invocation.output_policy)?;
    let result =
        super::with_admitted_cpu_pool(&original.resource_plan, || Ok(prepared.execute()?))?;
    let result_bytes = result.canonical_bytes()?;
    let facts = original
        .resource_plan
        .input
        .demand
        .output_facts(AnalysisNumericalObjectKind::FluidTransientCheckpoint)
        .ok_or_else(|| invalid("fluid transient checkpoint budget missing"))?;
    if result_bytes.len() as u64 > facts.encoded_payload_bytes_upper_bound.0 {
        return Err(invalid(
            "fluid transient checkpoint exceeds admitted payload",
        ));
    }
    let policy = invocation.output_policy;
    let output = prepare_analysis_objects(
        AnalysisNumericalObjectKind::FluidTransientCheckpoint,
        result.identity()?,
        &result_bytes,
        policy.chunk_bytes(),
        ObjectNamespace::ResultValue,
        policy.inventory_limits(),
    )?;
    Ok(vec![AnalysisKernelOutput::prepared(output)])
}
fn invalid(e: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(e.to_string())
}
