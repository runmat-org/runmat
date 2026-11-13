//! Admitted local load blocks. A partial block is not a completed semantic state.
use super::*;
use crate::solve::nonlinear::{
    run_nonlinear_load_block, NonlinearLoadBlockRequest, NonlinearLoadBlockStart,
    NonlinearLoadContinuation,
};
use runmat_execution::Digest;

pub enum NonlinearBlockInput<'a> {
    Fresh,
    Continue {
        checkpoint: &'a NonlinearLoadContinuation,
        expected_digest: Digest,
    },
}

/// Original raw numerical output and controller checkpoint. No completed study,
/// publication receipt, worker admission or restart authority is implied.
pub struct NonlinearBlockOutput {
    pub result: NonlinearSolveResult,
    pub checkpoint: NonlinearLoadContinuation,
    pub complete: bool,
}

/// Rebuild physical inputs from the original model under the same whole-pipeline
/// resource ceiling as fresh execution. Callers retain previous block outputs;
/// their memory must be separately admitted by the coordinating execution layer.
pub fn run_nonlinear_block(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: &NonlinearSolveOptions,
    input: NonlinearBlockInput<'_>,
    accepted_block_limit: usize,
) -> Result<NonlinearBlockOutput, FeaRunError> {
    if accepted_block_limit == 0 || accepted_block_limit > options.max_accepted_increments {
        return Err(FeaRunError::InvalidModel(
            "J2 block limit is outside its admitted accepted inventory".into(),
        ));
    }
    let prepared =
        preparation::prepare_with_output_limit(model, backend, options, accepted_block_limit, 0)?;
    let admission = preflight::tangent_admission(&prepared.resource_plan)?;
    let maximum_resident_bytes = prepared
        .execution_step
        .accuracy
        .resources
        .resident_byte_budget();
    let contact_inputs = match options.solver_mesh.as_ref() {
        Some(mesh) => super::contact::prepare_inputs(
            model,
            mesh,
            &prepared.summary,
            options.contact_enforcement.clone().unwrap_or_default(),
            maximum_resident_bytes,
        )?,
        None => None,
    };
    let contact_operator = contact_inputs
        .as_ref()
        .map(|inputs| super::contact::prepare_operator(model, inputs, prepared.summary.dof_count))
        .transpose()?;
    let contact_execution = contact_operator
        .as_ref()
        .map(|operator| super::contact::execution(operator, options, maximum_resident_bytes));
    let input = match input {
        NonlinearBlockInput::Fresh => NonlinearLoadBlockStart::Fresh,
        NonlinearBlockInput::Continue {
            checkpoint,
            expected_digest,
        } => NonlinearLoadBlockStart::Continue {
            continuation: checkpoint,
            expected_digest,
        },
    };
    let block = run_nonlinear_load_block(NonlinearLoadBlockRequest {
        model,
        summary: &prepared.summary,
        options,
        execution: crate::solve::nonlinear::NonlinearExecution {
            backend,
            contact: contact_execution,
            solver_admission: Some(admission),
        },
        start: prepared.load_factor_start,
        end: prepared.load_factor_end,
        schedule: &prepared.increment_schedule,
        input,
        accepted_block_limit,
    })?;
    Ok(NonlinearBlockOutput {
        result: block.result,
        checkpoint: block.continuation,
        complete: block.complete,
    })
}

#[cfg(test)]
mod tests;
