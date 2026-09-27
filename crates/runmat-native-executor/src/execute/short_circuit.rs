//! Resumable embedded short-circuit evaluation, sharing contextual-region
//! progress with selectors. A completed RHS step is never replayed on resume.
use runmat_mir::{MirOperand, MirPlace, MirShortCircuitOp, MirStmt, MirStmtKind};
use runmat_runtime::native::NativeValueRef;
use runmat_value::Value;

use super::{operand, state::HostState};
use crate::{NativeExecutorError, NativeExecutorResult};

pub(super) fn evaluate(
    state: &mut HostState,
    left: &MirOperand,
    operator: MirShortCircuitOp,
    statements: &[MirStmt],
    right: &MirOperand,
) -> NativeExecutorResult<Vec<NativeValueRef>> {
    let (region, next_step, completed) = state.enter_contextual_region();
    if let Some(result) = completed {
        return Ok(vec![result]);
    }
    // The owning site clears region state and captured sequences on errors or
    // cancellation. Suspension retains both until the exact site is resumed.
    if next_step == 0 {
        state.enter_contextual_step(region, 0);
        let left = operand::materialize_operand(state, left)?;
        let truth = logical_truth(state, &left, "short-circuit left operand")?;
        state.finish_contextual_step(region, 0, 1)?;
        let skipped = match operator {
            MirShortCircuitOp::And => !truth,
            MirShortCircuitOp::Or => truth,
        };
        if skipped {
            let result = state.arena.insert(Value::Bool(truth));
            state.finish_contextual_region(region, result);
            return Ok(vec![result]);
        }
    }
    // Reaching a later checkpoint proves the left branch was taken. Do not
    // reread the left local: an earlier RHS effect may have changed it.
    for (index, statement) in statements
        .iter()
        .enumerate()
        .skip(next_step.saturating_sub(1))
    {
        let step = index + 1;
        state.enter_contextual_step(region, step);
        execute_statement(state, &statement.kind)?;
        state.finish_contextual_step(region, step, step + 1)?;
    }
    let step = statements.len() + 1;
    state.enter_contextual_step(region, step);
    let right = operand::materialize_operand(state, right)?;
    let truth = logical_truth(state, &right, "short-circuit right operand")?;
    state.finish_contextual_step(region, step, step + 1)?;
    let result = state.arena.insert(Value::Bool(truth));
    state.finish_contextual_region(region, result);
    Ok(vec![result])
}

fn execute_statement(state: &mut HostState, statement: &MirStmtKind) -> NativeExecutorResult<()> {
    match statement {
        MirStmtKind::Assign {
            place: MirPlace::Local(local),
            value,
        } => {
            let native_local = u32::try_from(local.0)
                .map(runmat_native_codegen::NativeLocalId)
                .map_err(|_| {
                    NativeExecutorError::Host("embedded local exceeds native schema".into())
                })?;
            let mut values = operand::evaluate_rvalue(state, value, 1, Some(native_local))?;
            if values.len() != 1 {
                return Err(NativeExecutorError::Host(
                    "embedded short-circuit assignment did not produce one value".into(),
                ));
            }
            state.set_local(local.0, values.remove(0))
        }
        MirStmtKind::Expr(value) => {
            let _ = operand::evaluate_rvalue(state, value, 0, None)?;
            Ok(())
        }
        MirStmtKind::CaptureSequence {
            destination,
            source,
        } => super::call::capture_sequence(state, *destination, source),
        other => Err(NativeExecutorError::Host(format!(
            "verified short-circuit payload contains invalid statement {other:?}"
        ))),
    }
}

fn logical_truth(state: &mut HostState, value: &Value, label: &str) -> NativeExecutorResult<bool> {
    super::sync::complete(
        &state.runtime,
        runmat_runtime::condition::logical_truth_from_value(value, label),
        "logical truth evaluation",
    )
    .finish(state)
}
