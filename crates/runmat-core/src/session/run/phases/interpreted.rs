//! Only a cold/unselected native attempt constructs an interpreter child.
use super::*;
pub(super) fn select<'a>(
    active: Active<'a>,
) -> impl std::future::Future<Output = std::result::Result<Active<'a>, RunError>> + 'a {
    if active.state.execution_completed {
        futures::future::Either::Left(std::future::ready(Ok(active)))
    } else {
        futures::future::Either::Right(run(active))
    }
}
async fn run<'a>(mut active: Active<'a>) -> std::result::Result<Active<'a>, RunError> {
    let execution_bytecode = prepare(&mut active);
    let workspace = &mut active.state.data.workspace;
    let execution = active
        .session
        .interpret_with_context(
            &execution_bytecode,
            &mut workspace.variable_array,
            workspace.runtime.clone(),
            workspace.workspace_admission.take(),
        )
        .await;
    complete(&mut active, &execution_bytecode, execution);
    Ok(active)
}

fn prepare(active: &mut Active<'_>) -> runmat_vm::Bytecode {
    let state = &mut active.state;
    let workspace = &mut state.data.workspace;
    let mut bytecode = workspace.bytecode.clone();
    if state.is_expression_stmt
        && matches!(state.final_stmt_emit, FinalStmtEmitDisposition::Inline)
        && !bytecode.instructions.is_empty()
    {
        bytecode.instructions.pop();
        let slot = std::cmp::max(bytecode.var_count, workspace.max_var_id + 1);
        bytecode.instructions.push(runmat_vm::Instr::StoreVar(slot));
        bytecode.var_count = slot + 1;
        if workspace.variable_array.len() <= slot {
            workspace.variable_array.resize(slot + 1, Value::Num(0.0));
        }
    }
    bytecode
}

fn complete(
    active: &mut Active<'_>,
    bytecode: &runmat_vm::Bytecode,
    execution: runmat_vm::WorkspaceInterpreterOutcome,
) {
    let state = &mut active.state;
    state.vm_workspace_completion = execution.workspace;
    let workspace = &state.data.workspace;
    match execution.outcome {
        Ok(runmat_vm::InterpreterOutcome::Completed(completion)) => {
            let results = &completion.values;
            active.session.stats.interpreter_fallback += 1;
            if workspace.stmt_count == 1 && !workspace.display_context.single_stmt_non_assign {
                if let Some(slot) = workspace.display_context.first_assign_var {
                    if let Some(value) = workspace.variable_array.get(slot).cloned() {
                        if state.is_semicolon_suppressed {
                            state.suppressed_value = Some(value);
                        } else {
                            state.result_value = Some(value);
                        }
                    }
                }
            }
            if state.is_expression_stmt
                && matches!(state.final_stmt_emit, FinalStmtEmitDisposition::Inline)
                && !bytecode.instructions.is_empty()
                && state.result_value.is_none()
                && state.suppressed_value.is_none()
            {
                let slot = bytecode.var_count - 1;
                if let Some(value) = workspace.variable_array.get(slot).cloned() {
                    if state.is_semicolon_suppressed {
                        state.suppressed_value = Some(value);
                    } else {
                        state.ans_update = Some((slot, value.clone()));
                        state.result_value = Some(value);
                    }
                }
            } else if !state.is_semicolon_suppressed
                && matches!(
                    state.final_stmt_emit,
                    FinalStmtEmitDisposition::NeedsFallback
                )
                && state.result_value.is_none()
            {
                state.result_value = results
                    .len()
                    .checked_sub(1)
                    .and_then(|slot| results.get(slot))
                    .as_deref()
                    .cloned();
            }
            state.completed_vm_root = Some(runmat_vm::InterpreterOutcome::Completed(completion));
        }
        Err(error) => state.error = Some(error),
    }
    if active.session.verbose {
        debug!(
            "Interpreter completion: result={:?}, suppressed={:?}, error={:?}",
            state.result_value, state.suppressed_value, state.error
        );
    }
}
