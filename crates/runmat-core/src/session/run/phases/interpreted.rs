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
async fn run<'a>(active: Active<'a>) -> std::result::Result<Active<'a>, RunError> {
    let Active { session, state } = active;
    let _guards = state.data.guards;
    let State {
        used_jit,
        execution_completed,
        mut result_value,
        mut completed_vm_root,
        mut suppressed_value,
        mut error,
        workspace_updates,
        workspace_snapshot_force_full,
        vm_workspace_completion: _,
        mut ans_update,
        is_expression_stmt,
        is_semicolon_suppressed,
        final_stmt_emit,
        #[cfg(not(target_arch = "wasm32"))]
        completed_native_root,
        #[cfg(not(target_arch = "wasm32"))]
        ready_route_selection,
        #[cfg(not(target_arch = "wasm32"))]
        tiered_interactive,
        #[cfg(not(target_arch = "wasm32"))]
        native_loop_backedges,
        data:
            command::Data {
                input,
                workspace,
                start_time,
                debug_trace,
                stdin_events,
                ..
            },
    } = state;
    let preparation::workspace::PreparedInteractiveWorkspace {
        unit: interactive_unit,
        runtime: execution_runtime,
        bytecode,
        publication: source_registry_publication,
        durable_registry: function_registry_after_success,
        next_semantic_function_id: next_semantic_function_id_after_success,
        display_context,
        display_var_ids,
        stmt_count,
        execution_vars,
        max_var_id,
        workspace_projection,
        workspace_admission,
        fusion_snapshot,
        variable_array,
    } = workspace;
    let mut workspace_admission = workspace_admission;
    let mut variable_array = variable_array;

    if session.verbose {
        debug!(
            "Interpreter path: variable_array size: {}, bytecode.var_count: {}",
            variable_array.len(),
            bytecode.var_count
        );
    }

    // For expressions, modify bytecode to store result in a temp variable instead of using stack
    let mut execution_bytecode = bytecode.clone();
    if is_expression_stmt
        && matches!(final_stmt_emit, FinalStmtEmitDisposition::Inline)
        && !execution_bytecode.instructions.is_empty()
    {
        execution_bytecode.instructions.pop(); // Remove the Pop instruction

        // Add StoreVar instruction to store the result in a temporary variable
        let temp_var_id = std::cmp::max(execution_bytecode.var_count, max_var_id + 1);
        execution_bytecode
            .instructions
            .push(runmat_vm::Instr::StoreVar(temp_var_id));
        execution_bytecode.var_count = temp_var_id + 1; // Expand variable count for temp variable

        // Ensure our variable array can hold the temporary variable
        if variable_array.len() <= temp_var_id {
            variable_array.resize(temp_var_id + 1, Value::Num(0.0));
        }

        if session.verbose {
            debug!(
                "Modified expression bytecode, new instructions: {:?}",
                execution_bytecode.instructions
            );
        }
    }

    let execution = session
        .interpret_with_context(
            &execution_bytecode,
            &mut variable_array,
            execution_runtime.clone(),
            workspace_admission.take(),
        )
        .await;
    let vm_workspace_completion = execution.workspace;
    match execution.outcome {
        Ok(runmat_vm::InterpreterOutcome::Completed(completion)) => {
            let results = &completion.values;
            session.stats.interpreter_fallback += 1;
            if session.verbose {
                debug!("Interpreter results: {results:?}");
            }

            // Handle assignment statements (x = 42 should show the assigned value unless suppressed)
            if stmt_count == 1 {
                if !display_context.single_stmt_non_assign {
                    if let Some(var_id) = display_context.first_assign_var {
                        // For assignments, capture the assigned value for both display and type info
                        if var_id < variable_array.len() {
                            let assignment_value = variable_array[var_id].clone();
                            if !is_semicolon_suppressed {
                                result_value = Some(assignment_value);
                                if session.verbose {
                                    debug!("Interpreter assignment result: {result_value:?}");
                                }
                            } else {
                                suppressed_value = Some(assignment_value);
                                if session.verbose {
                                    debug!(
                                                "Interpreter assignment suppressed due to semicolon, captured for type info"
                                            );
                                }
                            }
                        }
                    }
                } else if !is_expression_stmt
                    && !results.is_empty()
                    && !is_semicolon_suppressed
                    && !display_context.single_stmt_non_assign
                    && matches!(final_stmt_emit, FinalStmtEmitDisposition::NeedsFallback)
                {
                    result_value = results.get(0).as_deref().cloned();
                }
            }

            // For expressions, get the result from the temporary variable (capture for both display and type info)
            if is_expression_stmt
                && matches!(final_stmt_emit, FinalStmtEmitDisposition::Inline)
                && !execution_bytecode.instructions.is_empty()
                && result_value.is_none()
                && suppressed_value.is_none()
            {
                let temp_var_id = execution_bytecode.var_count - 1; // The temp variable we added
                if temp_var_id < variable_array.len() {
                    let expression_value = variable_array[temp_var_id].clone();
                    if !is_semicolon_suppressed {
                        // Capture for 'ans' update when output is not suppressed
                        ans_update = Some((temp_var_id, expression_value.clone()));
                        result_value = Some(expression_value);
                        if session.verbose {
                            debug!(
                                "Expression result from temp var {temp_var_id}: {result_value:?}"
                            );
                        }
                    } else {
                        suppressed_value = Some(expression_value);
                        if session.verbose {
                            debug!(
                                        "Expression suppressed, captured for type info from temp var {temp_var_id}: {suppressed_value:?}"
                                    );
                        }
                    }
                }
            } else if !is_semicolon_suppressed
                && matches!(final_stmt_emit, FinalStmtEmitDisposition::NeedsFallback)
                && result_value.is_none()
            {
                result_value = results
                    .len()
                    .checked_sub(1)
                    .and_then(|slot| results.get(slot))
                    .as_deref()
                    .cloned();
                if session.verbose {
                    debug!("Fallback result from interpreter: {result_value:?}");
                }
            }

            if session.verbose {
                debug!("Final result_value: {result_value:?}");
            }
            debug!("Interpreter execution successful");
            completed_vm_root = Some(runmat_vm::InterpreterOutcome::Completed(completion));
        }

        Err(e) => {
            debug!("Interpreter execution failed: {e}");
            error = Some(e);
        }
    }
    Ok(Active {
        session,
        state: State {
            used_jit,
            execution_completed,
            result_value,
            completed_vm_root,
            suppressed_value,
            error,
            workspace_updates,
            workspace_snapshot_force_full,
            vm_workspace_completion,
            ans_update,
            is_expression_stmt,
            is_semicolon_suppressed,
            final_stmt_emit,
            #[cfg(not(target_arch = "wasm32"))]
            completed_native_root,
            #[cfg(not(target_arch = "wasm32"))]
            ready_route_selection,
            #[cfg(not(target_arch = "wasm32"))]
            tiered_interactive,
            #[cfg(not(target_arch = "wasm32"))]
            native_loop_backedges,
            data: command::Data {
                input,
                workspace: preparation::workspace::PreparedInteractiveWorkspace {
                    unit: interactive_unit,
                    runtime: execution_runtime,
                    bytecode,
                    publication: source_registry_publication,
                    durable_registry: function_registry_after_success,
                    next_semantic_function_id: next_semantic_function_id_after_success,
                    display_context,
                    display_var_ids,
                    stmt_count,
                    execution_vars,
                    max_var_id,
                    workspace_projection,
                    workspace_admission,
                    fusion_snapshot,
                    variable_array,
                },
                start_time,
                debug_trace,
                stdin_events,
                guards: _guards,
            },
        },
    })
}
