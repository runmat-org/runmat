//! Original workspace/result publication after the selected child has retired.
use super::*;
fn completed_value(
    completion: &Option<runmat_vm::InterpreterOutcome>,
    slots: &HashMap<String, usize>,
    #[cfg(not(target_arch = "wasm32"))] native: &Option<
        Box<crate::generic_native::RootedNativeExecution>,
    >,
    slot: usize,
) -> Option<Value> {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(native) = native {
        let snapshot = native.workspace.as_ref()?;
        return slots
            .iter()
            .find_map(|(name, index)| (*index == slot).then(|| snapshot.get(name)).flatten())
            .cloned();
    }
    let _ = slots;
    match completion {
        Some(runmat_vm::InterpreterOutcome::Completed(completion)) => {
            completion.values.get(slot).map(|value| value.clone())
        }
        None => None,
    }
}

pub(super) fn run(active: Active<'_>) -> std::result::Result<SessionExecution, RunError> {
    let Active { session, state } = active;
    let _guards = state.data.guards;
    let State {
        // Declared before the remaining owners so they retire before this
        // guard, which itself still precedes the original `_guards` teardown.
        mut workspace_publication,
        used_jit,
        execution_completed: _,
        mut result_value,
        completed_vm_root,
        suppressed_value,
        mut error,
        mut workspace_updates,
        mut workspace_snapshot_force_full,
        vm_workspace_completion,
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
                workspace,
                start_time,
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
        execution_vars,
        workspace_projection,
        fusion_snapshot,
        variable_array: _unstarted_frame,
        workspace_admission: _workspace_admission,
        ..
    } = workspace;

    let last_assign_var = display_context.last_assign_var;
    let last_expr_emits = display_context.last_expr_emits;
    // Only a completed entry can supply its final public result. Earlier
    // statement output has already been recorded in the console stream;
    // an errored final assignment must not read its initialized VM slot.
    if error.is_none() && !is_semicolon_suppressed && result_value.is_none() {
        let can_emit_from_context = !display_var_ids.is_empty() || last_expr_emits;
        if can_emit_from_context {
            if let Some(value) = runmat_runtime::console::take_last_value_output() {
                result_value = Some(value);
            }
            if result_value.is_none() {
                if let Some(var_id) = last_store_var_index(&bytecode) {
                    result_value = completed_value(
                        &completed_vm_root,
                        &execution_vars,
                        #[cfg(not(target_arch = "wasm32"))]
                        &completed_native_root,
                        var_id,
                    );
                }
                if result_value.is_none() {
                    if let Some(var_id) = last_assign_var {
                        result_value = completed_value(
                            &completed_vm_root,
                            &execution_vars,
                            #[cfg(not(target_arch = "wasm32"))]
                            &completed_native_root,
                            var_id,
                        );
                    }
                }
                if result_value.is_none() {
                    if let Some(var_id) = last_emit_var_index(&bytecode) {
                        result_value = completed_value(
                            &completed_vm_root,
                            &execution_vars,
                            #[cfg(not(target_arch = "wasm32"))]
                            &completed_native_root,
                            var_id,
                        );
                    }
                }
            }
        }
    }

    // Executor-visible workspace mutations are admitted in source order.
    // Publish them even when a later statement fails so an ordinary script
    // preserves MATLAB workspace semantics without replacing concurrent
    // canonical state from an end-of-run snapshot.
    workspace_publication
        .publish_pending()
        .map_err(RunError::Runtime)?;
    let published_changes = workspace_projection
        .take_published_changes_with_context(&execution_runtime)
        .map_err(RunError::Runtime)?;
    for change in published_changes.changes() {
        match change.kind() {
            runmat_runtime::workspace::WorkspaceChangeKind::Assign => {
                if let (Some(name), Some(value)) = (change.name(), change.value()) {
                    workspace_updates.push(workspace_entry(name, value));
                }
            }
            runmat_runtime::workspace::WorkspaceChangeKind::Remove
            | runmat_runtime::workspace::WorkspaceChangeKind::Clear => {
                workspace_snapshot_force_full = true;
            }
        }
    }

    let execution_time = start_time.elapsed();
    #[cfg(not(target_arch = "wasm32"))]
    if let Some((_, profile)) = tiered_interactive.as_ref() {
        session.observe_tiered_interactive(
            &interactive_unit,
            profile.as_ref(),
            execution_time,
            &native_loop_backedges,
        );
    }
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(selected) = ready_route_selection {
        selected.observe(if execution_runtime.is_cancelled() {
            runmat_execution::WholeWorkObservationOutcome::Cancelled
        } else if error.is_none() {
            runmat_execution::WholeWorkObservationOutcome::Success
        } else {
            runmat_execution::WholeWorkObservationOutcome::OperationalFailure
        });
    }
    let execution_time_ms = execution_time.as_millis() as u64;

    session.stats.total_execution_time_ms += execution_time_ms;
    session.stats.average_execution_time_ms =
        session.stats.total_execution_time_ms as f64 / session.stats.total_executions as f64;

    let vm_workspace_state = (!used_jit)
        .then(|| vm_workspace_completion.map(|completion| completion.state))
        .flatten();
    // Implicit ans is an ordinary compiler-authored MIR binding assignment.
    // Its exact frame owner has already published the same effect as any
    // explicit assignment, including a completed prefix before a later error.

    // Refresh only compiler metadata; the exact completed owner retains
    // values. The register file/native snapshot is never a second workspace.
    if let Some(workspace_state) = vm_workspace_state.as_ref() {
        let bindings = workspace_state.assigned.iter().filter_map(|name| {
            workspace_state
                .names
                .get(name)
                .map(|slot| (name.as_str(), *slot))
        });
        session
            .refresh_workspace_binding_slots(bindings, |_, slot| {
                Ok(workspace_state.global_aliases.binding_handle_for_slot(slot))
            })
            .map_err(RunError::Runtime)?;
    }
    #[cfg(not(target_arch = "wasm32"))]
    if used_jit && error.is_none() {
        if let Some(snapshot) = completed_native_root
            .as_ref()
            .and_then(|native| native.workspace.as_ref())
        {
            let bindings = execution_vars.iter().filter_map(|(name, slot)| {
                snapshot
                    .contains_key(name)
                    .then_some((name.as_str(), *slot))
            });
            let base_workspace = session.base_workspace.clone();
            session
                .refresh_workspace_binding_slots(bindings, |name, _| {
                    base_workspace
                        .global_target(name)
                        .map_err(runmat_runtime::workspace::base_workspace_error_to_runtime)
                })
                .map_err(RunError::Runtime)?;
        }
    }
    // Ingress reconciles external changes before execution. Every effect
    // in this invocation, including global declarations and a completed
    // error prefix, has already published its prepared binding patch.
    // Rebuilding that table here would require fresh admission after the
    // effect and could replace the original cancellation/error.

    // Function-definition publication remains transactional with a
    // successful language execution; value assignments above deliberately
    // survive a later runtime error.
    if error.is_none() {
        #[cfg(not(target_arch = "wasm32"))]
        session
            .generic_native_cache
            .publish_session_registry(&session.function_registry, &function_registry_after_success)
            .map_err(RunError::Runtime)?;
        session
            .program_service
            .publish(&interactive_unit, &function_registry_after_success);
        session.function_registry = function_registry_after_success;
        session.next_semantic_function_id = next_semantic_function_id_after_success;
        source_registry_publication.finish(session.function_registry.clone());
    } else {
        drop(source_registry_publication);
    }

    if session.verbose {
        debug!("Execution completed in {execution_time_ms}ms (JIT: {used_jit})");
    }

    if error.is_none()
        && !is_expression_stmt
        && !is_semicolon_suppressed
        && last_assign_var.is_some()
        && !display_context.single_stmt_non_assign
        && !display_var_ids.is_empty()
    {
        if let Some(var_id) = last_store_var_index(&bytecode) {
            result_value = completed_value(
                &completed_vm_root,
                &execution_vars,
                #[cfg(not(target_arch = "wasm32"))]
                &completed_native_root,
                var_id,
            );
        }
    }

    // Generate type info if we have a suppressed value
    let type_info = suppressed_value.as_ref().map(format_type_info);

    let streams = runmat_runtime::console::take_thread_buffer();
    let canonical_workspace = session
        .logical_workspace_values()
        .map_err(RunError::Runtime)?;
    let (workspace_entries, snapshot_full) = if workspace_snapshot_force_full {
        let mut entries = canonical_workspace
            .iter()
            .map(|(name, value)| workspace_entry(name, value))
            .collect::<Vec<_>>();
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        (entries, true)
    } else if workspace_updates.is_empty() {
        if canonical_workspace.is_empty() {
            (workspace_updates, false)
        } else {
            let mut entries = canonical_workspace
                .iter()
                .map(|(name, value)| workspace_entry(name, value))
                .collect::<Vec<_>>();
            entries.sort_by(|a, b| a.name.cmp(&b.name));
            (entries, true)
        }
    } else {
        (workspace_updates, false)
    };
    let workspace_snapshot = session
        .build_workspace_snapshot(workspace_entries, snapshot_full)
        .map_err(RunError::Runtime)?;
    let figures_touched = runmat_runtime::plotting_hooks::take_recent_figures();
    let stdin_events = stdin_events
        .lock()
        .map(|guard| guard.clone())
        .unwrap_or_default();

    let warnings = runmat_runtime::warning_store::take_all();
    let diary_error = runmat_runtime::console::take_diary_error();
    if error.is_none() {
        if let Some(diary_error) = diary_error {
            error = Some(
                build_runtime_error(format!("diary: failed to write diary file ({diary_error})"))
                    .with_identifier("RunMat:diary:IO")
                    .with_source(diary_error)
                    .build(),
            );
        }
    }

    if let Some(runtime_error) = &mut error {
        session.normalize_error_namespace(runtime_error);
        session.populate_callstack(runtime_error, &execution_runtime);
    }

    let suppress_public_value =
        is_expression_stmt && matches!(final_stmt_emit, FinalStmtEmitDisposition::Suppressed);
    let public_value = if is_semicolon_suppressed || suppress_public_value {
        None
    } else {
        result_value
    };

    let mut diagnostics = Vec::new();
    if let Some(error) = &error {
        diagnostics.push(crate::abi::RuntimeDiagnostic {
            code: error
                .identifier()
                .unwrap_or("RunMat:RuntimeError")
                .to_string(),
            severity: crate::abi::DiagnosticSeverity::Error,
            message: error.message().to_string(),
            span: runtime_error_span(error),
            callstack: if !error.context.call_stack.is_empty() {
                error.context.call_stack.clone()
            } else {
                error
                    .context
                    .call_frames
                    .iter()
                    .map(|frame| frame.function.clone())
                    .collect()
            },
            callstack_elided: error.context.call_frames_elided,
        });
    }
    diagnostics.extend(
        warnings
            .iter()
            .map(|warning| crate::abi::RuntimeDiagnostic {
                code: warning.identifier.clone(),
                severity: crate::abi::DiagnosticSeverity::Warning,
                message: warning.message.clone(),
                span: None,
                callstack: Vec::new(),
                callstack_elided: 0,
            }),
    );

    let display_events = public_value
        .as_ref()
        .map(|value| crate::abi::DisplayEvent {
            label: crate::abi::DisplayLabel::Anonymous,
            value: value.clone(),
            span: runmat_hir::Span::default(),
        })
        .into_iter()
        .collect();

    let outcome = crate::abi::ExecutionOutcome {
        flow: public_value
            .clone()
            .map(crate::abi::RuntimeFlow::Single)
            .unwrap_or(crate::abi::RuntimeFlow::NoValue),
        workspace_delta: crate::abi::WorkspaceDelta {
            version: workspace_snapshot.version,
            full_snapshot_required: workspace_snapshot.full,
            ..crate::abi::WorkspaceDelta::default()
        },
        display_events,
        streams,
        diagnostics,
        effects: Vec::new(),
        suspension: None,
        execution_time_ms,
        used_jit,
        type_info,
        figures_touched,
        stdin_events,
        fusion_plan: fusion_snapshot,
        profiling: None,
    };

    let outcome = runmat_runtime::execution::RootedResult::new(outcome)
        .map_err(|error| RunError::Runtime(error.into_runtime_error()))?;
    #[cfg(not(target_arch = "wasm32"))]
    drop(completed_native_root);
    drop(completed_vm_root);
    Ok(SessionExecution {
        outcome,
        workspace_snapshot,
    })
}
