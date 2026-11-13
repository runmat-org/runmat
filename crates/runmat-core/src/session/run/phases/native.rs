//! Original setup and exactly one actual tiered/native attempt.
use super::*;
pub(super) async fn run<'a>(
    session: &'a mut RunMatSession,
    data: command::Data<'a>,
) -> std::result::Result<Active<'a>, RunError> {
    // Declared first: these original guards retire only after every
    // execution/publication owner below, including on errors or abandonment.
    let _guards = data.guards;
    let command::Data {
        input,
        workspace,
        start_time,
        debug_trace,
        stdin_events,
        ..
    } = data;
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
        frame_extent,
        workspace_projection,
        workspace_admission,
        fusion_snapshot,
        variable_array,
    } = workspace;
    #[cfg(not(target_arch = "wasm32"))]
    let mut execution_vars = execution_vars;
    #[cfg(not(target_arch = "wasm32"))]
    let mut frame_extent = frame_extent;
    #[cfg(not(target_arch = "wasm32"))]
    let mut used_jit = false;
    #[cfg(target_arch = "wasm32")]
    let used_jit = false;
    #[cfg(not(target_arch = "wasm32"))]
    let mut execution_completed = false;
    #[cfg(target_arch = "wasm32")]
    let execution_completed = false;
    #[cfg(not(target_arch = "wasm32"))]
    let mut result_value: Option<Value> = None; // Always start fresh for each execution
    #[cfg(target_arch = "wasm32")]
    let result_value: Option<Value> = None; // Always start fresh for each execution
    let completed_vm_root = None;
    #[cfg(not(target_arch = "wasm32"))]
    let mut completed_native_root = None;
    #[cfg(not(target_arch = "wasm32"))]
    let mut ready_route_selection = None;
    #[cfg(not(target_arch = "wasm32"))]
    let mut suppressed_value: Option<Value> = None; // Track value for type info when suppressed
    #[cfg(target_arch = "wasm32")]
    let suppressed_value: Option<Value> = None; // Track value for type info when suppressed
    #[cfg(not(target_arch = "wasm32"))]
    let mut error = None;
    #[cfg(target_arch = "wasm32")]
    let error = None;
    let workspace_updates: Vec<WorkspaceEntry> = Vec::new();
    let workspace_snapshot_force_full = false;
    let vm_workspace_completion = None;
    #[cfg(not(target_arch = "wasm32"))]
    let tiered_interactive = session
        .prepare_tiered_interactive(&interactive_unit, &workspace_projection, &execution_runtime)
        .map_err(RunError::Runtime)?;
    #[cfg(not(target_arch = "wasm32"))]
    let mut native_loop_backedges = std::collections::BTreeMap::new();

    let observed_base = std::rc::Rc::new(
        runmat_runtime::workspace::BaseWorkspaceService::new(session.base_workspace.clone())
            .with_invocation_projection(workspace_projection.clone())
            .map_err(RunError::Runtime)?,
    );
    let execution_runtime = execution_runtime.clone().with_service_ports(
        execution_runtime
            .service_ports()
            .clone()
            .with_workspace(workspace_projection.clone())
            .with_base_workspace(observed_base),
    );
    let workspace_publication = workspace_projection
        .terminal_publication(&execution_runtime)
        .map_err(RunError::Runtime)?;

    // Check if this is an expression statement (ends with Pop)
    let is_expression_stmt = bytecode
        .instructions
        .last()
        .map(|instr| matches!(instr, runmat_vm::Instr::Pop))
        .unwrap_or(false);

    // Determine whether the final statement ended with a semicolon by inspecting the raw input.
    let is_semicolon_suppressed = {
        let toks = tokenize_detailed(input);
        toks.into_iter()
            .rev()
            .map(|t| t.token)
            .find(|token| {
                !matches!(
                    token,
                    LexToken::Newline
                        | LexToken::LineComment
                        | LexToken::BlockComment
                        | LexToken::Section
                )
            })
            .map(|t| matches!(t, LexToken::Semicolon))
            .unwrap_or(false)
    };
    let final_stmt_emit = display_context.final_stmt_emit;

    if session.verbose {
        debug!("is_semicolon_suppressed: {is_semicolon_suppressed}");
    }

    #[cfg(not(target_arch = "wasm32"))]
    if let Some((workspace, profile)) = tiered_interactive.as_ref() {
        match session
            .invoke_tiered_interactive(
                &interactive_unit,
                &execution_runtime,
                &bytecode.function_registry,
                workspace,
                profile.as_ref(),
                !(is_expression_stmt
                    && matches!(final_stmt_emit, FinalStmtEmitDisposition::Inline)),
            )
            .await
        {
            Ok(crate::session::executable::TieredInteractiveInvocation::Native(
                Ok(native),
                selected,
            )) => {
                ready_route_selection = selected;
                native_loop_backedges = native.loop_backedges.clone();
                if is_expression_stmt {
                    if let Some(expression) = native.expression.clone() {
                        if is_semicolon_suppressed {
                            suppressed_value = Some(expression);
                        } else {
                            result_value = Some(expression);
                        }
                    }
                }
                let snapshot = native.workspace.as_ref().ok_or_else(|| {
                    RunError::Runtime(
                        build_runtime_error(
                            "interactive native execution produced no workspace snapshot",
                        )
                        .with_identifier("RunMat:NativeWorkspace")
                        .build(),
                    )
                })?;
                execution_vars.retain(|name, _| snapshot.contains_key(name));
                for (name, _) in snapshot.iter() {
                    if !execution_vars.contains_key(name) {
                        execution_vars.insert(name.to_owned(), frame_extent);
                        frame_extent = frame_extent.checked_add(1).ok_or_else(|| {
                            RunError::Runtime(
                                build_runtime_error("native workspace slot extent overflow")
                                    .with_identifier("RunMat:WorkspaceBindingSlots")
                                    .build(),
                            )
                        })?;
                    }
                }
                if !display_context.single_stmt_non_assign {
                    if let Some(var_id) = display_context.first_assign_var {
                        let assignment = execution_vars
                            .iter()
                            .find_map(|(name, slot)| {
                                (*slot == var_id).then(|| snapshot.get(name)).flatten()
                            })
                            .cloned();
                        if let Some(value) = assignment {
                            if is_semicolon_suppressed {
                                suppressed_value = Some(value);
                            } else {
                                result_value = Some(value);
                            }
                        }
                    }
                }
                // The native result already retains its authentic workspace.
                // Retire the unused cold VM input instead of mirroring output
                // Values into another frame or resizing after effects.
                completed_native_root = Some(native);
                used_jit = true;
                execution_completed = true;
            }
            Ok(crate::session::executable::TieredInteractiveInvocation::Cold(selected)) => {
                ready_route_selection = selected;
            }
            Ok(crate::session::executable::TieredInteractiveInvocation::Native(
                Err(native_error),
                selected,
            )) => {
                ready_route_selection = selected;
                used_jit = true;
                execution_completed = true;
                error = Some(*native_error);
            }
            Err(admission_error) => {
                execution_completed = true;
                error = Some(admission_error);
            }
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
            is_expression_stmt,
            is_semicolon_suppressed,
            final_stmt_emit,
            workspace_publication,
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
                    frame_extent,
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
