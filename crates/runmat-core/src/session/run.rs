use super::*;

fn entrypoint_target_function(
    assembly: &runmat_hir::HirAssembly,
) -> Option<runmat_hir::FunctionId> {
    assembly
        .entrypoints
        .first()
        .map(|entrypoint| entrypoint.target)
}

fn mir_local_fact_count_for_entrypoint(
    analysis: &runmat_mir::analysis::AnalysisStore,
    assembly: &runmat_hir::HirAssembly,
) -> usize {
    let Some(entrypoint_target) = entrypoint_target_function(assembly) else {
        return analysis.mir_locals.len();
    };
    analysis
        .mir_locals
        .keys()
        .filter(|key| key.function == entrypoint_target)
        .count()
}

fn discover_known_project_symbols(source_name: Option<&str>) -> HashSet<String> {
    use runmat_config::discover_known_project_symbols_from_source_name;

    let Ok(cwd) = std::env::current_dir() else {
        return HashSet::new();
    };
    discover_known_project_symbols_from_source_name(source_name, &cwd)
}

impl RunMatSession {
    /// Execute MATLAB/Octave code
    pub async fn execute(
        &mut self,
        input: &str,
    ) -> std::result::Result<SessionExecutionResult, RunError> {
        self.execute_internal(input, false)
            .await
            .map(session_result_from_execution)
    }

    /// Execute MATLAB/Octave code and return the runtime/workspace ABI outcome.
    pub async fn execute_outcome(
        &mut self,
        input: &str,
    ) -> std::result::Result<crate::abi::ExecutionOutcome, RunError> {
        self.run(input).await
    }

    /// Parse, lower, compile, and execute input through the runtime/workspace ABI boundary.
    pub async fn run(
        &mut self,
        input: &str,
    ) -> std::result::Result<crate::abi::ExecutionOutcome, RunError> {
        let previous_workspace_names = self
            .workspace_values
            .keys()
            .cloned()
            .collect::<HashSet<_>>();
        let mut execution = self.execute_internal(input, true).await?;
        let workspace_names = execution
            .workspace_snapshot
            .values
            .iter()
            .map(|entry| entry.name.clone())
            .collect::<Vec<_>>();
        let workspace_full = execution.workspace_snapshot.full;
        let outcome = &mut execution.outcome;
        outcome.workspace_delta.upserts = self.abi_workspace_upserts(workspace_names);
        if workspace_full {
            outcome.workspace_delta.removals =
                self.abi_workspace_removals(previous_workspace_names);
            if !outcome.workspace_delta.removals.is_empty() {
                outcome.effects.push(crate::abi::ObservedEffect::Workspace(
                    crate::abi::WorkspaceEffectKind::Clear,
                ));
            }
        }
        Ok(execution.outcome)
    }

    /// Execute a structured runtime/workspace ABI request.
    pub async fn execute_request(
        &mut self,
        request: crate::abi::ExecutionRequest,
    ) -> std::result::Result<crate::abi::ExecutionOutcome, RunError> {
        let requested_outputs = request.requested_outputs.clone();
        let (source_name, source_text) = source_input_text(request.source)?;
        let previous_compat = self.compat_mode;
        let previous_top_level_await_enabled = self.top_level_await_enabled;
        let previous_source_override = self.source_name_override.clone();
        let previous_workspace_handle = self.abi_workspace_handle;

        self.compat_mode = request.compatibility;
        self.top_level_await_enabled = request.host_policy.top_level_await;
        self.source_name_override = Some(source_name);
        self.abi_workspace_handle = request.workspace;

        let result = self.execute_outcome(&source_text).await;

        self.compat_mode = previous_compat;
        self.top_level_await_enabled = previous_top_level_await_enabled;
        self.source_name_override = previous_source_override;
        self.abi_workspace_handle = previous_workspace_handle;

        result.map(|mut outcome| {
            if matches!(requested_outputs, runmat_hir::RequestedOutputCount::Zero) {
                outcome.flow = crate::abi::RuntimeFlow::NoValue;
            }
            outcome
        })
    }

    async fn execute_internal(
        &mut self,
        input: &str,
        preserve_layout_var_names: bool,
    ) -> std::result::Result<SessionExecution, RunError> {
        let _active = ActiveExecutionGuard::new(self).map_err(|err| {
            RunError::Runtime(
                build_runtime_error(err.to_string())
                    .with_identifier("RunMat:ExecutionAlreadyActive")
                    .build(),
            )
        })?;
        runmat_vm::set_call_stack_limit(self.callstack_limit);
        runmat_vm::set_error_namespace(&self.error_namespace);
        runmat_hir::set_error_namespace(&self.error_namespace);
        let exec_span = info_span!(
            "runtime.execute",
            input_len = input.len(),
            verbose = self.verbose
        );
        let _exec_guard = exec_span.enter();
        runmat_runtime::console::reset_thread_buffer();
        runmat_runtime::plotting_hooks::reset_recent_figures();
        runmat_runtime::warning_store::reset();
        runmat_builtins::set_display_format(self.format_mode);
        reset_provider_telemetry();
        self.interrupt_flag.store(false, Ordering::Relaxed);
        let _interrupt_guard =
            runmat_runtime::interrupt::replace_interrupt(Some(self.interrupt_flag.clone()));
        let start_time = Instant::now();
        self.stats.total_executions += 1;
        let debug_trace = std::env::var("RUNMAT_DEBUG_REPL").is_ok();
        let stdin_events: Arc<Mutex<Vec<StdinEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let host_async_handler = self.async_input_handler.clone();
        let stdin_events_async = Arc::clone(&stdin_events);
        let runtime_async_handler: Arc<runmat_runtime::interaction::AsyncInteractionHandler> =
            Arc::new(
                move |prompt: runmat_runtime::interaction::InteractionPromptOwned| {
                    let request_kind = match prompt.kind {
                        runmat_runtime::interaction::InteractionKind::Line { echo } => {
                            InputRequestKind::Line { echo }
                        }
                        runmat_runtime::interaction::InteractionKind::KeyPress => {
                            InputRequestKind::KeyPress
                        }
                    };
                    let request = InputRequest {
                        prompt: prompt.prompt,
                        kind: request_kind,
                    };
                    let (event_kind, echo_flag) = match &request.kind {
                        InputRequestKind::Line { echo } => (StdinEventKind::Line, *echo),
                        InputRequestKind::KeyPress => (StdinEventKind::KeyPress, false),
                    };
                    let mut event = StdinEvent {
                        prompt: request.prompt.clone(),
                        kind: event_kind,
                        echo: echo_flag,
                        value: None,
                        error: None,
                    };

                    let stdin_events_async = Arc::clone(&stdin_events_async);
                    let host_async_handler = host_async_handler.clone();
                    Box::pin(async move {
                        let resp: Result<InputResponse, String> =
                            if let Some(handler) = host_async_handler {
                                handler(request).await
                            } else {
                                match &request.kind {
                                    InputRequestKind::Line { echo } => {
                                        runmat_runtime::interaction::default_read_line(
                                            &request.prompt,
                                            *echo,
                                        )
                                        .map(InputResponse::Line)
                                    }
                                    InputRequestKind::KeyPress => {
                                        runmat_runtime::interaction::default_wait_for_key(
                                            &request.prompt,
                                        )
                                        .map(|_| InputResponse::KeyPress)
                                    }
                                }
                            };

                        let resp = resp.inspect_err(|err| {
                            event.error = Some(err.clone());
                            if let Ok(mut guard) = stdin_events_async.lock() {
                                guard.push(event.clone());
                            }
                        })?;

                        let interaction_resp = match resp {
                            InputResponse::Line(value) => {
                                event.value = Some(value.clone());
                                if let Ok(mut guard) = stdin_events_async.lock() {
                                    guard.push(event);
                                }
                                runmat_runtime::interaction::InteractionResponse::Line(value)
                            }
                            InputResponse::KeyPress => {
                                if let Ok(mut guard) = stdin_events_async.lock() {
                                    guard.push(event);
                                }
                                runmat_runtime::interaction::InteractionResponse::KeyPress
                            }
                        };
                        Ok(interaction_resp)
                    })
                },
            );
        let _async_input_guard =
            runmat_runtime::interaction::replace_async_handler(Some(runtime_async_handler));

        // Install a stateless expression evaluator for `input()` numeric parsing.
        //
        // The hook runs the full parse → lower → compile → interpret pipeline so
        // that users can type arbitrary MATLAB expressions at an input() prompt:
        // `sqrt(2)`, `pi/2`, `ones(3)`, `[1 2; 3 4]`, etc.
        //
        // Stack-overflow hazard: the hook calls runmat_vm::interpret() while
        // the outer interpret() is already on the call stack. On WASM the JS event
        // loop drives both as async state-machines and the WASM linear stack is
        // large, so nesting is safe. On native the default thread stack is too
        // small for two nested interpret() invocations, so we instead run the inner
        // interpret() on a dedicated thread that has its own 16 MB stack and block
        // the calling future synchronously on the result (safe because the native
        // executor — futures::executor::block_on — is already synchronous).
        let compat = self.compat_mode;
        let top_level_await_enabled = self.top_level_await_enabled;
        let source_name_for_eval_hook = self.current_source_name().to_string();
        let known_project_symbols_for_eval_hook = Arc::new(discover_known_project_symbols(Some(
            source_name_for_eval_hook.as_str(),
        )));
        let _eval_hook_guard =
            runmat_runtime::interaction::replace_eval_hook(Some(std::sync::Arc::new(
                move |expr: String| -> runmat_runtime::interaction::EvalHookFuture {
                    // Shared eval logic, used by both the WASM async path and the
                    // native thread path below.
                    async fn eval_expr(
                        expr: String,
                        compat: runmat_parser::CompatMode,
                        top_level_await_enabled: bool,
                        known_project_symbols: Arc<HashSet<String>>,
                    ) -> Result<Value, RuntimeError> {
                        let wrapped = format!("__runmat_input_result__ = ({expr});");
                        let ast = parse_with_options(&wrapped, ParserOptions::new(compat))
                            .map_err(|e| {
                                build_runtime_error(format!("input: parse error: {e}"))
                                    .with_identifier("RunMat:input:ParseError")
                                    .build()
                            })?;
                        let lowering = runmat_hir::lower(
                            &ast,
                            &LoweringContext::new(&HashMap::new())
                                .with_known_project_symbols(&known_project_symbols)
                                .with_runmat_extensions_enabled(compat.allows_runmat_extensions())
                                .with_top_level_await_enabled(top_level_await_enabled),
                        )
                        .map_err(|e| {
                            build_runtime_error(format!("input: lowering error: {e}"))
                                .with_identifier("RunMat:input:LowerError")
                                .build()
                        })?;
                        let bc =
                            compile_eval_hook_bytecode(&lowering).map_err(RuntimeError::from)?;
                        let result_idx = bc.var_names.iter().find_map(|(idx, name)| {
                            (name == "__runmat_input_result__").then_some(*idx)
                        });
                        let vars = runmat_vm::interpret(&bc).await?;
                        result_idx
                            .and_then(|idx| vars.get(idx).cloned())
                            .ok_or_else(|| {
                                build_runtime_error("input: expression produced no value")
                                    .with_identifier("RunMat:input:NoValue")
                                    .build()
                            })
                    }

                    #[cfg(target_arch = "wasm32")]
                    {
                        // On WASM: await the inner interpret() directly. The JS async
                        // runtime handles both futures as cooperative state-machines and
                        // the WASM linear stack is large enough for the extra frames.
                        Box::pin(eval_expr(
                            expr,
                            compat,
                            top_level_await_enabled,
                            Arc::clone(&known_project_symbols_for_eval_hook),
                        ))
                    }

                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        // On native: run interpret() on a dedicated thread so it gets
                        // its own 16 MB stack, fully isolated from the outer interpret()
                        // call stack. The result is sent back via a tokio oneshot channel
                        // and awaited asynchronously so the tokio worker thread is never
                        // blocked by a synchronous recv().
                        let (tx, rx) = tokio::sync::oneshot::channel();
                        let known_project_symbols =
                            Arc::clone(&known_project_symbols_for_eval_hook);
                        let spawn_result = std::thread::Builder::new()
                            .stack_size(16 * 1024 * 1024)
                            .spawn(move || {
                                let result = futures::executor::block_on(eval_expr(
                                    expr,
                                    compat,
                                    top_level_await_enabled,
                                    known_project_symbols,
                                ));
                                let _ = tx.send(result);
                            });
                        Box::pin(async move {
                            spawn_result.map_err(|err| {
                                build_runtime_error(format!(
                                    "input: failed to spawn eval thread: {err}"
                                ))
                                .with_identifier("RunMat:input:EvalThreadSpawnFailed")
                                .build()
                            })?;
                            rx.await.unwrap_or_else(|_| {
                                Err(build_runtime_error("input: eval thread panicked")
                                    .with_identifier("RunMat:input:EvalThreadPanic")
                                    .build())
                            })
                        })
                    }
                },
            )));

        if self.verbose {
            debug!("Executing: {}", input.trim());
        }

        let _source_guard = runmat_runtime::source_context::replace_current_source(Some(input));

        let PreparedExecution {
            ast,
            lowering,
            mir,
            mut bytecode,
            semantic_function_registry_after_success,
            next_semantic_function_id_after_success,
        } = self.compile_input(input)?;
        if self.verbose {
            debug!("AST: {ast:?}");
        }
        let display = execution_display_context(&lowering.assembly, bytecode.layout.as_ref());
        let display_context = display.context;
        let display_var_ids = display.display_var_ids;
        let semantic_stmt_count = semantic_entry_statement_count(&lowering.assembly);
        let execution_vars = execution_workspace_mapping(&bytecode);
        let max_var_id = execution_vars.values().copied().max().unwrap_or(0);
        if debug_trace {
            debug!(?execution_vars, "[repl] execution vars");
        }
        if debug_trace {
            debug!(workspace_values_before = ?self.workspace_values, "[repl] workspace snapshot before execution");
        }
        let id_to_name: HashMap<usize, String> = execution_vars
            .iter()
            .map(|(name, var_id)| (*var_id, name.clone()))
            .collect();
        let mut assigned_this_execution: HashSet<String> = HashSet::new();
        let assigned_snapshot: HashSet<String> = execution_vars
            .keys()
            .filter(|name| self.workspace_values.contains_key(name.as_str()))
            .cloned()
            .collect();
        let prev_assigned_snapshot = assigned_snapshot.clone();
        if debug_trace {
            debug!(?assigned_snapshot, "[repl] assigned snapshot");
        }
        let _pending_workspace_guard =
            runmat_vm::push_pending_workspace(execution_vars.clone(), assigned_snapshot.clone());
        if self.verbose {
            debug!("HIR generated successfully");
        }

        if preserve_layout_var_names && bytecode.layout.is_some() {
            for (slot, name) in &id_to_name {
                bytecode
                    .var_names
                    .entry(*slot)
                    .or_insert_with(|| name.clone());
            }
        } else {
            bytecode.var_names = id_to_name.clone();
        }
        if self.verbose {
            debug!(
                "Bytecode compiled: {} instructions",
                bytecode.instructions.len()
            );
        }

        #[cfg(not(target_arch = "wasm32"))]
        let fusion_snapshot = if self.emit_fusion_plan {
            let analysis = runmat_mir::analysis::analyze_assembly(&mir);
            build_fusion_snapshot(
                bytecode.accel_graph.as_ref(),
                &bytecode.fusion_groups,
                &bytecode
                    .semantic_fusion_metadata
                    .mir_fusion_candidate_groups,
                &bytecode
                    .semantic_fusion_metadata
                    .semantic_instruction_windows,
                Some(crate::fusion::FusionPlannerMetadata {
                    source: "semantic-mir-analysis-runtime".to_string(),
                    accel_graph_state: if bytecode.accel_graph.is_some() {
                        "present".to_string()
                    } else {
                        "missing".to_string()
                    },
                    mir_local_fact_count: mir_local_fact_count_for_entrypoint(
                        &analysis,
                        &lowering.assembly,
                    ),
                    mir_diagnostic_count: analysis.diagnostics.len(),
                    mir_fusion_signal_count: bytecode
                        .semantic_fusion_metadata
                        .mir_fusion_signal_count,
                    mir_fusion_candidate_group_count: bytecode
                        .semantic_fusion_metadata
                        .mir_fusion_candidate_group_count,
                    mir_semantic_instruction_window_count: bytecode
                        .semantic_fusion_metadata
                        .semantic_instruction_window_count,
                }),
            )
        } else {
            None
        };
        #[cfg(target_arch = "wasm32")]
        let fusion_snapshot: Option<FusionPlanSnapshot> = None;

        // Prepare variable array with existing values before execution
        self.prepare_variable_array_for_execution(&bytecode, &execution_vars, debug_trace);

        if self.verbose {
            debug!(
                "Variable array after preparation: {:?}",
                self.variable_array
            );
            debug!("Bytecode instructions: {:?}", bytecode.instructions);
        }

        #[cfg(feature = "jit")]
        let mut used_jit = false;
        #[cfg(not(feature = "jit"))]
        let used_jit = false;
        #[cfg(feature = "jit")]
        let mut execution_completed = false;
        #[cfg(not(feature = "jit"))]
        let execution_completed = false;
        let mut result_value: Option<Value> = None; // Always start fresh for each execution
        let mut suppressed_value: Option<Value> = None; // Track value for type info when suppressed
        let mut error = None;
        let mut workspace_updates: Vec<WorkspaceEntry> = Vec::new();
        let mut workspace_snapshot_force_full = false;
        let mut ans_update: Option<(usize, Value)> = None;

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

        if self.verbose {
            debug!("Semantic entry body len: {semantic_stmt_count}");
            if let Some(stmt) = semantic_first_entry_statement(&lowering.assembly) {
                debug!("Semantic HIR statement: {stmt:?}");
            }
            debug!("is_semicolon_suppressed: {is_semicolon_suppressed}");
        }

        // Use JIT for assignments, interpreter for expressions (to capture results properly)
        #[cfg(feature = "jit")]
        {
            if let Some(ref mut jit_engine) = &mut self.jit_engine {
                if !is_expression_stmt {
                    // Ensure variable array is large enough
                    if self.variable_array.len() < bytecode.var_count {
                        self.variable_array
                            .resize(bytecode.var_count, Value::Num(0.0));
                    }

                    if self.verbose {
                        debug!(
                            "JIT path for assignment: variable_array size: {}, bytecode.var_count: {}",
                            self.variable_array.len(),
                            bytecode.var_count
                        );
                    }

                    // Use JIT for assignments
                    match jit_engine.execute_or_compile(&bytecode, &mut self.variable_array) {
                        Ok((_, actual_used_jit)) => {
                            used_jit = actual_used_jit;
                            execution_completed = true;
                            if actual_used_jit {
                                self.stats.jit_compiled += 1;
                            } else {
                                self.stats.interpreter_fallback += 1;
                            }
                            if let Some(var_id) = display_context.first_assign_var {
                                if let Some(name) = id_to_name.get(&var_id) {
                                    assigned_this_execution.insert(name.clone());
                                }
                                if var_id < self.variable_array.len() {
                                    let assignment_value = self.variable_array[var_id].clone();
                                    if !is_semicolon_suppressed {
                                        result_value = Some(assignment_value);
                                        if self.verbose {
                                            debug!("JIT assignment result: {result_value:?}");
                                        }
                                    } else {
                                        suppressed_value = Some(assignment_value);
                                        if self.verbose {
                                            debug!("JIT assignment suppressed due to semicolon, captured for type info");
                                        }
                                    }
                                }
                            }

                            if self.verbose {
                                debug!(
                                    "{} assignment successful, variable_array: {:?}",
                                    if actual_used_jit {
                                        "JIT"
                                    } else {
                                        "Interpreter"
                                    },
                                    self.variable_array
                                );
                            }
                        }
                        Err(e) => {
                            if self.verbose {
                                debug!("JIT execution failed: {e}, using interpreter");
                            }
                            // Fall back to interpreter
                        }
                    }
                }
            }
        }

        // Use interpreter if JIT failed or is disabled
        if !execution_completed {
            if self.verbose {
                debug!(
                    "Interpreter path: variable_array size: {}, bytecode.var_count: {}",
                    self.variable_array.len(),
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
                if self.variable_array.len() <= temp_var_id {
                    self.variable_array.resize(temp_var_id + 1, Value::Num(0.0));
                }

                if self.verbose {
                    debug!(
                        "Modified expression bytecode, new instructions: {:?}",
                        execution_bytecode.instructions
                    );
                }
            }

            match self.interpret_with_context(&execution_bytecode).await {
                Ok(runmat_vm::InterpreterOutcome::Completed(results)) => {
                    // Only increment interpreter_fallback if JIT wasn't attempted
                    if !self.has_jit() || is_expression_stmt {
                        self.stats.interpreter_fallback += 1;
                    }
                    if self.verbose {
                        debug!("Interpreter results: {results:?}");
                    }

                    // Handle assignment statements (x = 42 should show the assigned value unless suppressed)
                    if semantic_stmt_count == 1 {
                        if let Some(var_id) = display_context.first_assign_var {
                            if let Some(name) = id_to_name.get(&var_id) {
                                assigned_this_execution.insert(name.clone());
                            }
                            // For assignments, capture the assigned value for both display and type info
                            if var_id < self.variable_array.len() {
                                let assignment_value = self.variable_array[var_id].clone();
                                if !is_semicolon_suppressed {
                                    result_value = Some(assignment_value);
                                    if self.verbose {
                                        debug!("Interpreter assignment result: {result_value:?}");
                                    }
                                } else {
                                    suppressed_value = Some(assignment_value);
                                    if self.verbose {
                                        debug!("Interpreter assignment suppressed due to semicolon, captured for type info");
                                    }
                                }
                            }
                        } else if !is_expression_stmt
                            && !results.is_empty()
                            && !is_semicolon_suppressed
                            && matches!(final_stmt_emit, FinalStmtEmitDisposition::NeedsFallback)
                        {
                            result_value = Some(results[0].clone());
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
                        if temp_var_id < self.variable_array.len() {
                            let expression_value = self.variable_array[temp_var_id].clone();
                            if !is_semicolon_suppressed {
                                // Capture for 'ans' update when output is not suppressed
                                ans_update = Some((temp_var_id, expression_value.clone()));
                                result_value = Some(expression_value);
                                if self.verbose {
                                    debug!("Expression result from temp var {temp_var_id}: {result_value:?}");
                                }
                            } else {
                                suppressed_value = Some(expression_value);
                                if self.verbose {
                                    debug!("Expression suppressed, captured for type info from temp var {temp_var_id}: {suppressed_value:?}");
                                }
                            }
                        }
                    } else if !is_semicolon_suppressed
                        && matches!(final_stmt_emit, FinalStmtEmitDisposition::NeedsFallback)
                        && result_value.is_none()
                    {
                        result_value = results.into_iter().last();
                        if self.verbose {
                            debug!("Fallback result from interpreter: {result_value:?}");
                        }
                    }

                    if self.verbose {
                        debug!("Final result_value: {result_value:?}");
                    }
                    debug!("Interpreter execution successful");
                }

                Err(e) => {
                    debug!("Interpreter execution failed: {e}");
                    error = Some(e);
                }
            }
        }

        let last_assign_var = display_context.last_assign_var;
        let last_expr_emits = display_context.last_expr_emits;
        if !is_semicolon_suppressed && result_value.is_none() {
            if last_assign_var.is_some() || last_expr_emits {
                if let Some(value) = runmat_runtime::console::take_last_value_output() {
                    result_value = Some(value);
                }
            }
            if result_value.is_none() {
                if last_assign_var.is_some() {
                    if let Some(var_id) = last_emit_var_index(&bytecode) {
                        if var_id < self.variable_array.len() {
                            result_value = Some(self.variable_array[var_id].clone());
                        }
                    }
                }
                if result_value.is_none() {
                    if let Some(var_id) = last_assign_var {
                        if var_id < self.variable_array.len() {
                            result_value = Some(self.variable_array[var_id].clone());
                        }
                    }
                }
            }
        }

        let execution_time = start_time.elapsed();
        let execution_time_ms = execution_time.as_millis() as u64;

        self.stats.total_execution_time_ms += execution_time_ms;
        self.stats.average_execution_time_ms =
            self.stats.total_execution_time_ms as f64 / self.stats.total_executions as f64;

        // Update variable names mapping and function definitions if execution was successful
        if error.is_none() {
            if let Some((mutated_names, assigned)) = runmat_vm::take_updated_workspace_state() {
                if debug_trace {
                    debug!(
                        ?mutated_names,
                        ?assigned,
                        "[repl] mutated names and assigned return values"
                    );
                }
                self.workspace_bindings.clear();
                for (name, slot) in &mutated_names {
                    self.bind_workspace_slot(name.clone(), *slot);
                }
                let previous_workspace = self.workspace_values.clone();
                let current_names: HashSet<String> = assigned
                    .iter()
                    .filter(|name| {
                        mutated_names
                            .get(*name)
                            .map(|var_id| *var_id < self.variable_array.len())
                            .unwrap_or(false)
                    })
                    .cloned()
                    .collect();
                let removed_names: HashSet<String> = previous_workspace
                    .keys()
                    .filter(|name| !current_names.contains(*name))
                    .cloned()
                    .collect();
                let mut rebuilt_workspace = HashMap::new();
                let mut changed_names: HashSet<String> = assigned
                    .difference(&prev_assigned_snapshot)
                    .cloned()
                    .collect();
                changed_names.extend(assigned_this_execution.iter().cloned());

                for name in &current_names {
                    let Some(var_id) = mutated_names.get(name).copied() else {
                        continue;
                    };
                    if var_id >= self.variable_array.len() {
                        continue;
                    }
                    let value_clone = self.variable_array[var_id].clone();
                    if previous_workspace.get(name) != Some(&value_clone) {
                        changed_names.insert(name.clone());
                    }
                    rebuilt_workspace.insert(name.clone(), value_clone);
                }

                if debug_trace {
                    debug!(?changed_names, ?removed_names, "[repl] workspace changes");
                }

                self.workspace_values = rebuilt_workspace;
                if !removed_names.is_empty() {
                    workspace_snapshot_force_full = true;
                } else {
                    for name in changed_names {
                        if let Some(value_clone) = self.workspace_values.get(&name).cloned() {
                            workspace_updates.push(workspace_entry(&name, &value_clone));
                            if debug_trace {
                                debug!(name, ?value_clone, "[repl] workspace update");
                            }
                        }
                    }
                }
            } else {
                for name in &assigned_this_execution {
                    if let Some(var_id) =
                        id_to_name
                            .iter()
                            .find_map(|(vid, n)| if n == name { Some(*vid) } else { None })
                    {
                        if var_id < self.variable_array.len() {
                            let value_clone = self.variable_array[var_id].clone();
                            self.workspace_values
                                .insert(name.clone(), value_clone.clone());
                            workspace_updates.push(workspace_entry(name, &value_clone));
                        }
                    }
                }
            }
            self.semantic_function_registry = semantic_function_registry_after_success;
            self.next_semantic_function_id = next_semantic_function_id_after_success;
            // Apply 'ans' update if applicable (persisting expression result)
            if let Some((var_id, value)) = ans_update {
                self.bind_workspace_slot("ans".to_string(), var_id);
                self.workspace_values.insert("ans".to_string(), value);
                if debug_trace {
                    println!("Updated 'ans' to var_id {}", var_id);
                }
            }
        }

        if self.verbose {
            debug!("Execution completed in {execution_time_ms}ms (JIT: {used_jit})");
        }

        if !is_expression_stmt
            && !is_semicolon_suppressed
            && matches!(final_stmt_emit, FinalStmtEmitDisposition::NeedsFallback)
            && result_value.is_none()
        {
            if let Some(v) = self
                .variable_array
                .iter()
                .rev()
                .find(|v| !matches!(v, Value::Num(0.0)))
                .cloned()
            {
                result_value = Some(v);
            }
        }

        if !is_semicolon_suppressed
            && (!display_var_ids.is_empty()
                || matches!(final_stmt_emit, FinalStmtEmitDisposition::NeedsFallback)
                || display_context.single_assign_var.is_some()
                || (is_expression_stmt
                    && matches!(final_stmt_emit, FinalStmtEmitDisposition::Inline)))
        {
            if runmat_runtime::console::take_last_value_output().is_none() {
                if display_var_ids.is_empty() {
                    if let Some(value) = result_value.as_ref() {
                        let label = last_emit_var_index(&bytecode)
                            .and_then(|var_id| id_to_name.get(&var_id).cloned())
                            .or_else(|| {
                                determine_display_label_from_context(
                                    display_context.single_assign_var,
                                    &id_to_name,
                                    is_expression_stmt,
                                    display_context.single_stmt_non_assign,
                                )
                            });
                        runmat_runtime::console::record_value_output(label.as_deref(), value);
                    }
                } else {
                    for var_id in display_var_ids {
                        if let (Some(label), Some(display_value)) =
                            (id_to_name.get(&var_id), self.variable_array.get(var_id))
                        {
                            runmat_runtime::console::record_value_output(
                                Some(label.as_str()),
                                display_value,
                            );
                        }
                    }
                }
            }
        }

        // Generate type info if we have a suppressed value
        let type_info = suppressed_value.as_ref().map(format_type_info);

        let streams = runmat_runtime::console::take_thread_buffer()
            .into_iter()
            .map(|entry| ExecutionStreamEntry {
                stream: match entry.stream {
                    runmat_runtime::console::ConsoleStream::Stdout => ExecutionStreamKind::Stdout,
                    runmat_runtime::console::ConsoleStream::Stderr => ExecutionStreamKind::Stderr,
                    runmat_runtime::console::ConsoleStream::ClearScreen => {
                        ExecutionStreamKind::ClearScreen
                    }
                },
                text: entry.text,
                timestamp_ms: entry.timestamp_ms,
            })
            .collect();
        let (workspace_entries, snapshot_full) = if workspace_snapshot_force_full {
            let mut entries: Vec<WorkspaceEntry> = self
                .workspace_values
                .iter()
                .map(|(name, value)| workspace_entry(name, value))
                .collect();
            entries.sort_by(|a, b| a.name.cmp(&b.name));
            (entries, true)
        } else if workspace_updates.is_empty() {
            if self.workspace_values.is_empty() {
                (workspace_updates, false)
            } else {
                let mut entries: Vec<WorkspaceEntry> = self
                    .workspace_values
                    .iter()
                    .map(|(name, value)| workspace_entry(name, value))
                    .collect();
                entries.sort_by(|a, b| a.name.cmp(&b.name));
                (entries, true)
            }
        } else {
            (workspace_updates, false)
        };
        let workspace_snapshot = self.build_workspace_snapshot(workspace_entries, snapshot_full);
        let figures_touched = runmat_runtime::plotting_hooks::take_recent_figures();
        let stdin_events = stdin_events
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default();

        let warnings = runmat_runtime::warning_store::take_all();

        if let Some(runtime_error) = &mut error {
            self.normalize_error_namespace(runtime_error);
            self.populate_callstack(runtime_error);
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
                span: None,
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

        let profiling = gather_profiling(execution_time_ms);
        let outcome = crate::abi::ExecutionOutcome {
            flow: public_value
                .clone()
                .map(crate::abi::RuntimeFlow::Single)
                .unwrap_or(crate::abi::RuntimeFlow::NoValue),
            workspace_delta: crate::abi::WorkspaceDelta {
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
            profiling,
        };

        self.format_mode = runmat_builtins::get_display_format();
        Ok(SessionExecution {
            outcome,
            workspace_snapshot,
            public_value,
            error,
            warnings,
        })
    }

    /// Interpret bytecode with persistent variable context
    async fn interpret_with_context(
        &mut self,
        bytecode: &runmat_vm::Bytecode,
    ) -> Result<runmat_vm::InterpreterOutcome, RuntimeError> {
        let source_name = self.current_source_name().to_string();
        runmat_vm::interpret_with_vars(
            bytecode,
            &mut self.variable_array,
            Some(source_name.as_str()),
        )
        .await
    }

    fn abi_workspace_upserts(
        &self,
        workspace_names: Vec<String>,
    ) -> Vec<crate::abi::WorkspaceBindingValue> {
        let mut workspace_names = workspace_names;
        workspace_names.sort();
        workspace_names.dedup();
        workspace_names
            .into_iter()
            .filter_map(|name| {
                let value = self.workspace_values.get(&name)?.clone();
                let binding = runmat_hir::BindingName(name);
                let key = self
                    .workspace_bindings
                    .get(&binding.0)
                    .map(|binding| binding.key.clone())
                    .unwrap_or_else(|| self.workspace_binding_key(&binding.0));
                Some(crate::abi::WorkspaceBindingValue { key, value })
            })
            .collect()
    }

    fn abi_workspace_removals(
        &self,
        previous_workspace_names: HashSet<String>,
    ) -> Vec<crate::abi::WorkspaceBindingKey> {
        let mut removed_names = previous_workspace_names
            .into_iter()
            .filter(|name| !self.workspace_values.contains_key(name))
            .collect::<Vec<_>>();
        removed_names.sort();
        removed_names
            .into_iter()
            .map(|name| self.workspace_binding_key(&name))
            .collect()
    }
}

fn compile_eval_hook_bytecode(
    lowering: &runmat_hir::LoweringResult,
) -> Result<runmat_vm::Bytecode, runmat_vm::CompileError> {
    let entrypoint = lowering.assembly.entrypoints.first().ok_or_else(|| {
        runmat_vm::CompileError::new("semantic eval hook compile requires an entrypoint")
    })?;
    let mir = runmat_mir::lowering::lower_assembly(&lowering.assembly)
        .map_err(runmat_vm::CompileError::from)?;
    runmat_vm::compile(&lowering.assembly, &mir, entrypoint.id)
}

fn execution_workspace_mapping(bytecode: &runmat_vm::Bytecode) -> HashMap<String, usize> {
    let Some(layout) = &bytecode.layout else {
        return HashMap::new();
    };
    let mut mapping = HashMap::new();
    for entrypoint in layout.entrypoints.values() {
        for export in &entrypoint.exports {
            mapping.insert(export.name.clone(), export.slot.0);
        }
    }
    mapping
}

fn semantic_entry_function(assembly: &runmat_hir::HirAssembly) -> Option<&runmat_hir::HirFunction> {
    let entrypoint = assembly.entrypoints.first()?;
    assembly
        .functions
        .iter()
        .find(|function| function.id == entrypoint.target)
}

fn semantic_entry_statement_count(assembly: &runmat_hir::HirAssembly) -> usize {
    semantic_entry_function(assembly)
        .map(|function| function.body.statements.len())
        .unwrap_or(0)
}

fn semantic_first_entry_statement(
    assembly: &runmat_hir::HirAssembly,
) -> Option<&runmat_hir::HirStmt> {
    semantic_entry_function(assembly)?.body.statements.first()
}

struct SessionExecution {
    outcome: crate::abi::ExecutionOutcome,
    workspace_snapshot: WorkspaceSnapshot,
    public_value: Option<Value>,
    error: Option<RuntimeError>,
    warnings: Vec<runmat_runtime::warning_store::RuntimeWarning>,
}

fn session_result_from_execution(execution: SessionExecution) -> SessionExecutionResult {
    SessionExecutionResult {
        value: execution.public_value,
        execution_time_ms: execution.outcome.execution_time_ms,
        used_jit: execution.outcome.used_jit,
        error: execution.error,
        type_info: execution.outcome.type_info,
        streams: execution.outcome.streams,
        workspace: execution.workspace_snapshot,
        figures_touched: execution.outcome.figures_touched,
        warnings: execution.warnings,
        profiling: execution.outcome.profiling,
        fusion_plan: execution.outcome.fusion_plan,
        stdin_events: execution.outcome.stdin_events,
    }
}

fn source_input_text(
    source: crate::abi::SourceInput,
) -> std::result::Result<(String, String), RunError> {
    match source {
        crate::abi::SourceInput::Text { name, text } => Ok((name, text)),
        crate::abi::SourceInput::Path(path) => {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let resolved_path = resolve_path_source_input(&path)?;
                let text = std::fs::read_to_string(&resolved_path).map_err(|err| {
                    RunError::Runtime(
                        build_runtime_error(format!(
                            "failed to read source path '{}': {err}",
                            resolved_path.display()
                        ))
                        .with_identifier("RunMat:SourceReadFailed")
                        .build(),
                    )
                })?;
                Ok((resolved_path.to_string_lossy().to_string(), text))
            }

            #[cfg(target_arch = "wasm32")]
            {
                Err(RunError::Runtime(
                    build_runtime_error("path source execution is unavailable on wasm")
                        .with_identifier("RunMat:PathSourceUnavailable")
                        .build(),
                ))
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_path_source_input(path: &str) -> std::result::Result<std::path::PathBuf, RunError> {
    use runmat_config::resolve_project_source_input_from;
    use std::path::Path;

    let cwd = std::env::current_dir().map_err(|err| {
        RunError::Runtime(
            build_runtime_error(format!(
                "failed to resolve current working directory while resolving source path '{path}': {err}"
            ))
            .with_identifier("RunMat:SourceResolveFailed")
            .build(),
        )
    })?;

    resolve_project_source_input_from(&cwd, Path::new(path)).map_err(|err| {
        RunError::Runtime(
            build_runtime_error(format!(
                "failed to resolve source input '{}' from working directory {}: {}",
                path,
                cwd.display(),
                err
            ))
            .with_identifier("RunMat:EntrypointResolveFailed")
            .build(),
        )
    })
}

#[cfg(test)]
mod tests {
    #[cfg(not(target_arch = "wasm32"))]
    use super::discover_known_project_symbols;
    #[cfg(not(target_arch = "wasm32"))]
    use super::source_input_text;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::RunError;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::abi::SourceInput;
    #[cfg(not(target_arch = "wasm32"))]
    use std::fs;
    #[cfg(not(target_arch = "wasm32"))]
    use std::sync::Mutex;

    #[cfg(not(target_arch = "wasm32"))]
    static CWD_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn source_input_path_resolves_named_manifest_entrypoint() {
        let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();
        fs::write(
            tmp.path().join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[[entrypoints]]
name = "main"
path = "src/main"
"#,
        )
        .unwrap();
        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();
        let (source_name, source_text) = source_input_text(SourceInput::Path("main".to_string()))
            .expect("named entrypoint should resolve");
        std::env::set_current_dir(original).unwrap();
        let resolved = std::path::PathBuf::from(source_name)
            .canonicalize()
            .unwrap();
        let expected = tmp.path().join("src/main.m").canonicalize().unwrap();
        assert_eq!(
            resolved, expected,
            "resolved source path should match manifest entrypoint target"
        );
        assert_eq!(source_text, "x = 1;");
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn source_input_path_infers_m_extension_for_relative_path() {
        let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();
        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();

        let (source_name, source_text) =
            source_input_text(SourceInput::Path("src/main".to_string()))
                .expect("path without extension should infer .m");

        std::env::set_current_dir(prev).unwrap();

        assert!(source_name.ends_with("src/main.m"));
        assert_eq!(source_text.trim(), "x = 1;");
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn source_input_path_errors_for_invalid_named_entrypoint_target() {
        let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(
            tmp.path().join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[[entrypoints]]
name = "server"
module = "app.server"
function = "main"
"#,
        )
        .unwrap();
        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();
        let err = source_input_text(SourceInput::Path("server".to_string()))
            .expect_err("invalid module/function entrypoint should report resolve error");
        std::env::set_current_dir(original).unwrap();
        let RunError::Runtime(runtime_err) = err else {
            panic!("expected runtime error");
        };
        assert_eq!(runtime_err.identifier.as_deref(), Some("RunMat:EntrypointResolveFailed"));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn discover_known_project_symbols_reads_manifest_source_context() {
        let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("+stats")).unwrap();
        fs::write(
            tmp.path().join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
        )
        .unwrap();
        fs::write(
            tmp.path().join("+stats/summarize.m"),
            "function y = summarize(x); y = x; end",
        )
        .unwrap();
        fs::write(tmp.path().join("main.m"), "x = 1;").unwrap();
        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();

        let symbols = discover_known_project_symbols(Some(
            tmp.path().join("main.m").to_string_lossy().as_ref(),
        ));

        std::env::set_current_dir(prev).unwrap();
        assert!(
            symbols.contains("stats.summarize"),
            "source-context discovery should include project symbols for eval-hook lowering"
        );
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn discover_known_project_symbols_includes_dependency_alias_qualified_names() {
        let _guard = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        let tmp = tempfile::TempDir::new().unwrap();
        let dep_root = tmp.path().join("deps/statslib");
        fs::create_dir_all(&dep_root).unwrap();
        fs::write(
            tmp.path().join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["."]

[dependencies]
statsdep = { path = "deps/statslib" }
"#,
        )
        .unwrap();
        fs::write(
            dep_root.join("runmat.toml"),
            r#"
[package]
name = "statslib"

[sources]
roots = ["."]
"#,
        )
        .unwrap();
        fs::write(
            dep_root.join("summarize.m"),
            "function y = summarize(x); y = x; end",
        )
        .unwrap();
        fs::write(tmp.path().join("main.m"), "x = 1;").unwrap();
        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();

        let symbols = discover_known_project_symbols(Some(
            tmp.path().join("main.m").to_string_lossy().as_ref(),
        ));

        std::env::set_current_dir(prev).unwrap();
        assert!(
            symbols.contains("summarize"),
            "expected base dependency symbol in known-project discovery"
        );
        assert!(
            symbols.contains("statslib.summarize"),
            "expected package-qualified dependency symbol in known-project discovery"
        );
        assert!(
            symbols.contains("statsdep.summarize"),
            "expected dependency-alias-qualified symbol in known-project discovery"
        );
    }
}
