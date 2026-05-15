use super::*;

impl RunMatSession {
    /// Execute MATLAB/Octave code
    pub async fn execute(&mut self, input: &str) -> std::result::Result<ExecutionResult, RunError> {
        self.run(input).await
    }

    /// Parse, lower, compile, and execute input.
    pub async fn run(&mut self, input: &str) -> std::result::Result<ExecutionResult, RunError> {
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
        let _eval_hook_guard =
            runmat_runtime::interaction::replace_eval_hook(Some(std::sync::Arc::new(
                move |expr: String| -> runmat_runtime::interaction::EvalHookFuture {
                    // Shared eval logic, used by both the WASM async path and the
                    // native thread path below.
                    async fn eval_expr(
                        expr: String,
                        compat: runmat_parser::CompatMode,
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
                            &LoweringContext::new(&HashMap::new(), &HashMap::new()),
                        )
                        .map_err(|e| {
                            build_runtime_error(format!("input: lowering error: {e}"))
                                .with_identifier("RunMat:input:LowerError")
                                .build()
                        })?;
                        let result_idx = lowering.variables.get("__runmat_input_result__").copied();
                        let bc = runmat_vm::compile(&lowering.hir, &HashMap::new())
                            .map_err(RuntimeError::from)?;
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
                        Box::pin(eval_expr(expr, compat))
                    }

                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        // On native: run interpret() on a dedicated thread so it gets
                        // its own 16 MB stack, fully isolated from the outer interpret()
                        // call stack. The result is sent back via a tokio oneshot channel
                        // and awaited asynchronously so the tokio worker thread is never
                        // blocked by a synchronous recv().
                        let (tx, rx) = tokio::sync::oneshot::channel();
                        let spawn_result = std::thread::Builder::new()
                            .stack_size(16 * 1024 * 1024)
                            .spawn(move || {
                                let result = futures::executor::block_on(eval_expr(expr, compat));
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
            mut bytecode,
        } = self.compile_input(input)?;
        if self.verbose {
            debug!("AST: {ast:?}");
        }
        let (hir, updated_vars, updated_functions, var_names_map) = (
            lowering.hir,
            lowering.variables,
            lowering.functions,
            lowering.var_names,
        );
        let max_var_id = updated_vars.values().copied().max().unwrap_or(0);
        if debug_trace {
            debug!(?updated_vars, "[repl] updated_vars");
        }
        if debug_trace {
            debug!(workspace_values_before = ?self.workspace_values, "[repl] workspace snapshot before execution");
        }
        let id_to_name: HashMap<usize, String> = var_names_map
            .iter()
            .map(|(var_id, name)| (var_id.0, name.clone()))
            .collect();
        let mut assigned_this_execution: HashSet<String> = HashSet::new();
        let assigned_snapshot: HashSet<String> = updated_vars
            .keys()
            .filter(|name| self.workspace_values.contains_key(name.as_str()))
            .cloned()
            .collect();
        let prev_assigned_snapshot = assigned_snapshot.clone();
        if debug_trace {
            debug!(?assigned_snapshot, "[repl] assigned snapshot");
        }
        let _pending_workspace_guard =
            runmat_vm::push_pending_workspace(updated_vars.clone(), assigned_snapshot.clone());
        if self.verbose {
            debug!("HIR generated successfully");
        }

        let (single_assign_var, single_stmt_non_assign) = if hir.body.len() == 1 {
            match &hir.body[0] {
                runmat_hir::HirStmt::Assign(var_id, _, _, _) => (Some(var_id.0), false),
                _ => (None, true),
            }
        } else {
            (None, false)
        };

        bytecode.var_names = id_to_name.clone();
        if self.verbose {
            debug!(
                "Bytecode compiled: {} instructions",
                bytecode.instructions.len()
            );
        }

        #[cfg(not(target_arch = "wasm32"))]
        let fusion_snapshot = if self.emit_fusion_plan {
            build_fusion_snapshot(bytecode.accel_graph.as_ref(), &bytecode.fusion_groups)
        } else {
            None
        };
        #[cfg(target_arch = "wasm32")]
        let fusion_snapshot: Option<FusionPlanSnapshot> = None;

        // Prepare variable array with existing values before execution
        self.prepare_variable_array_for_execution(&bytecode, &updated_vars, debug_trace);

        if self.verbose {
            debug!(
                "Variable array after preparation: {:?}",
                self.variable_array
            );
            debug!("Updated variable mapping: {updated_vars:?}");
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
        let final_stmt_emit = last_displayable_statement_emit_disposition(&hir.body);

        if self.verbose {
            debug!("HIR body len: {}", hir.body.len());
            if !hir.body.is_empty() {
                debug!("HIR statement: {:?}", &hir.body[0]);
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
                    match jit_engine
                        .execute_or_compile_with_workspace(&bytecode, &mut self.variable_array)
                    {
                        Ok((_, actual_used_jit)) => {
                            used_jit = actual_used_jit;
                            execution_completed = true;
                            if actual_used_jit {
                                self.stats.jit_compiled += 1;
                            } else {
                                self.stats.interpreter_fallback += 1;
                            }
                            if let Some(var_id) = single_assign_var {
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
                    if hir.body.len() == 1 {
                        if let runmat_hir::HirStmt::Assign(var_id, _, _, _) = &hir.body[0] {
                            // For assignments, capture the assigned value for both display and type info
                            if var_id.0 < self.variable_array.len() {
                                let assignment_value = self.variable_array[var_id.0].clone();
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

        let last_assign_var = last_unsuppressed_assign_var(&hir.body);
        let last_expr_emits = last_expr_emits_value(&hir.body);
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
                if let Some(assigned_report) = runmat_vm::take_updated_workspace_assigned_report() {
                    assigned_this_execution.extend(
                        assigned_report
                            .ids
                            .iter()
                            .filter_map(|var_id| id_to_name.get(var_id).cloned()),
                    );
                    assigned_this_execution.extend(assigned_report.names);
                }
                if debug_trace {
                    debug!(
                        ?mutated_names,
                        ?assigned,
                        ?assigned_this_execution,
                        "[repl] mutated names and assigned return values"
                    );
                }
                self.variable_names = mutated_names.clone();
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
            let mut repl_source_id: Option<SourceId> = None;
            for (name, stmt) in &updated_functions {
                if matches!(stmt, runmat_hir::HirStmt::Function { .. }) {
                    let source_id = *repl_source_id
                        .get_or_insert_with(|| self.source_pool.intern("<repl>", input));
                    self.function_source_ids.insert(name.clone(), source_id);
                }
            }
            self.function_definitions = updated_functions;
            // Apply 'ans' update if applicable (persisting expression result)
            if let Some((var_id, value)) = ans_update {
                self.variable_names.insert("ans".to_string(), var_id);
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
            && matches!(final_stmt_emit, FinalStmtEmitDisposition::NeedsFallback)
        {
            if let Some(value) = result_value.as_ref() {
                let label = determine_display_label_from_context(
                    single_assign_var,
                    &id_to_name,
                    is_expression_stmt,
                    single_stmt_non_assign,
                );
                runmat_runtime::console::record_value_output(label.as_deref(), value);
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
            let source_map = if self.workspace_values.is_empty() {
                &self.variables
            } else {
                &self.workspace_values
            };
            if source_map.is_empty() {
                (workspace_updates, false)
            } else {
                let mut entries: Vec<WorkspaceEntry> = source_map
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

        self.format_mode = runmat_builtins::get_display_format();
        Ok(ExecutionResult {
            value: public_value,
            execution_time_ms,
            used_jit,
            error,
            type_info,
            streams,
            workspace: workspace_snapshot,
            figures_touched,
            warnings,
            profiling: gather_profiling(execution_time_ms),
            fusion_plan: fusion_snapshot,
            stdin_events,
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
}
