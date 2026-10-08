use super::*;
use std::rc::Rc;

#[cfg(not(target_arch = "wasm32"))]
fn entrypoint_target_function(
    assembly: &runmat_hir::HirAssembly,
) -> Option<runmat_hir::FunctionId> {
    assembly
        .entrypoints
        .first()
        .map(|entrypoint| entrypoint.target)
}

#[cfg(not(target_arch = "wasm32"))]
fn mir_local_fact_count_for_entrypoint(
    analysis: &runmat_mir::analysis::AnalysisStore,
    assembly: &runmat_hir::HirAssembly,
) -> usize {
    let function = entrypoint_target_function(assembly)
        .and_then(|function| u32::try_from(function.0).ok())
        .map(runmat_types::ProgramFunctionId);
    analysis.local_value_count(function)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
fn discover_source_catalog(
    source_name: Option<&str>,
) -> Option<runmat_package::DiscoveredSourceSymbols> {
    use runmat_package::discover_source_symbols_from_source_name;
    use std::path::{Path, PathBuf};

    let Ok(cwd) = runmat_filesystem::current_dir() else {
        let source_name = source_name?;
        let source_path = PathBuf::from(source_name);
        if source_path.is_absolute() {
            let source_cwd = source_path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("/"));
            return discover_source_symbols_from_source_name(source_name, &source_cwd)
                .ok()
                .flatten();
        }
        return None;
    };
    let source_name = source_name?;
    discover_source_symbols_from_source_name(source_name, &cwd)
        .ok()
        .flatten()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
fn discover_known_project_symbols(source_name: Option<&str>) -> HashSet<String> {
    discover_source_catalog(source_name)
        .map(|catalog| catalog.symbols)
        .unwrap_or_default()
}

impl RunMatSession {
    pub(super) fn configure_dynamic_function_services(&self) {
        let dynamic_environment = super::dynamic::DynamicFunctionEnvironment {
            compat: self.compat_mode,
            top_level_await_enabled: self.top_level_await_enabled,
            cache: Rc::clone(&self.dynamic_function_cache),
            project_handoff: self.project_handoff.clone(),
            source_evaluator: Rc::clone(&self.source_evaluator),
            #[cfg(not(target_arch = "wasm32"))]
            mex_runtime: std::rc::Rc::clone(&self.mex_runtime),
        };
        let loader: std::rc::Rc<runmat_runtime::user_functions::DynamicFunctionLoader> =
            std::rc::Rc::new(
                move |runtime, name, args, requested_outputs, phase, input_callsite| {
                    let environment = dynamic_environment.clone();
                    let args = match runmat_runtime::execution::RootedValues::new(args) {
                        Ok(args) => args,
                        Err(error) => {
                            return Box::pin(async move { Some(Err(error.into_runtime_error())) });
                        }
                    };
                    Box::pin(super::dynamic::load_dynamic_function(
                        runtime,
                        name,
                        args,
                        requested_outputs,
                        phase,
                        input_callsite,
                        environment,
                    ))
                },
            );
        self.runtime_context
            .set_dynamic_function_loader(Some(loader));

        let dynamic_function_cache = Rc::clone(&self.dynamic_function_cache);
        #[cfg(not(target_arch = "wasm32"))]
        let mex_runtime = std::rc::Rc::clone(&self.mex_runtime);
        let clearer: std::rc::Rc<runmat_runtime::user_functions::DynamicFunctionClearer> =
            std::rc::Rc::new(move |runtime, request| {
                let dynamic_function_cache = Rc::clone(&dynamic_function_cache);
                #[cfg(not(target_arch = "wasm32"))]
                let mex_runtime = std::rc::Rc::clone(&mex_runtime);
                Box::pin(async move {
                    if !matches!(
                        request,
                        runmat_runtime::user_functions::DynamicFunctionClearRequest::NativeExtensions
                    ) {
                        match &request {
                            runmat_runtime::user_functions::DynamicFunctionClearRequest::All => {
                                dynamic_function_cache.clear();
                            }
                            runmat_runtime::user_functions::DynamicFunctionClearRequest::Named(
                                name,
                            ) => {
                                dynamic_function_cache.clear_named(name);
                            }
                            runmat_runtime::user_functions::DynamicFunctionClearRequest::NativeExtensions => {}
                        }
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    mex_runtime
                        .clear_gracefully(&request, runtime.clone())
                        .await?;
                    // A successful typed function clear invalidates its
                    // selected session's persistent bindings. Do this after
                    // fallible module retirement so a failed clear cannot
                    // erase live function state.
                    let authority = runtime.session_variable_authority();
                    match &request {
                        runmat_runtime::user_functions::DynamicFunctionClearRequest::All => {
                            authority.clear_all_persistents()?;
                        }
                        runmat_runtime::user_functions::DynamicFunctionClearRequest::Named(name) => {
                            authority.clear_persistent_function(name)?;
                        }
                        runmat_runtime::user_functions::DynamicFunctionClearRequest::NativeExtensions => {}
                    }
                    Ok(())
                })
            });
        self.runtime_context
            .set_dynamic_function_clearer(Some(clearer));

        let refresh_cache = Rc::clone(&self.dynamic_function_cache);
        let refresher: std::rc::Rc<runmat_runtime::user_functions::DynamicFunctionRefresher> =
            std::rc::Rc::new(move |runtime, _request| {
                let cache = Rc::clone(&refresh_cache);
                Box::pin(async move {
                    // This refresh is one invocation-owned cache mutation.
                    // Yield/cancel before changing either revision or entries;
                    // the two mutations then form one short synchronous step.
                    runtime
                        .cpu_cooperation()
                        .checkpoint(1)
                        .await
                        .map_err(|source| source.into_runtime_error())?;
                    // RunMat has one live source cache, not a separate persisted
                    // matlabroot/toolbox cache. All four public requests refresh
                    // its lookup revision and compiled source entries; none
                    // unload a MEX module or clear workspace values.
                    if let Some(path) = runtime.search_path() {
                        path.refresh_generation().map_err(|source| {
                            build_runtime_error(format!("rehash: {source}"))
                                .with_builtin("rehash")
                                .with_source(source)
                                .build()
                        })?;
                    }
                    cache.clear();
                    Ok(())
                })
            });
        self.runtime_context
            .set_dynamic_function_refresher(Some(refresher));
    }

    async fn run(
        &mut self,
        input: &str,
    ) -> std::result::Result<crate::abi::ExecutionOutcome, RunError> {
        self.configure_runtime_context();
        let runtime = self.runtime_context.clone();
        runtime.scope(self.run_in_context(input)).await
    }

    async fn run_in_context(
        &mut self,
        input: &str,
    ) -> std::result::Result<crate::abi::ExecutionOutcome, RunError> {
        let _test_services = runmat_runtime::testing::install_test_services(
            crate::testing::runtime_adapter::services(
                self.compat_mode,
                self.project_handoff.clone(),
            ),
        );
        self.configure_dynamic_function_services();
        let source_lookup_name = self
            .current_source_fullpath_name()
            .unwrap_or_else(|| self.current_source_name());
        let companion = if let Some(handoff) = self.project_handoff.as_ref() {
            super::compile::companion_source_statements_from_handoff_async(
                source_lookup_name,
                self.compat_mode,
                handoff,
            )
            .await
        } else {
            super::compile::discover_companion_source_statements_async(
                source_lookup_name,
                self.compat_mode,
            )
            .await
        }
        .map_err(|error| {
            RunError::Runtime(
                build_runtime_error(format!("project composition failed: {error}"))
                    .with_identifier("RunMat:ProjectComposition")
                    .build(),
            )
        })?;
        self.pending_companion_source_discovery = Some(companion);
        let previous_workspace_bindings = self
            .logical_workspace_values()
            .map_err(RunError::Runtime)?
            .into_keys()
            .map(|name| {
                let key = self
                    .workspace_bindings
                    .get(&name)
                    .map(|binding| binding.key.clone())
                    .unwrap_or_else(|| self.workspace_binding_key(&name));
                (name, key)
            })
            .collect::<HashMap<_, _>>();
        let mut execution = self.execute_internal(input, true).await?;
        let workspace_names = execution
            .workspace_snapshot
            .values
            .iter()
            .map(|entry| entry.name.clone())
            .collect::<Vec<_>>();
        let workspace_full = execution.workspace_snapshot.full;
        let outcome = &mut execution.outcome;
        outcome.workspace_delta.upserts = self.abi_workspace_upserts(workspace_names)?;
        let (binding_removals, removed_names) =
            self.abi_workspace_removals(previous_workspace_bindings, workspace_full)?;
        outcome.workspace_delta.removals = binding_removals;
        if workspace_full && removed_names {
            outcome.effects.push(crate::abi::ObservedEffect::Workspace(
                crate::abi::WorkspaceEffectKind::Clear,
            ));
        }
        Ok(execution.outcome)
    }

    /// Execute a structured runtime/workspace ABI request.
    pub async fn execute_request(
        &mut self,
        request: crate::abi::ExecutionRequest,
    ) -> crate::abi::ExecutionResponse {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let local_tasks = self.local_tasks.clone();
            // The driver rejects admission before polling its child. Keep the
            // request outside that child until admission so an error can
            // report its source without copying source text on the hot path.
            let mut request = Some(request);
            return match local_tasks
                .participate_foreground(async {
                    self.execute_request_inner(
                        request.take().expect("execution request polled only once"),
                    )
                    .await
                })
                .await
            {
                Ok(response) => response,
                Err(error) => crate::abi::ExecutionResponse {
                    source_context: unresolved_source_context(
                        &request
                            .as_ref()
                            .expect("rejected foreground request was not polled")
                            .source,
                    ),
                    result: Err(RunError::Runtime(error)),
                },
            };
        }
        #[cfg(target_arch = "wasm32")]
        self.execute_request_inner(request).await
    }

    async fn execute_request_inner(
        &mut self,
        request: crate::abi::ExecutionRequest,
    ) -> crate::abi::ExecutionResponse {
        let _admission = self.admit_execution(None);
        let requested_outputs = request.requested_outputs;
        let source_input = request.source.clone();
        if let Err(error) = self.ensure_user_path_hydrated().await {
            return crate::abi::ExecutionResponse {
                source_context: unresolved_source_context(&source_input),
                result: Err(error),
            };
        }
        let source_resolution = match source_input_text(request.source).await {
            Ok(resolved) => resolved,
            Err(err) => {
                return crate::abi::ExecutionResponse {
                    source_context: unresolved_source_context(&source_input),
                    result: Err(err),
                };
            }
        };
        let source_name = source_resolution.display_name;
        let source_fullpath_name = source_resolution.fullpath_name;
        let source_text = source_resolution.text;
        let source_identity = resolve_source_identity(&source_input, &source_text);
        let previous_compat = self.compat_mode;
        let previous_top_level_await_enabled = self.top_level_await_enabled;
        let previous_dynamic_eval_enabled = self.dynamic_eval_enabled;
        let previous_source_name = self.active_source_name.clone();
        let previous_source_fullpath_name = self.active_source_fullpath_name.clone();
        let previous_workspace_handle = self.abi_workspace_handle;
        let previous_source_identity = self.active_source_identity.clone();

        self.compat_mode = request.compatibility;
        self.top_level_await_enabled = request.host_policy.top_level_await;
        self.dynamic_eval_enabled = request.host_policy.dynamic_eval;
        self.active_source_name = source_name.clone();
        self.active_source_fullpath_name = source_fullpath_name.clone();
        self.abi_workspace_handle = request.workspace;
        self.active_source_identity = source_identity.clone();

        let result = self.run(&source_text).await;

        self.compat_mode = previous_compat;
        self.top_level_await_enabled = previous_top_level_await_enabled;
        self.dynamic_eval_enabled = previous_dynamic_eval_enabled;
        self.active_source_name = previous_source_name;
        self.active_source_fullpath_name = previous_source_fullpath_name;
        self.abi_workspace_handle = previous_workspace_handle;
        self.active_source_identity = previous_source_identity;
        self.pending_companion_source_discovery = None;

        crate::abi::ExecutionResponse {
            source_context: crate::abi::ExecutionSourceContext {
                name: source_name,
                text: Some(source_text),
                identity: source_identity,
            },
            result: result
                .and_then(|outcome| apply_requested_output_policy(outcome, &requested_outputs)),
        }
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
        runmat_vm::set_dynamic_eval_options(
            self.compat_mode,
            self.compat_mode.allows_runmat_extensions(),
            self.top_level_await_enabled,
            self.dynamic_eval_enabled,
        );
        runmat_hir::set_error_namespace(&self.error_namespace);
        let exec_span = info_span!(
            "runtime.execute",
            input_len = input.len(),
            verbose = self.verbose
        );
        let _exec_guard = exec_span.enter();
        runmat_runtime::console::reset_thread_buffer();
        runmat_runtime::console::record_diary_command(input);
        runmat_runtime::plotting_hooks::reset_recent_figures();
        runmat_runtime::warning_store::reset();
        runmat_value::set_display_format(self.format_mode);
        reset_provider_telemetry();
        let _interrupt_guard =
            runmat_runtime::interrupt::replace_interrupt(Some(self.runtime_context.cancellation()));
        let start_time = Instant::now();
        self.stats.total_executions += 1;
        let debug_trace = std::env::var("RUNMAT_DEBUG_REPL").is_ok();
        let (_async_input_guard, stdin_events) = self.install_execution_input();

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
        // small for two nested interpret() invocations. We cannot move the inner
        // evaluation to another thread because `Value` can carry thread-confined
        // GC handles, so the native path grows the stack around each poll of the
        // inner eval future. To avoid re-entering async-yielding prompt code
        // recursively, native prompt eval deliberately disables top-level await.
        let compat = self.compat_mode;
        #[cfg(not(target_arch = "wasm32"))]
        let dynamic_eval_enabled = self.dynamic_eval_enabled;
        #[cfg(target_arch = "wasm32")]
        let top_level_await_enabled = self.top_level_await_enabled;
        let source_catalog_for_eval_hook = Arc::new(
            self.pending_companion_source_discovery
                .as_ref()
                .and_then(|companion| companion.source_catalog.clone()),
        );
        let _eval_hook_guard = runmat_runtime::interaction::replace_eval_hook(Some(
            std::sync::Arc::new(
                move |expr: String| -> runmat_runtime::interaction::EvalHookFuture {
                    // Shared eval logic, used by both the WASM async path and the
                    // native thread path below.
                    async fn eval_expr(
                        expr: String,
                        compat: runmat_parser::CompatMode,
                        top_level_await_enabled: bool,
                        source_catalog: Arc<Option<runmat_package::DiscoveredSourceSymbols>>,
                    ) -> Result<Value, RuntimeError> {
                        let wrapped = format!("__runmat_input_result__ = ({expr});");
                        let ast = parse_with_options(&wrapped, ParserOptions::new(compat))
                            .map_err(|e| {
                                build_runtime_error(format!("input: parse error: {e}"))
                                    .with_identifier("RunMat:input:ParseError")
                                    .with_failure_origin(runmat_runtime::runtime_error::RuntimeFailureOrigin::LanguageEvaluation)
                                    .build()
                            })?;
                        let known_project_symbols = source_catalog
                            .as_ref()
                            .as_ref()
                            .map(|catalog| &catalog.symbols)
                            .cloned()
                            .unwrap_or_default();
                        let frontend =
                            runmat_static_analysis::frontend::analyze_program_with_catalog(
                                &ast,
                                &LoweringContext::new(&HashMap::new())
                                    .with_known_project_symbols(&known_project_symbols)
                                    .with_runmat_extensions_enabled(
                                        compat.allows_runmat_extensions(),
                                    )
                                    .with_top_level_await_enabled(top_level_await_enabled),
                                source_catalog.as_ref().as_ref(),
                            );
                        if let Some(e) = frontend.lowering_failure {
                            return Err(build_runtime_error(format!("input: lowering error: {e}"))
                                .with_identifier("RunMat:input:LowerError")
                                .with_failure_origin(runmat_runtime::runtime_error::RuntimeFailureOrigin::LanguageEvaluation)
                                .build());
                        }
                        if let Some(e) = frontend.compile_failure {
                            return Err(RuntimeError::from(e));
                        }
                        let bc = frontend.bytecode.ok_or_else(|| {
                            build_runtime_error("input: canonical frontend produced no bytecode")
                                .with_identifier("RunMat:input:CompileError")
                                .build()
                        })?;
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

                    let source_catalog = Arc::clone(&source_catalog_for_eval_hook);
                    #[cfg(target_arch = "wasm32")]
                    {
                        Box::pin(async move {
                            runmat_runtime::interaction::EvalHookOutcome::from_evaluation(
                                eval_expr(expr, compat, top_level_await_enabled, source_catalog)
                                    .await,
                            )
                        })
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        const INPUT_EVAL_STACK_BYTES: usize = 16 * 1024 * 1024;
                        let mut eval_future =
                            Box::pin(eval_expr(expr, compat, false, source_catalog));
                        Box::pin(futures::future::poll_fn(move |cx| {
                            stacker::grow(INPUT_EVAL_STACK_BYTES, || {
                                let _dynamic_eval_guard = runmat_vm::push_dynamic_eval_options(
                                    compat,
                                    compat.allows_runmat_extensions(),
                                    false,
                                    dynamic_eval_enabled,
                                );
                                eval_future.as_mut().poll(cx).map(
                                    runmat_runtime::interaction::EvalHookOutcome::from_evaluation,
                                )
                            })
                        }))
                    }
                },
            ),
        ));

        if self.verbose {
            debug!("Executing: {}", input.trim());
        }

        let source_name_for_context = self.current_source_name().to_string();
        let _fallback_source_guard = runmat_runtime::source_context::replace_current_source_context(
            Some(&source_name_for_context),
            Some(input),
        );

        let source = self.canonical_interactive_source(input)?;
        let mut prepared = self.compile_input(input)?;
        let revision = crate::ExecutableRevision::derive(
            &source,
            self.runtime_context.program_revision().cloned(),
            crate::program_environment(self.compat_mode),
        );
        let source_map = prepared
            .portable_source_map
            .take()
            .expect("the canonical frontend always retains its program source table");
        prepared
            .attach_function_origins(&revision.program_revision, &source_map)
            .map_err(|error| {
                RunError::Runtime(
                    build_runtime_error(error.to_string())
                        .with_identifier("RunMat:ExecutableFunctionIdentity")
                        .with_source(error)
                        .build(),
                )
            })?;
        let PreparedExecution {
            ast,
            lowering,
            mir,
            analysis,
            bytecode,
            portable_source_map: _,
            mut function_registry_after_success,
            current_function_ids,
            next_semantic_function_id_after_success,
        } = prepared;
        let interactive_unit = {
            crate::ExecutableUnit::new(
                source.clone(),
                revision,
                source_map,
                mir,
                analysis,
                bytecode,
            )
            .map_err(|error| {
                RunError::Runtime(
                    build_runtime_error(error)
                        .with_identifier("RunMat:ExecutableProduct")
                        .build(),
                )
            })?
        };
        #[cfg(not(target_arch = "wasm32"))]
        let analysis = interactive_unit.analysis();
        let execution_runtime =
            interactive_unit
                .bind_runtime_sources(self.runtime_context.clone().with_program_revision(Some(
                    interactive_unit.revision().program_revision.clone(),
                )))
                .map_err(|error| {
                    RunError::Runtime(
                        build_runtime_error(error)
                            .with_identifier("RunMat:ExecutableSourceIdentity")
                            .build(),
                    )
                })?;
        let mut bytecode = interactive_unit
            .runtime_bytecode(&execution_runtime)
            .map_err(|error| {
                RunError::Runtime(
                    build_runtime_error(error)
                        .with_identifier("RunMat:ExecutableSourceIdentity")
                        .build(),
                )
            })?;
        let active_registry = bytecode.function_registry();
        super::semantic_definition::finalize_current_definitions(
            &mut function_registry_after_success,
            &current_function_ids,
            &active_registry,
        )
        .map_err(|error| {
            RunError::Runtime(
                build_runtime_error(error)
                    .with_identifier("RunMat:ExecutableFunctionIdentity")
                    .build(),
            )
        })?;
        // Only this compilation's definitions need a new artifact. Inherited
        // session definitions already retain their own indexed envelopes.
        let has_current_artifact_origin = current_function_ids.iter().any(|function| {
            active_registry
                .get(*function)
                .is_some_and(|body| body.binding_origin.is_some())
        });
        let callable_artifact = if has_current_artifact_origin {
            Some(
                (if interactive_unit.bytecode().instructions.is_empty() {
                    let primary_source = lowering.assembly.primary_source.ok_or_else(|| {
                        RunError::Runtime(
                            build_runtime_error("definition-only executable has no primary source")
                                .with_identifier("RunMat:CallableArtifact:ExecutableProduct")
                                .build(),
                        )
                    })?;
                    interactive_unit.portable_envelope_for_current_definition(
                        &current_function_ids,
                        primary_source,
                    )
                } else {
                    interactive_unit.portable_envelope()
                })
                .and_then(|envelope| {
                    envelope
                        .canonical_bytes()
                        .map_err(|error| error.to_string())
                })
                .map_err(|error| {
                    RunError::Runtime(
                        build_runtime_error(error)
                            .with_identifier("RunMat:CallableArtifact:ExecutableProduct")
                            .build(),
                    )
                })?,
            )
        } else {
            None
        };
        let source_registry_publication = self
            .source_evaluator
            .publish_for_execution(active_registry, callable_artifact)
            .map_err(RunError::Runtime)?;
        let _source_id_guard = runmat_runtime::source_context::replace_current_source_id_in_context(
            &execution_runtime,
            bytecode
                .source_id
                .and_then(|source| source.runtime())
                .and_then(|source| source.resolve_in(&execution_runtime)),
        );
        if self.verbose {
            debug!("AST: {ast:?}");
        }
        let display = execution_display_context(&lowering.assembly, bytecode.layout.as_ref());
        let display_context = display.context;
        let display_var_ids = display.display_var_ids;
        let stmt_count = entry_statement_count(&lowering.assembly);
        let execution_vars = execution_workspace_mapping(&bytecode);
        #[cfg(not(target_arch = "wasm32"))]
        let mut execution_vars = execution_vars;
        let max_var_id = execution_vars.values().copied().max().unwrap_or(0);
        if debug_trace {
            debug!(?execution_vars, "[repl] execution vars");
        }
        let workspace_projection = self
            .base_workspace
            .begin_projection()
            .map_err(runmat_runtime::workspace::base_workspace_error_to_runtime)
            .map_err(RunError::Runtime)?;
        let workspace_projection = std::rc::Rc::new(
            runmat_runtime::workspace::BaseWorkspaceProjectionService::new(workspace_projection),
        );
        if debug_trace {
            let workspace_values_before = workspace_projection
                .snapshot()
                .map_err(RunError::Runtime)?
                .into_iter()
                .map(|(name, value)| (name.to_string(), value.clone()))
                .collect::<HashMap<_, _>>();
            debug!(
                ?workspace_values_before,
                "[repl] workspace snapshot before execution"
            );
        }
        let id_to_name: HashMap<usize, String> = execution_vars
            .iter()
            .map(|(name, var_id)| (*var_id, name.clone()))
            .collect();
        let global_aliases = runmat_runtime::workspace::bindings::FrameGlobalBindings::default();
        let inherited_global_names = self
            .session_global_binding_names()
            .map_err(RunError::Runtime)?;
        let mut assigned_snapshot = HashSet::new();
        for (name, slot) in &execution_vars {
            if inherited_global_names.contains(name) {
                global_aliases.declare_named(*slot, name.clone());
                if self
                    .runtime_context
                    .session_global_value(name)
                    .map_err(RunError::Runtime)?
                    .is_some()
                {
                    assigned_snapshot.insert(name.clone());
                }
            } else if workspace_projection
                .lookup(name)
                .map_err(RunError::Runtime)?
                .is_some()
            {
                assigned_snapshot.insert(name.clone());
            }
        }
        if debug_trace {
            debug!(?assigned_snapshot, "[repl] assigned snapshot");
        }
        let mut workspace_admission = Some(runmat_vm::workspace_admission_with_global_bindings(
            execution_vars.clone(),
            assigned_snapshot.clone(),
            global_aliases,
        ));
        if self.verbose {
            debug!("HIR generated successfully");
        }

        if preserve_layout_var_names && bytecode.layout.is_some() {
            for (slot, name) in &id_to_name {
                bytecode.var_names.insert(*slot, name.clone());
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
            let runtime_groups = bytecode.runtime_fusion_groups();
            let (runtime_graph, runtime_graph_source) =
                bytecode.runtime_accel_graph_for_fusion_with_source(&runtime_groups);
            build_fusion_snapshot(
                &runtime_groups,
                &bytecode.fusion_metadata.mir_fusion_candidate_groups,
                &bytecode.fusion_metadata.instruction_windows,
                Some(crate::fusion::FusionPlannerMetadata {
                    source: "semantic-mir-analysis-runtime".to_string(),
                    accel_graph_state: if runtime_graph.is_some() {
                        "present".to_string()
                    } else {
                        "missing".to_string()
                    },
                    accel_graph_source: runtime_graph_source.as_str().to_string(),
                    mir_local_fact_count: mir_local_fact_count_for_entrypoint(
                        analysis,
                        &lowering.assembly,
                    ),
                    mir_diagnostic_count: analysis.diagnostics.len(),
                    mir_fusion_signal_count: bytecode.fusion_metadata.mir_fusion_signal_count,
                    mir_fusion_candidate_group_count: bytecode
                        .fusion_metadata
                        .mir_fusion_candidate_group_count,
                    mir_semantic_instruction_window_count: bytecode
                        .fusion_metadata
                        .instruction_window_count,
                }),
            )
        } else {
            None
        };
        #[cfg(target_arch = "wasm32")]
        let fusion_snapshot: Option<FusionPlanSnapshot> = None;

        // Prepare variable array with existing values before execution
        let mut variable_array = self
            .prepare_variable_array_for_execution(
                &bytecode,
                &execution_vars,
                &workspace_projection,
                debug_trace,
            )
            .map_err(RunError::Runtime)?;

        if self.verbose {
            debug!("Variable array after preparation: {:?}", variable_array);
            debug!("Bytecode instructions: {:?}", bytecode.instructions);
        }

        #[cfg(not(target_arch = "wasm32"))]
        let mut used_jit = false;
        #[cfg(target_arch = "wasm32")]
        let used_jit = false;
        #[cfg(not(target_arch = "wasm32"))]
        let mut execution_completed = false;
        #[cfg(target_arch = "wasm32")]
        let execution_completed = false;
        let mut result_value: Option<Value> = None; // Always start fresh for each execution
        let mut suppressed_value: Option<Value> = None; // Track value for type info when suppressed
        let mut error = None;
        let mut workspace_updates: Vec<WorkspaceEntry> = Vec::new();
        let mut workspace_snapshot_force_full = false;
        let mut completed_global_names = HashSet::new();
        let mut vm_workspace_completion = None;
        let mut ans_update: Option<(usize, Value)> = None;
        #[cfg(not(target_arch = "wasm32"))]
        let tiered_interactive = self
            .prepare_tiered_interactive(&interactive_unit, &workspace_projection)
            .map_err(RunError::Runtime)?;
        #[cfg(not(target_arch = "wasm32"))]
        let mut native_loop_backedges = std::collections::BTreeMap::new();

        let execution_runtime = execution_runtime.clone().with_service_ports(
            execution_runtime
                .service_ports()
                .clone()
                .with_workspace(workspace_projection.clone()),
        );

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
            debug!("Semantic entry body len: {stmt_count}");
            if let Some(stmt) = first_entry_statement(&lowering.assembly) {
                debug!("Semantic HIR statement: {stmt:?}");
            }
            debug!("is_semicolon_suppressed: {is_semicolon_suppressed}");
        }

        #[cfg(not(target_arch = "wasm32"))]
        if let Some((workspace, profile)) = tiered_interactive.as_ref() {
            match self
                .invoke_tiered_interactive(
                    &interactive_unit,
                    &execution_runtime,
                    &bytecode.function_registry,
                    workspace,
                    profile.as_ref(),
                )
                .await
            {
                Ok(super::executable::TieredInteractiveInvocation::Native(Ok(native))) => {
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
                    let snapshot = native.workspace.ok_or_else(|| {
                        RunError::Runtime(
                            build_runtime_error(
                                "interactive native execution produced no workspace snapshot",
                            )
                            .with_identifier("RunMat:NativeWorkspace")
                            .build(),
                        )
                    })?;
                    completed_global_names = snapshot.global_bindings.into_iter().collect();
                    execution_vars.retain(|name, _| snapshot.values.contains_key(name));
                    for (name, value) in snapshot.values {
                        let slot = execution_vars.get(&name).copied().unwrap_or_else(|| {
                            let slot = variable_array.len();
                            execution_vars.insert(name.clone(), slot);
                            slot
                        });
                        if variable_array.len() <= slot {
                            variable_array.resize(slot + 1, Value::Num(0.0));
                        }
                        variable_array[slot] = value;
                    }
                    used_jit = true;
                    execution_completed = true;
                    if !display_context.single_stmt_non_assign {
                        if let Some(var_id) = display_context.first_assign_var {
                            if let Some(assignment_value) = variable_array.get(var_id).cloned() {
                                if is_semicolon_suppressed {
                                    suppressed_value = Some(assignment_value);
                                } else {
                                    result_value = Some(assignment_value);
                                }
                            }
                        }
                    }
                }
                Ok(super::executable::TieredInteractiveInvocation::Cold) => {}
                Ok(super::executable::TieredInteractiveInvocation::Native(Err(native_error))) => {
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

        // Use interpreter if JIT failed or is disabled
        if !execution_completed {
            if self.verbose {
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

                if self.verbose {
                    debug!(
                        "Modified expression bytecode, new instructions: {:?}",
                        execution_bytecode.instructions
                    );
                }
            }

            let execution = self
                .interpret_with_context(
                    &execution_bytecode,
                    &mut variable_array,
                    execution_runtime.clone(),
                    workspace_admission.take(),
                )
                .await;
            vm_workspace_completion = execution.workspace;
            match execution.outcome {
                Ok(runmat_vm::InterpreterOutcome::Completed(completion)) => {
                    let results = completion.values;
                    self.stats.interpreter_fallback += 1;
                    if self.verbose {
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
                                        if self.verbose {
                                            debug!(
                                                "Interpreter assignment result: {result_value:?}"
                                            );
                                        }
                                    } else {
                                        suppressed_value = Some(assignment_value);
                                        if self.verbose {
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
                        if temp_var_id < variable_array.len() {
                            let expression_value = variable_array[temp_var_id].clone();
                            if !is_semicolon_suppressed {
                                // Capture for 'ans' update when output is not suppressed
                                ans_update = Some((temp_var_id, expression_value.clone()));
                                result_value = Some(expression_value);
                                if self.verbose {
                                    debug!(
                                        "Expression result from temp var {temp_var_id}: {result_value:?}"
                                    );
                                }
                            } else {
                                suppressed_value = Some(expression_value);
                                if self.verbose {
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
            let can_emit_from_context = !display_var_ids.is_empty() || last_expr_emits;
            if can_emit_from_context {
                if let Some(value) = runmat_runtime::console::take_last_value_output() {
                    result_value = Some(value);
                }
                if result_value.is_none() {
                    if let Some(var_id) = last_store_var_index(&bytecode) {
                        if var_id < variable_array.len() {
                            result_value = Some(variable_array[var_id].clone());
                        }
                    }
                    if result_value.is_none() {
                        if let Some(var_id) = last_assign_var {
                            if var_id < variable_array.len() {
                                result_value = Some(variable_array[var_id].clone());
                            }
                        }
                    }
                    if result_value.is_none() {
                        if let Some(var_id) = last_emit_var_index(&bytecode) {
                            if var_id < variable_array.len() {
                                result_value = Some(variable_array[var_id].clone());
                            }
                        }
                    }
                }
            }
        }

        // Executor-visible workspace mutations are admitted in source order.
        // Publish them even when a later statement fails so an ordinary script
        // preserves MATLAB workspace semantics without replacing concurrent
        // canonical state from an end-of-run snapshot.
        if workspace_projection
            .has_pending_changes()
            .map_err(RunError::Runtime)?
        {
            workspace_projection.publish().map_err(RunError::Runtime)?;
        }
        let published_changes = workspace_projection
            .take_published_changes()
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

        #[cfg(not(target_arch = "wasm32"))]
        if used_jit && is_expression_stmt {
            if let Some(value) = result_value.clone() {
                let slot = execution_vars.get("ans").copied().unwrap_or_else(|| {
                    let slot = variable_array.len();
                    execution_vars.insert("ans".to_string(), slot);
                    slot
                });
                if variable_array.len() <= slot {
                    variable_array.resize(slot + 1, Value::Num(0.0));
                }
                variable_array[slot] = value.clone();
                ans_update = Some((slot, value));
            }
        }

        let execution_time = start_time.elapsed();
        #[cfg(not(target_arch = "wasm32"))]
        if let Some((_, profile)) = tiered_interactive.as_ref() {
            self.observe_tiered_interactive(
                &interactive_unit,
                profile.as_ref(),
                execution_time,
                &native_loop_backedges,
            );
        }
        let execution_time_ms = execution_time.as_millis() as u64;

        self.stats.total_execution_time_ms += execution_time_ms;
        self.stats.average_execution_time_ms =
            self.stats.total_execution_time_ms as f64 / self.stats.total_executions as f64;

        let vm_workspace_state = (!used_jit)
            .then(|| vm_workspace_completion.map(|completion| completion.state))
            .flatten();
        let mut completed_workspace = if let Some(workspace_state) = vm_workspace_state {
            completed_global_names = workspace_state
                .global_aliases
                .values()
                .filter_map(|binding| match binding {
                    runmat_runtime::workspace::bindings::FrameGlobalBinding::Named(name) => {
                        Some(name.clone())
                    }
                    runmat_runtime::workspace::bindings::FrameGlobalBinding::LegacySlot => None,
                })
                .collect();
            let mutated_names = workspace_state.names;
            let assigned = workspace_state.assigned;
            if debug_trace {
                debug!(
                    ?mutated_names,
                    ?assigned,
                    "[repl] mutated names and assigned return values"
                );
            }
            Some(
                assigned
                    .into_iter()
                    .filter_map(|name| {
                        let slot = mutated_names.get(&name).copied()?;
                        let value = workspace_state.vars.get(slot)?.clone();
                        Some((name, (slot, value)))
                    })
                    .collect::<HashMap<_, _>>(),
            )
        } else if used_jit && error.is_none() {
            Some(
                execution_vars
                    .iter()
                    .filter_map(|(name, slot)| {
                        variable_array
                            .get(*slot)
                            .cloned()
                            .map(|value| (name.clone(), (*slot, value)))
                    })
                    .collect::<HashMap<_, _>>(),
            )
        } else {
            None
        };

        // An unsuppressed expression updates `ans` only after successful
        // execution. This is a language-visible host write, so admit it
        // explicitly instead of inferring it from the completed register file.
        if error.is_none() {
            if let (Some(workspace), Some((var_id, value))) =
                (completed_workspace.as_mut(), ans_update)
            {
                runmat_runtime::execution::validate_storable_value(&value)
                    .map_err(RunError::Runtime)?;
                workspace_projection
                    .assign("ans", value.clone())
                    .map_err(RunError::Runtime)?;
                workspace_projection.publish().map_err(RunError::Runtime)?;
                let published_changes = workspace_projection
                    .take_published_changes()
                    .map_err(RunError::Runtime)?;
                for change in published_changes.changes() {
                    if let (Some(name), Some(value)) = (change.name(), change.value()) {
                        workspace_updates.push(workspace_entry(name, value));
                    }
                }
                workspace.insert("ans".to_string(), (var_id, value));
            }
        }

        if let Some(completed_workspace) = completed_workspace {
            // The executor has already admitted every language-visible
            // mutation in source order. The completed register file now only
            // refreshes VM slot metadata; it is never a second workspace
            // authority and must not overwrite a newer canonical value.
            self.refresh_workspace_binding_slots(&completed_workspace, &completed_global_names);
        }
        let durable_global_names = self
            .session_global_binding_names()
            .map_err(RunError::Runtime)?;
        self.reconcile_workspace_global_binding_keys(&execution_vars, &durable_global_names)
            .map_err(RunError::Runtime)?;

        // Function-definition publication remains transactional with a
        // successful language execution; value assignments above deliberately
        // survive a later runtime error.
        if error.is_none() {
            #[cfg(not(target_arch = "wasm32"))]
            self.generic_native_cache
                .publish_session_registry(&self.function_registry, &function_registry_after_success)
                .map_err(RunError::Runtime)?;
            self.function_registry = function_registry_after_success;
            self.next_semantic_function_id = next_semantic_function_id_after_success;
            source_registry_publication.finish(self.function_registry.clone());
        } else {
            drop(source_registry_publication);
        }

        if self.verbose {
            debug!("Execution completed in {execution_time_ms}ms (JIT: {used_jit})");
        }

        if !is_expression_stmt
            && !is_semicolon_suppressed
            && last_assign_var.is_some()
            && !display_context.single_stmt_non_assign
            && !display_var_ids.is_empty()
        {
            if let Some(var_id) = last_store_var_index(&bytecode) {
                if var_id < variable_array.len() {
                    result_value = Some(variable_array[var_id].clone());
                }
            } else if matches!(final_stmt_emit, FinalStmtEmitDisposition::NeedsFallback)
                && result_value.is_none()
            {
                if let Some(v) = variable_array
                    .iter()
                    .rev()
                    .find(|v| !matches!(v, Value::Num(0.0)))
                    .cloned()
                {
                    result_value = Some(v);
                }
            }
        }

        if !is_semicolon_suppressed
            && (!display_var_ids.is_empty()
                || matches!(final_stmt_emit, FinalStmtEmitDisposition::NeedsFallback)
                || display_context.single_assign_var.is_some()
                || (is_expression_stmt
                    && matches!(final_stmt_emit, FinalStmtEmitDisposition::Inline)))
            && runmat_runtime::console::take_last_value_output().is_none()
        {
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
                        (id_to_name.get(&var_id), variable_array.get(var_id))
                    {
                        runmat_runtime::console::record_value_output(
                            Some(label.as_str()),
                            display_value,
                        );
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
        let canonical_workspace = self.logical_workspace_values().map_err(RunError::Runtime)?;
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
        let workspace_snapshot = self
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
                    build_runtime_error(format!(
                        "diary: failed to write diary file ({diary_error})"
                    ))
                    .with_identifier("RunMat:diary:IO")
                    .with_source(diary_error)
                    .build(),
                );
            }
        }

        if let Some(runtime_error) = &mut error {
            self.normalize_error_namespace(runtime_error);
            self.populate_callstack(runtime_error, &execution_runtime);
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

        let profiling = gather_profiling(execution_time_ms);
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
            profiling,
        };

        self.format_mode = runmat_value::get_display_format();
        Ok(SessionExecution {
            outcome,
            workspace_snapshot,
        })
    }

    /// Interpret bytecode with persistent variable context
    async fn interpret_with_context(
        &mut self,
        bytecode: &runmat_vm::Bytecode,
        variable_array: &mut Vec<Value>,
        runtime: runmat_runtime::context::RuntimeContext,
        admission: Option<runmat_vm::PendingWorkspaceAdmission>,
    ) -> runmat_vm::WorkspaceInterpreterOutcome {
        let source_name = self.current_source_name().to_string();
        runmat_vm::interpret_with_workspace_in_context(
            bytecode,
            variable_array,
            Some(source_name.as_str()),
            runtime,
            admission,
        )
        .await
    }

    fn abi_workspace_upserts(
        &self,
        workspace_names: Vec<String>,
    ) -> std::result::Result<Vec<crate::abi::WorkspaceBindingValue>, RunError> {
        let workspace = self.logical_workspace_values().map_err(RunError::Runtime)?;
        let mut workspace_names = workspace_names;
        workspace_names.sort();
        workspace_names.dedup();
        Ok(workspace_names
            .into_iter()
            .filter_map(|name| {
                let value = workspace.get(&name)?.clone();
                let binding = runmat_hir::BindingName(name);
                let key = self
                    .workspace_bindings
                    .get(&binding.0)
                    .map(|binding| binding.key.clone())
                    .unwrap_or_else(|| self.workspace_binding_key(&binding.0));
                Some(crate::abi::WorkspaceBindingValue { key, value })
            })
            .collect())
    }

    fn abi_workspace_removals(
        &self,
        previous_workspace_bindings: HashMap<String, crate::abi::WorkspaceBindingKey>,
        include_missing_names: bool,
    ) -> std::result::Result<(Vec<crate::abi::WorkspaceBindingKey>, bool), RunError> {
        let workspace = self.logical_workspace_values().map_err(RunError::Runtime)?;
        let mut removed_names = false;
        let mut removed = previous_workspace_bindings
            .into_iter()
            .filter(|(name, previous_key)| {
                if !workspace.contains_key(name) {
                    removed_names |= include_missing_names;
                    return include_missing_names;
                }
                self.workspace_bindings
                    .get(name)
                    .map(|binding| &binding.key)
                    .is_some_and(|current_key| current_key != previous_key)
            })
            .collect::<Vec<_>>();
        removed.sort_by(|left, right| left.0.cmp(&right.0));
        Ok((
            removed.into_iter().map(|(_, key)| key).collect(),
            removed_names,
        ))
    }
}

fn apply_requested_output_policy(
    mut outcome: crate::abi::ExecutionOutcome,
    requested_outputs: &runmat_hir::RequestedOutputCount,
) -> Result<crate::abi::ExecutionOutcome, RunError> {
    use crate::abi::RuntimeFlow;
    use runmat_hir::RequestedOutputCount;
    use runmat_value::{ValueSequence, ValueSequenceKind};

    let flow = match outcome.flow {
        RuntimeFlow::DynamicList(handle) => {
            outcome.flow = match requested_outputs {
                RequestedOutputCount::Zero | RequestedOutputCount::Exactly(0) => {
                    RuntimeFlow::NoValue
                }
                _ => RuntimeFlow::DynamicList(handle),
            };
            return Ok(outcome);
        }
        RuntimeFlow::NoValue => Ok(ValueSequence::empty()),
        RuntimeFlow::Single(value) => ValueSequence::single(value),
        RuntimeFlow::OutputList(values) => ValueSequence::comma_separated(values),
    }
    .map_err(runmat_runtime::sequence::sequence_error_to_runtime)
    .map_err(RunError::Runtime)?;

    outcome.flow = match requested_outputs {
        RequestedOutputCount::Zero => RuntimeFlow::NoValue,
        RequestedOutputCount::One => first_runtime_flow_value(flow),
        RequestedOutputCount::Exactly(count) => {
            if *count == 0 {
                RuntimeFlow::NoValue
            } else if *count == 1 {
                first_runtime_flow_value(flow)
            } else {
                let mut values = flow.into_values();
                values.truncate(*count);
                RuntimeFlow::OutputList(values)
            }
        }
        RequestedOutputCount::CurrentFunctionNargout
        | RequestedOutputCount::DestinationSequenceCardinality => match flow.kind() {
            ValueSequenceKind::Single => first_runtime_flow_value(flow),
            ValueSequenceKind::CommaSeparated => RuntimeFlow::OutputList(flow.into_values()),
        },
    };
    Ok(outcome)
}

fn first_runtime_flow_value(sequence: runmat_value::ValueSequence) -> crate::abi::RuntimeFlow {
    sequence
        .into_values()
        .into_iter()
        .next()
        .map(crate::abi::RuntimeFlow::Single)
        .unwrap_or(crate::abi::RuntimeFlow::NoValue)
}

fn resolve_source_identity(
    source: &crate::abi::SourceInput,
    source_text: &str,
) -> Option<crate::abi::SourceIdentity> {
    match source {
        crate::abi::SourceInput::Path(path) => {
            Some(crate::abi::SourceIdentity::PathAndContentHash {
                path: path.clone(),
                hash: source_text_hash(source_text),
            })
        }
        crate::abi::SourceInput::Text { name, .. } => {
            if name.starts_with('<') {
                None
            } else {
                Some(crate::abi::SourceIdentity::PathAndContentHash {
                    path: name.clone(),
                    hash: source_text_hash(source_text),
                })
            }
        }
    }
}

fn source_text_hash(source_text: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source_text.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn unresolved_source_context(
    source: &crate::abi::SourceInput,
) -> crate::abi::ExecutionSourceContext {
    let name = match source {
        crate::abi::SourceInput::Path(path) => path.clone(),
        crate::abi::SourceInput::Text { name, .. } => name.clone(),
    };
    crate::abi::ExecutionSourceContext {
        name,
        text: match source {
            crate::abi::SourceInput::Text { text, .. } => Some(text.clone()),
            crate::abi::SourceInput::Path(_) => None,
        },
        identity: None,
    }
}

fn runtime_error_span(error: &runmat_runtime::RuntimeError) -> Option<runmat_hir::Span> {
    error.span.as_ref().map(|span| {
        let start = span.offset();
        runmat_hir::Span {
            start,
            end: start + span.len().max(1),
        }
    })
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

fn entry_function(assembly: &runmat_hir::HirAssembly) -> Option<&runmat_hir::HirFunction> {
    let entrypoint = assembly.entrypoints.first()?;
    assembly
        .functions
        .iter()
        .find(|function| function.id == entrypoint.target)
}

fn entry_statement_count(assembly: &runmat_hir::HirAssembly) -> usize {
    entry_function(assembly)
        .map(|function| function.body.statements.len())
        .unwrap_or(0)
}

fn first_entry_statement(assembly: &runmat_hir::HirAssembly) -> Option<&runmat_hir::HirStmt> {
    entry_function(assembly)?.body.statements.first()
}

struct SessionExecution {
    outcome: crate::abi::ExecutionOutcome,
    workspace_snapshot: WorkspaceSnapshot,
}

#[derive(Debug)]
struct ResolvedSourceInput {
    display_name: String,
    fullpath_name: Option<String>,
    text: String,
}

async fn source_input_text(
    source: crate::abi::SourceInput,
) -> std::result::Result<ResolvedSourceInput, RunError> {
    match source {
        crate::abi::SourceInput::Text { name, text } => Ok(ResolvedSourceInput {
            display_name: name,
            fullpath_name: None,
            text,
        }),
        crate::abi::SourceInput::Path(path) => {
            let source_path = resolve_path_source_input(&path).await?;
            let source_name = crate::diagnostic_path::display_path_for_current_cwd(&source_path);
            let source_fullpath_name = source_path.to_string_lossy().to_string();

            let text = match runmat_runtime::builtins::io::repl_fs::pcode::read_source_text_async(
                &source_path,
            )
            .await
            {
                Ok(text) => text,
                Err(
                    runmat_runtime::builtins::io::repl_fs::pcode::PcodeSourceReadError::InvalidPcode(
                        err,
                    ),
                ) => {
                    return Err(RunError::Runtime(
                        runmat_runtime::builtins::io::repl_fs::pcode::invalid_pcode_runtime_error(
                            format!("{} ({err})", source_path.display()),
                        ),
                    ));
                }
                Err(runmat_runtime::builtins::io::repl_fs::pcode::PcodeSourceReadError::Io(err)) => {
                    return Err(RunError::Runtime(
                        build_runtime_error(format!(
                            "failed to read source path '{}': {err}",
                            source_path.display()
                        ))
                        .with_identifier("RunMat:SourceReadFailed")
                        .build(),
                    ));
                }
            };
            Ok(ResolvedSourceInput {
                display_name: source_name,
                fullpath_name: Some(source_fullpath_name),
                text,
            })
        }
    }
}

async fn resolve_path_source_input(
    path: &str,
) -> std::result::Result<std::path::PathBuf, RunError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use runmat_config::project::resolve_project_source_input_from;
        use std::path::Path;

        let cwd = runmat_filesystem::current_dir().map_err(|err| {
            RunError::Runtime(
                build_runtime_error(format!(
                    "failed to resolve current working directory while resolving source path '{path}': {err}"
                ))
                .with_identifier("RunMat:SourceResolveFailed")
                .build(),
            )
        })?;

        let source_path = std::path::PathBuf::from(path);
        let candidate = crate::diagnostic_path::resolve_against_base(path, &cwd);

        if let Ok(metadata) = runmat_filesystem::metadata_async(&candidate).await {
            if metadata.is_file() {
                return Ok(
                    runmat_runtime::builtins::io::repl_fs::pcode::prefer_pcode_source_path(
                        &candidate,
                    )
                    .await,
                );
            }
        }

        if source_path.extension().is_none() {
            for extension in ["p", "m"] {
                let with_ext = candidate.with_extension(extension);
                if let Ok(metadata) = runmat_filesystem::metadata_async(&with_ext).await {
                    if metadata.is_file() {
                        return Ok(
                            runmat_runtime::builtins::io::repl_fs::pcode::prefer_pcode_source_path(
                                &with_ext,
                            )
                            .await,
                        );
                    }
                }
            }
        }

        let resolved = resolve_project_source_input_from(&cwd, Path::new(path)).map_err(|err| {
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
        })?;
        let resolved = if resolved.is_absolute() {
            resolved
        } else {
            cwd.join(resolved)
        };
        Ok(runmat_runtime::builtins::io::repl_fs::pcode::prefer_pcode_source_path(&resolved).await)
    }

    #[cfg(target_arch = "wasm32")]
    {
        use std::path::PathBuf;

        let cwd = runmat_filesystem::current_dir().map_err(|err| {
            RunError::Runtime(
                build_runtime_error(format!(
                    "failed to resolve current working directory while resolving source path '{path}': {err}"
                ))
                .with_identifier("RunMat:SourceResolveFailed")
                .build(),
            )
        })?;
        let source_path = PathBuf::from(path);
        let candidate = crate::diagnostic_path::resolve_against_base(path, &cwd);

        if let Ok(metadata) = runmat_filesystem::metadata_async(&candidate).await {
            if metadata.is_file() {
                return Ok(
                    runmat_runtime::builtins::io::repl_fs::pcode::prefer_pcode_source_path(
                        &candidate,
                    )
                    .await,
                );
            }
        }

        if source_path.extension().is_none() {
            for extension in ["p", "m"] {
                let with_ext = candidate.with_extension(extension);
                if let Ok(metadata) = runmat_filesystem::metadata_async(&with_ext).await {
                    if metadata.is_file() {
                        return Ok(
                            runmat_runtime::builtins::io::repl_fs::pcode::prefer_pcode_source_path(
                                &with_ext,
                            )
                            .await,
                        );
                    }
                }
            }
        }

        Ok(candidate)
    }
}

#[cfg(test)]
mod tests {
    use super::apply_requested_output_policy;
    #[cfg(not(target_arch = "wasm32"))]
    use super::discover_known_project_symbols;
    #[cfg(not(target_arch = "wasm32"))]
    use super::source_input_text;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::abi::SourceInput;
    use crate::abi::{ExecutionOutcome, RuntimeFlow};
    #[cfg(not(target_arch = "wasm32"))]
    use crate::RunError;
    use runmat_hir::RequestedOutputCount;
    use runmat_value::Value;
    #[cfg(not(target_arch = "wasm32"))]
    use std::fs;
    #[cfg(not(target_arch = "wasm32"))]
    use std::path::{Path, PathBuf};
    #[cfg(not(target_arch = "wasm32"))]
    use std::sync::Arc;

    #[test]
    fn requested_output_policy_maps_checked_zero_one_and_many_results() {
        let outcome = |flow| ExecutionOutcome {
            flow,
            ..ExecutionOutcome::default()
        };

        let zero = apply_requested_output_policy(
            outcome(RuntimeFlow::OutputList(vec![Value::Num(1.0)])),
            &RequestedOutputCount::Zero,
        )
        .unwrap();
        assert!(matches!(zero.flow, RuntimeFlow::NoValue));

        let one = apply_requested_output_policy(
            outcome(RuntimeFlow::OutputList(vec![
                Value::Num(1.0),
                Value::Num(2.0),
            ])),
            &RequestedOutputCount::One,
        )
        .unwrap();
        assert!(matches!(one.flow, RuntimeFlow::Single(Value::Num(1.0))));

        let many = apply_requested_output_policy(
            outcome(RuntimeFlow::Single(Value::Num(1.0))),
            &RequestedOutputCount::Exactly(3),
        )
        .unwrap();
        assert!(matches!(
            many.flow,
            RuntimeFlow::OutputList(ref values) if values == &[Value::Num(1.0)]
        ));
    }

    #[test]
    fn requested_output_policy_rejects_legacy_sequences_at_any_depth() {
        for flow in [
            RuntimeFlow::Single(Value::OutputList(vec![Value::Num(1.0)])),
            RuntimeFlow::OutputList(vec![Value::OutputList(vec![Value::Num(1.0)])]),
        ] {
            let error = apply_requested_output_policy(
                ExecutionOutcome {
                    flow,
                    ..ExecutionOutcome::default()
                },
                &RequestedOutputCount::Zero,
            )
            .unwrap_err();
            let RunError::Runtime(error) = error else {
                panic!("expected runtime error");
            };
            assert_eq!(error.identifier(), Some("RunMat:NestedLegacyOutputList"));
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    struct CwdGuard {
        original: PathBuf,
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn cwd_lock() -> std::sync::MutexGuard<'static, ()> {
        runmat_filesystem::provider_override_lock()
    }

    #[cfg(not(target_arch = "wasm32"))]
    impl Drop for CwdGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.original);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn push_cwd(path: &Path) -> CwdGuard {
        let original = std::env::current_dir().expect("read cwd");
        std::env::set_current_dir(path).expect("set cwd");
        CwdGuard { original }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn source_input_path_resolves_named_manifest_entrypoint() {
        let _guard = cwd_lock();
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

[entrypoints.main]
path = "src/main"
"#,
        )
        .unwrap();
        let _cwd = push_cwd(tmp.path());
        let resolved =
            futures::executor::block_on(source_input_text(SourceInput::Path("main".to_string())))
                .expect("named entrypoint should resolve");
        assert_eq!(
            PathBuf::from(&resolved.display_name),
            PathBuf::from("src").join("main.m")
        );
        let resolved_path = std::path::PathBuf::from(
            resolved
                .fullpath_name
                .as_deref()
                .expect("path source should carry fullpath name"),
        )
        .canonicalize()
        .unwrap();
        let expected = tmp.path().join("src/main.m").canonicalize().unwrap();
        assert_eq!(
            resolved_path, expected,
            "resolved source path should match manifest entrypoint target"
        );
        assert_eq!(resolved.text, "x = 1;");
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn source_input_path_infers_m_extension_for_relative_path() {
        let _guard = cwd_lock();
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();
        let _cwd = push_cwd(tmp.path());

        let resolved = futures::executor::block_on(source_input_text(SourceInput::Path(
            "src/main".to_string(),
        )))
        .expect("path without extension should infer .m");

        assert_eq!(
            PathBuf::from(&resolved.display_name),
            PathBuf::from("src").join("main.m")
        );
        assert_eq!(resolved.text.trim(), "x = 1;");
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn source_input_path_prefers_runmat_pcode_over_m_extension() {
        let _guard = cwd_lock();
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();
        let encoded = runmat_runtime::builtins::io::repl_fs::pcode::encode_pcode_source(
            "x = 2;",
            "src/main.m",
            runmat_runtime::builtins::io::repl_fs::pcode::PcodeAlgorithm::R2007b,
        );
        fs::write(tmp.path().join("src/main.p"), encoded).unwrap();
        let _cwd = push_cwd(tmp.path());

        let resolved = futures::executor::block_on(source_input_text(SourceInput::Path(
            "src/main".to_string(),
        )))
        .expect("path without extension should prefer .p over .m");

        assert_eq!(
            PathBuf::from(&resolved.display_name),
            PathBuf::from("src").join("main.p")
        );
        assert_eq!(resolved.text.trim(), "x = 2;");
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn source_input_path_prefers_runmat_pcode_over_explicit_m_path() {
        let _guard = cwd_lock();
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();
        let encoded = runmat_runtime::builtins::io::repl_fs::pcode::encode_pcode_source(
            "x = 3;",
            "src/main.m",
            runmat_runtime::builtins::io::repl_fs::pcode::PcodeAlgorithm::R2007b,
        );
        fs::write(tmp.path().join("src/main.p"), encoded).unwrap();
        let _cwd = push_cwd(tmp.path());

        let resolved = futures::executor::block_on(source_input_text(SourceInput::Path(
            "src/main.m".to_string(),
        )))
        .expect("explicit .m path should prefer sibling .p");

        assert_eq!(
            PathBuf::from(&resolved.display_name),
            PathBuf::from("src").join("main.p")
        );
        assert_eq!(resolved.text.trim(), "x = 3;");
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn source_input_manifest_entrypoint_prefers_runmat_pcode_over_m_path() {
        let _guard = cwd_lock();
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();
        let encoded = runmat_runtime::builtins::io::repl_fs::pcode::encode_pcode_source(
            "x = 4;",
            "src/main.m",
            runmat_runtime::builtins::io::repl_fs::pcode::PcodeAlgorithm::R2007b,
        );
        fs::write(tmp.path().join("src/main.p"), encoded).unwrap();
        fs::write(
            tmp.path().join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[entrypoints.main]
path = "src/main"
"#,
        )
        .unwrap();
        let _cwd = push_cwd(tmp.path());

        let resolved =
            futures::executor::block_on(source_input_text(SourceInput::Path("main".to_string())))
                .expect("named entrypoint should resolve to P-code sibling");

        assert_eq!(
            PathBuf::from(&resolved.display_name),
            PathBuf::from("src").join("main.p")
        );
        assert_eq!(resolved.text.trim(), "x = 4;");
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn source_input_path_infers_m_extension_from_memory_provider() {
        let _guard = cwd_lock();
        let provider = runmat_filesystem::MemoryFsProvider::new();
        provider.write_project_path("/main.m", b"x = 1;").unwrap();

        runmat_filesystem::with_provider_override(Arc::new(provider), || {
            let resolved = futures::executor::block_on(source_input_text(SourceInput::Path(
                "main".to_string(),
            )))
            .expect("memory provider should resolve path without extension");

            assert_eq!(
                PathBuf::from(&resolved.display_name),
                PathBuf::from("main.m")
            );
            assert_eq!(resolved.text, "x = 1;");
        });
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn source_input_path_errors_for_invalid_named_entrypoint_target() {
        let _guard = cwd_lock();
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(
            tmp.path().join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[entrypoints.server]
module = "app.server"
function = "main"
"#,
        )
        .unwrap();
        let _cwd = push_cwd(tmp.path());
        let err =
            futures::executor::block_on(source_input_text(SourceInput::Path("server".to_string())))
                .expect_err("invalid module/function entrypoint should report resolve error");
        let RunError::Runtime(runtime_err) = err else {
            panic!("expected runtime error");
        };
        assert_eq!(
            runtime_err.identifier.as_deref(),
            Some("RunMat:EntrypointResolveFailed")
        );
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn discover_known_project_symbols_reads_manifest_source_context() {
        let _guard = cwd_lock();
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
        let _cwd = push_cwd(tmp.path());

        let symbols = discover_known_project_symbols(Some(
            tmp.path().join("main.m").to_string_lossy().as_ref(),
        ));
        assert!(
            symbols.contains("stats.summarize"),
            "source-context discovery should include project symbols for eval-hook lowering"
        );
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn discover_known_project_symbols_includes_dependency_alias_qualified_names() {
        let _guard = cwd_lock();
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
        let _cwd = push_cwd(tmp.path());

        let symbols = discover_known_project_symbols(Some(
            tmp.path().join("main.m").to_string_lossy().as_ref(),
        ));
        assert!(
            symbols.contains("summarize"),
            "expected base dependency symbol in known-project discovery"
        );
        assert!(
            !symbols.contains("statslib.summarize"),
            "dependency package identity must not become a MATLAB namespace"
        );
        assert!(
            symbols.contains("statsdep.summarize"),
            "expected dependency-alias-qualified symbol in known-project discovery"
        );
    }
}
