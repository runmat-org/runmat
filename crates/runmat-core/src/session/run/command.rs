//! Source/workspace preparation retires before running the existing target.
use super::*;

pub(super) struct Guards {
    // Fields drop in declaration order: the original reverse lexical order.
    _source_id: runmat_runtime::source_context::SourceContextGuard,
    _fallback_source: runmat_runtime::source_context::SourceContextGuard,
    _eval_hook: runmat_runtime::interaction::EvalHookGuard,
    _input: runmat_runtime::interaction::AsyncHandlerGuard,
    _interrupt: runmat_runtime::interrupt::InterruptGuard,
    _active: ActiveExecutionGuard,
}
pub(super) struct Data<'a> {
    pub input: &'a str,
    pub workspace: preparation::workspace::PreparedInteractiveWorkspace,
    pub start_time: Instant,
    pub debug_trace: bool,
    pub stdin_events: Arc<std::sync::Mutex<Vec<StdinEvent>>>,
    // Metadata/workspace owners retire before restoring scope guards.
    pub guards: Guards,
}
pub(super) struct Prepared<'a> {
    session: &'a mut RunMatSession,
    data: Data<'a>,
}
pub(super) fn run<'a>(
    prepared: Prepared<'a>,
) -> impl std::future::Future<Output = std::result::Result<SessionExecution, RunError>> + 'a {
    super::phases::run(prepared.session, prepared.data)
}
impl RunMatSession {
    pub(super) async fn prepare_command<'a>(
        &'a mut self,
        input: &'a str,
        preserve_layout_var_names: bool,
    ) -> std::result::Result<Prepared<'a>, RunError> {
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
        runmat_runtime::console::reset_thread_buffer();
        runmat_runtime::console::record_diary_command(input);
        runmat_runtime::plotting_hooks::reset_recent_figures();
        runmat_runtime::warning_store::reset();
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
                    ) -> Result<runmat_runtime::execution::RootedValue, RuntimeError>
                    {
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
                        vars.with_owned_output_storage(|vars| {
                            let index = result_idx.ok_or_else(|| {
                                build_runtime_error("input: expression produced no value")
                                    .with_identifier("RunMat:input:NoValue")
                                    .build()
                            })?;
                            let mut vars = vars.try_into_owned().map_err(|_| {
                                runmat_runtime::sequence::sequence_error_to_runtime(
                                    runmat_value::ValueSequenceError::SharedOutputStorage,
                                )
                            })?;
                            let value = vars
                                .as_mut_slice()
                                .get_mut(index)
                                .map(|slot| std::mem::replace(slot, Value::Num(0.0)))
                                .ok_or_else(|| {
                                    build_runtime_error("input: expression produced no value")
                                        .with_identifier("RunMat:input:NoValue")
                                        .build()
                                })?;
                            runmat_runtime::execution::RootedValue::new(value)
                                .map_err(|error| error.into_runtime_error())
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

        let product = self.prepare_interactive_product(input)?;
        let _source_id_guard = runmat_runtime::source_context::replace_current_source_id_in_context(
            &product.runtime,
            product
                .bytecode
                .source_id
                .and_then(|source| source.runtime())
                .and_then(|source| source.resolve_in(&product.runtime)),
        );

        let workspace =
            self.prepare_interactive_workspace(product, preserve_layout_var_names, debug_trace)?;
        Ok(Prepared {
            session: self,
            data: Data {
                input,
                workspace,
                start_time,
                debug_trace,
                stdin_events,
                guards: Guards {
                    _source_id: _source_id_guard,
                    _fallback_source: _fallback_source_guard,
                    _eval_hook: _eval_hook_guard,
                    _input: _async_input_guard,
                    _interrupt: _interrupt_guard,
                    _active,
                },
            },
        })
    }
}
