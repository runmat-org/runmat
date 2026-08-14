use super::*;

impl RunMatSession {
    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn set_native_tiering_config_for_testing(
        &mut self,
        config: runmat_jit::tiering::TieringConfig,
    ) {
        self.generic_native_cache =
            crate::generic_native::GenericNativeCache::with_tiering_config(config);
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn native_tiering_snapshot_for_testing(
        &self,
    ) -> runmat_jit::tiering::TierFeedbackSnapshot {
        self.generic_native_cache.tiering_snapshot()
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn generic_native_cache_counts(&self) -> (usize, usize) {
        (
            self.generic_native_cache.compilation_count(),
            self.generic_native_cache.published_entry_count(),
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn invoke_generic_native(
        &mut self,
        unit: &crate::ExecutableUnit,
        preferred_function: Option<&str>,
        arguments: Vec<Value>,
        requested_outputs: usize,
    ) -> std::result::Result<Value, RuntimeError> {
        let published = self
            .generic_native_cache
            .resolve_or_compile(unit, preferred_function)?;
        crate::generic_native::invoke(
            unit,
            published,
            preferred_function,
            arguments,
            requested_outputs,
            self.runtime_context
                .clone()
                .with_program_revision(Some(unit.revision().program_revision.clone())),
        )
        .await
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn invoke_tiered_entrypoint(
        &mut self,
        unit: &crate::ExecutableUnit,
    ) -> std::result::Result<Value, RuntimeError> {
        if !self.native_tiering_enabled {
            return self.invoke_entrypoint_interpreted(unit).await;
        }
        let Some(profile) = self.generic_native_cache.representation_profile(&[]) else {
            return self.invoke_entrypoint_interpreted(unit).await;
        };
        let started = std::time::Instant::now();
        let published = self.generic_native_cache.resolve_or_schedule(unit, None)?;
        let result = match published {
            Some(published) => {
                self.stats.jit_compiled += 1;
                crate::generic_native::invoke(
                    unit,
                    published,
                    None,
                    Vec::new(),
                    0,
                    self.runtime_context
                        .clone()
                        .with_program_revision(Some(unit.revision().program_revision.clone())),
                )
                .await
            }
            None => {
                self.stats.interpreter_fallback += 1;
                self.invoke_entrypoint_interpreted(unit).await
            }
        };
        if let Err(error) = self.generic_native_cache.observe_invocation(
            unit,
            None,
            &profile,
            runmat_time::duration_ns_saturating(started.elapsed()),
        ) {
            log::warn!("native tier feedback was not recorded: {error}");
        }
        result
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn invoke_tiered_procedure(
        &mut self,
        unit: &crate::ExecutableUnit,
        function: runmat_hir::FunctionId,
        name: &str,
        arguments: Vec<Value>,
        requested_outputs: usize,
    ) -> std::result::Result<Value, RuntimeError> {
        if !self.native_tiering_enabled {
            return self
                .invoke_procedure_interpreted(unit, function, arguments, requested_outputs)
                .await;
        }
        let Some(profile) = self.generic_native_cache.representation_profile(&arguments) else {
            return self
                .invoke_procedure_interpreted(unit, function, arguments, requested_outputs)
                .await;
        };
        let started = std::time::Instant::now();
        let published = self
            .generic_native_cache
            .resolve_or_schedule(unit, Some(name))?;
        let result = match published {
            Some(published) => {
                self.stats.jit_compiled += 1;
                crate::generic_native::invoke(
                    unit,
                    published,
                    Some(name),
                    arguments,
                    requested_outputs,
                    self.runtime_context
                        .clone()
                        .with_program_revision(Some(unit.revision().program_revision.clone())),
                )
                .await
            }
            None => {
                self.stats.interpreter_fallback += 1;
                self.invoke_procedure_interpreted(unit, function, arguments, requested_outputs)
                    .await
            }
        };
        if let Err(error) = self.generic_native_cache.observe_invocation(
            unit,
            Some(name),
            &profile,
            runmat_time::duration_ns_saturating(started.elapsed()),
        ) {
            log::warn!("native tier feedback was not recorded: {error}");
        }
        result
    }

    /// Invoke one immutable unit while collecting its backend-independent
    /// function and statement coverage sites.
    pub async fn invoke_executable_with_coverage(
        &mut self,
        unit: &crate::ExecutableUnit,
        invocation: crate::ProcedureInvocation,
        control: &crate::InvocationControl,
    ) -> std::result::Result<(Value, crate::CoverageFragment), RunError> {
        let coverage = runmat_vm::coverage::CoverageSession::start(&self.runtime_context);
        let value = self.invoke_executable(unit, invocation, control).await?;
        let program_revision = unit.revision().program_revision.canonical_identity();
        let fragment = unit
            .coverage_plan()
            .fragment(program_revision, coverage.counts());
        Ok((value, fragment))
    }

    /// Invoke an immutable executable entrypoint or exact semantic procedure.
    ///
    /// This path does not synthesize or append source and does not publish the
    /// unit into the interactive workspace. When Turbine is enabled, exact
    /// procedure calls pass through the same tiering policy as ordinary
    /// session execution.
    pub async fn invoke_executable(
        &mut self,
        unit: &crate::ExecutableUnit,
        invocation: crate::ProcedureInvocation,
        control: &crate::InvocationControl,
    ) -> std::result::Result<Value, RunError> {
        self.configure_runtime_context();
        let runtime = self
            .runtime_context
            .clone()
            .with_program_revision(Some(unit.revision().program_revision.clone()));
        runtime
            .scope(self.invoke_executable_in_context(unit, invocation, control))
            .await
    }

    async fn invoke_executable_in_context(
        &mut self,
        unit: &crate::ExecutableUnit,
        invocation: crate::ProcedureInvocation,
        control: &crate::InvocationControl,
    ) -> std::result::Result<Value, RunError> {
        let _test_services = runmat_runtime::testing::install_test_services(
            crate::testing::runtime_adapter::services(
                self.compat_mode,
                self.project_handoff.clone(),
            ),
        );
        // Builtin class registration is session-owned. Register the testing
        // projection after entering this session's runtime context so
        // function-based TestCase receivers resolve inherited methods here,
        // rather than only in the caller thread's fallback registry.
        runmat_runtime::testing::ensure_testing_classes();
        ensure_invocation_allowed(control)?;
        self.stats.total_executions += 1;
        self.configure_executable_runtime();
        let _interrupt_guard =
            runmat_runtime::interrupt::replace_interrupt(Some(self.interrupt_flag.clone()));
        let _source_guard = runmat_runtime::source_context::replace_current_source_context(
            Some(&unit.source().relative_path),
            Some(&unit.source().text),
        );
        let _source_catalog_guard =
            runmat_runtime::source_context::replace_source_catalog_with_fullpaths(
                unit.source_map()
                    .entries()
                    .iter()
                    .map(|entry| {
                        (
                            runmat_hir::SourceId(entry.source_id),
                            entry.display_name.clone(),
                            entry.full_path.clone(),
                            entry.text.clone(),
                        )
                    })
                    .collect(),
            );
        let _source_id_guard =
            runmat_runtime::source_context::replace_current_source_id(unit.bytecode().source_id);

        let result = match invocation.target {
            crate::ProcedureTarget::Entrypoint => {
                if !invocation.arguments.is_empty() {
                    return Err(runtime_error(
                        "RunMat:ExecutableArity",
                        "executable entrypoint does not accept invocation arguments",
                    ));
                }
                #[cfg(not(target_arch = "wasm32"))]
                if control.backend() == crate::ExecutableBackendPolicy::ForcedGenericNative {
                    self.invoke_generic_native(unit, None, Vec::new(), 0).await
                } else {
                    self.invoke_tiered_entrypoint(unit).await
                }
                #[cfg(target_arch = "wasm32")]
                self.invoke_entrypoint_interpreted(unit).await
            }
            crate::ProcedureTarget::Function(name) => {
                let function = unit.functions().resolve_name(&name).ok_or_else(|| {
                    runtime_error(
                        "RunMat:UndefinedExecutableProcedure",
                        format!("executable unit does not contain semantic procedure '{name}'"),
                    )
                })?;
                #[cfg(not(target_arch = "wasm32"))]
                if control.backend() == crate::ExecutableBackendPolicy::ForcedGenericNative {
                    self.invoke_generic_native(
                        unit,
                        Some(name.as_str()),
                        invocation.arguments,
                        invocation.requested_outputs,
                    )
                    .await
                } else {
                    self.invoke_tiered_procedure(
                        unit,
                        function,
                        &name,
                        invocation.arguments,
                        invocation.requested_outputs,
                    )
                    .await
                }
                #[cfg(target_arch = "wasm32")]
                self.invoke_procedure_interpreted(
                    unit,
                    function,
                    invocation.arguments,
                    invocation.requested_outputs,
                )
                .await
            }
        };
        let mut value = result.map_err(RunError::Runtime)?;
        ensure_invocation_allowed(control)?;
        if invocation.requested_outputs == 0 {
            value = Value::OutputList(Vec::new());
        }
        Ok(value)
    }

    fn configure_executable_runtime(&self) {
        runmat_vm::set_dynamic_eval_options(
            self.compat_mode,
            self.compat_mode.allows_runmat_extensions(),
            self.top_level_await_enabled,
            self.dynamic_eval_enabled,
        );
    }

    async fn invoke_entrypoint_interpreted(
        &mut self,
        unit: &crate::ExecutableUnit,
    ) -> std::result::Result<Value, RuntimeError> {
        let mut variables = vec![Value::Num(0.0); unit.bytecode().var_count];
        match runmat_vm::interpret_with_vars_in_context(
            unit.bytecode(),
            &mut variables,
            Some(&unit.source().relative_path),
            self.runtime_context
                .clone()
                .with_program_revision(Some(unit.revision().program_revision.clone())),
        )
        .await?
        {
            runmat_vm::InterpreterOutcome::Completed(values) => Ok(values
                .into_iter()
                .last()
                .unwrap_or(Value::OutputList(Vec::new()))),
        }
    }

    async fn invoke_procedure_interpreted(
        &mut self,
        unit: &crate::ExecutableUnit,
        function: runmat_hir::FunctionId,
        arguments: Vec<Value>,
        requested_outputs: usize,
    ) -> std::result::Result<Value, RuntimeError> {
        runmat_vm::invoke_semantic_function_value_in_context(
            function.0,
            &arguments,
            requested_outputs,
            unit.functions(),
            self.runtime_context
                .clone()
                .with_program_revision(Some(unit.revision().program_revision.clone())),
        )
        .await
    }
}

fn ensure_invocation_allowed(
    control: &crate::InvocationControl,
) -> std::result::Result<(), RunError> {
    if control.is_cancelled() {
        return Err(runtime_error(
            "RunMat:ExecutionCancelled",
            "executable invocation was cancelled",
        ));
    }
    if control.deadline_elapsed() {
        return Err(runtime_error(
            "RunMat:ExecutionDeadline",
            "executable invocation deadline elapsed",
        ));
    }
    Ok(())
}

fn runtime_error(identifier: &'static str, message: impl Into<String>) -> RunError {
    RunError::Runtime(
        build_runtime_error(message)
            .with_identifier(identifier)
            .build(),
    )
}
