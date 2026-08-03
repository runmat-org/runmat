use super::*;

impl RunMatSession {
    /// Invoke one immutable unit while collecting its backend-independent
    /// function and statement coverage sites.
    pub async fn invoke_executable_with_coverage(
        &mut self,
        unit: &crate::ExecutableUnit,
        invocation: crate::ProcedureInvocation,
        control: &crate::InvocationControl,
    ) -> std::result::Result<(Value, crate::CoverageFragment), RunError> {
        let coverage = runmat_vm::coverage::CoverageSession::start();
        let value = self.invoke_executable(unit, invocation, control).await?;
        let program_revision = unit
            .revision()
            .program_revision
            .as_ref()
            .map(runmat_execution::ProgramRevision::canonical_identity)
            .unwrap_or_else(|| unit.revision().source_digest.clone());
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
        let _test_services = runmat_runtime::testing::install_test_services(
            crate::testing::runtime_adapter::services(
                self.compat_mode,
                self.project_handoff.clone(),
            ),
        );
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
                self.invoke_entrypoint(unit).await
            }
            crate::ProcedureTarget::Function(name) => {
                let function = unit.functions().resolve_name(&name).ok_or_else(|| {
                    runtime_error(
                        "RunMat:UndefinedExecutableProcedure",
                        format!("executable unit does not contain semantic procedure '{name}'"),
                    )
                })?;
                self.invoke_procedure(
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
        runmat_vm::set_call_stack_limit(self.callstack_limit);
        runmat_vm::set_error_namespace(&self.error_namespace);
        runmat_vm::set_dynamic_eval_options(
            self.compat_mode,
            self.compat_mode.allows_runmat_extensions(),
            self.top_level_await_enabled,
            self.dynamic_eval_enabled,
        );
    }

    async fn invoke_entrypoint(
        &mut self,
        unit: &crate::ExecutableUnit,
    ) -> std::result::Result<Value, RuntimeError> {
        let mut variables = vec![Value::Num(0.0); unit.bytecode().var_count];
        #[cfg(feature = "jit")]
        if let Some(result) = self.execute_executable_backend(
            unit.bytecode(),
            &mut variables,
            &unit.backend_cache_namespace(),
        ) {
            result?;
            return Ok(Value::OutputList(Vec::new()));
        }
        match runmat_vm::interpret_with_vars(
            unit.bytecode(),
            &mut variables,
            Some(&unit.source().relative_path),
        )
        .await?
        {
            runmat_vm::InterpreterOutcome::Completed(values) => Ok(values
                .into_iter()
                .last()
                .unwrap_or(Value::OutputList(Vec::new()))),
        }
    }

    async fn invoke_procedure(
        &mut self,
        unit: &crate::ExecutableUnit,
        function: runmat_hir::FunctionId,
        arguments: Vec<Value>,
        requested_outputs: usize,
    ) -> std::result::Result<Value, RuntimeError> {
        #[cfg(feature = "jit")]
        if let Some(mut program) =
            unit.procedure_program(function, arguments.clone(), requested_outputs)
        {
            if let Some(result) = self.execute_executable_backend(
                &program.bytecode,
                &mut program.variables,
                &unit.backend_cache_namespace(),
            ) {
                result?;
                return Ok(program
                    .result_slot
                    .and_then(|slot| program.variables.get(slot).cloned())
                    .unwrap_or(Value::OutputList(Vec::new())));
            }
        }
        runmat_vm::invoke_semantic_function_value(
            function.0,
            &arguments,
            requested_outputs,
            unit.functions(),
        )
        .await
    }

    #[cfg(feature = "jit")]
    fn execute_executable_backend(
        &mut self,
        bytecode: &runmat_vm::Bytecode,
        variables: &mut Vec<Value>,
        cache_namespace: &str,
    ) -> Option<std::result::Result<(), RuntimeError>> {
        let engine = self.jit_engine.as_mut()?;
        Some(
            engine
                .execute_or_compile_with_cache_namespace(bytecode, variables, Some(cache_namespace))
                .map(|(_, used_jit)| {
                    if used_jit {
                        self.stats.jit_compiled += 1;
                    } else {
                        self.stats.interpreter_fallback += 1;
                    }
                })
                .map_err(|error| match error {
                    runmat_turbine::TurbineError::ExecutionError(error) => error,
                    error => build_runtime_error(error.to_string())
                        .with_identifier("RunMat:ExecutableBackend")
                        .build(),
                }),
        )
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
