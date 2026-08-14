use super::*;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::BTreeMap;

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
    pub(crate) fn specialized_native_version_count_for_testing(&self) -> usize {
        self.generic_native_cache.specialized_version_count()
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn native_osr_transfer_count_for_testing(&self) -> usize {
        self.stats.native_osr_transfers
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn vectorized_native_region_count_for_testing(&self) -> u64 {
        self.stats.vectorized_native_regions
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn optimized_region_plan_count_for_testing(
        &self,
        unit: &crate::ExecutableUnit,
        preferred_function: Option<&str>,
        arguments: &[Value],
    ) -> usize {
        self.generic_native_cache
            .representation_profile(arguments)
            .map(|profile| {
                self.generic_native_cache.optimized_region_plan_count(
                    unit,
                    preferred_function,
                    &profile,
                )
            })
            .unwrap_or(0)
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
            crate::generic_native::NativeInvocation {
                published,
                osr: None,
                preferred_function,
                arguments,
                requested_outputs,
                runtime: self
                    .runtime_context
                    .clone()
                    .with_program_revision(Some(unit.revision().program_revision.clone())),
                workspace: None,
            },
        )
        .await
        .map(|execution| execution.value)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn prepare_tiered_interactive(
        &self,
        unit: &crate::ExecutableUnit,
    ) -> Option<(
        runmat_jit::execute::NativeWorkspaceInput,
        runmat_jit::tiering::RepresentationProfile,
    )> {
        if !self.native_tiering_enabled {
            return None;
        }
        let values = self
            .workspace_values
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect::<BTreeMap<_, _>>();
        let local_names = unit
            .entrypoint_workspace_bindings()
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        let bindings = local_names
            .iter()
            .map(|(binding, name)| (*binding, name.clone()))
            .filter_map(|(binding, name)| {
                values.get(&name).cloned().map(|value| {
                    runmat_jit::execute::NativeWorkspaceBinding {
                        binding,
                        name,
                        value,
                    }
                })
            })
            .collect();
        let workspace = runmat_jit::execute::NativeWorkspaceInput {
            values,
            local_names,
            bindings,
        };
        let profile_values = workspace.profile_values();
        self.generic_native_cache
            .representation_profile(&profile_values)
            .map(|profile| (workspace, profile))
    }

    /// Execute only an already-admitted native product. `None` preserves the
    /// canonical cold interpreter path; observation happens after either path
    /// completes so timing and hotness describe real semantic execution.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) async fn invoke_tiered_interactive(
        &mut self,
        unit: &crate::ExecutableUnit,
        workspace: &runmat_jit::execute::NativeWorkspaceInput,
        profile: &runmat_jit::tiering::RepresentationProfile,
    ) -> std::result::Result<Option<crate::generic_native::NativeExecution>, RuntimeError> {
        let published = self
            .generic_native_cache
            .resolve_or_schedule(unit, None, profile)?;
        let osr = if published
            .as_ref()
            .is_some_and(|entry| matches!(entry.tier, runmat_jit::entry::PublishedTier::Generic))
        {
            self.generic_native_cache
                .resolve_or_schedule_osr(unit, None, profile)?
        } else {
            None
        };
        let Some(published) = published else {
            return Ok(None);
        };
        self.stats.jit_compiled += 1;
        let execution = crate::generic_native::invoke(
            unit,
            crate::generic_native::NativeInvocation {
                published,
                osr,
                preferred_function: None,
                arguments: Vec::new(),
                requested_outputs: 0,
                runtime: self
                    .runtime_context
                    .clone()
                    .with_program_revision(Some(unit.revision().program_revision.clone())),
                workspace: Some(workspace.clone()),
            },
        )
        .await?;
        self.stats.native_osr_transfers += usize::from(execution.osr_entry.is_some());
        self.stats.vectorized_native_regions = self
            .stats
            .vectorized_native_regions
            .saturating_add(execution.vectorized_regions);
        Ok(Some(execution))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn observe_tiered_interactive(
        &self,
        unit: &crate::ExecutableUnit,
        profile: &runmat_jit::tiering::RepresentationProfile,
        elapsed: std::time::Duration,
        backedges: &BTreeMap<runmat_types::ProgramPointId, u64>,
    ) {
        if let Err(error) = self.generic_native_cache.observe_invocation(
            unit,
            None,
            profile,
            runmat_time::duration_ns_saturating(elapsed),
        ) {
            log::warn!("native tier feedback was not recorded: {error}");
        }
        if let Err(error) = self
            .generic_native_cache
            .observe_loop_backedges(unit, None, backedges)
        {
            log::warn!("native loop feedback was not recorded: {error}");
        }
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
        let published = self
            .generic_native_cache
            .resolve_or_schedule(unit, None, &profile)?;
        let osr = if published
            .as_ref()
            .is_some_and(|entry| matches!(entry.tier, runmat_jit::entry::PublishedTier::Generic))
        {
            self.generic_native_cache
                .resolve_or_schedule_osr(unit, None, &profile)?
        } else {
            None
        };
        let (result, backedges, osr_entry, vectorized_regions) = match published {
            Some(published) => {
                self.stats.jit_compiled += 1;
                let execution = crate::generic_native::invoke(
                    unit,
                    crate::generic_native::NativeInvocation {
                        published,
                        osr,
                        preferred_function: None,
                        arguments: Vec::new(),
                        requested_outputs: 0,
                        runtime: self
                            .runtime_context
                            .clone()
                            .with_program_revision(Some(unit.revision().program_revision.clone())),
                        workspace: None,
                    },
                )
                .await;
                match execution {
                    Ok(execution) => (
                        Ok(execution.value),
                        execution.loop_backedges,
                        execution.osr_entry,
                        execution.vectorized_regions,
                    ),
                    Err(error) => (Err(error), BTreeMap::new(), None, 0),
                }
            }
            None => {
                self.stats.interpreter_fallback += 1;
                (
                    self.invoke_entrypoint_interpreted(unit).await,
                    BTreeMap::new(),
                    None,
                    0,
                )
            }
        };
        self.stats.native_osr_transfers += usize::from(osr_entry.is_some());
        self.stats.vectorized_native_regions = self
            .stats
            .vectorized_native_regions
            .saturating_add(vectorized_regions);
        if let Err(error) = self.generic_native_cache.observe_invocation(
            unit,
            None,
            &profile,
            runmat_time::duration_ns_saturating(started.elapsed()),
        ) {
            log::warn!("native tier feedback was not recorded: {error}");
        }
        if let Err(error) = self
            .generic_native_cache
            .observe_loop_backedges(unit, None, &backedges)
        {
            log::warn!("native loop feedback was not recorded: {error}");
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
        let published =
            self.generic_native_cache
                .resolve_or_schedule(unit, Some(name), &profile)?;
        let osr = if published
            .as_ref()
            .is_some_and(|entry| matches!(entry.tier, runmat_jit::entry::PublishedTier::Generic))
        {
            self.generic_native_cache
                .resolve_or_schedule_osr(unit, Some(name), &profile)?
        } else {
            None
        };
        let (result, backedges, osr_entry, vectorized_regions) = match published {
            Some(published) => {
                self.stats.jit_compiled += 1;
                let execution = crate::generic_native::invoke(
                    unit,
                    crate::generic_native::NativeInvocation {
                        published,
                        osr,
                        preferred_function: Some(name),
                        arguments,
                        requested_outputs,
                        runtime: self
                            .runtime_context
                            .clone()
                            .with_program_revision(Some(unit.revision().program_revision.clone())),
                        workspace: None,
                    },
                )
                .await;
                match execution {
                    Ok(execution) => (
                        Ok(execution.value),
                        execution.loop_backedges,
                        execution.osr_entry,
                        execution.vectorized_regions,
                    ),
                    Err(error) => (Err(error), BTreeMap::new(), None, 0),
                }
            }
            None => {
                self.stats.interpreter_fallback += 1;
                (
                    self.invoke_procedure_interpreted(unit, function, arguments, requested_outputs)
                        .await,
                    BTreeMap::new(),
                    None,
                    0,
                )
            }
        };
        self.stats.native_osr_transfers += usize::from(osr_entry.is_some());
        self.stats.vectorized_native_regions = self
            .stats
            .vectorized_native_regions
            .saturating_add(vectorized_regions);
        if let Err(error) = self.generic_native_cache.observe_invocation(
            unit,
            Some(name),
            &profile,
            runmat_time::duration_ns_saturating(started.elapsed()),
        ) {
            log::warn!("native tier feedback was not recorded: {error}");
        }
        if let Err(error) =
            self.generic_native_cache
                .observe_loop_backedges(unit, Some(name), &backedges)
        {
            log::warn!("native loop feedback was not recorded: {error}");
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
    /// unit into the interactive workspace. Native-capable sessions collect
    /// bounded facts and tier hot exact procedures through the MIR/Native-IR
    /// executor; other targets retain the canonical portable path.
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
