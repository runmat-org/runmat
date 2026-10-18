use super::*;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::BTreeMap;
use std::rc::Rc;

mod selection;

type InvocationResult =
    std::result::Result<runmat_runtime::execution::RootedValueSequence, RunError>;
type CoveredInvocationResult = std::result::Result<
    (
        runmat_runtime::execution::RootedValueSequence,
        crate::CoverageFragment,
    ),
    RunError,
>;

#[cfg(not(target_arch = "wasm32"))]
pub(super) enum TieredInteractiveInvocation {
    Cold,
    Native(
        std::result::Result<Box<crate::generic_native::RootedNativeExecution>, Box<RuntimeError>>,
    ),
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeProcedureTarget<'a> {
    function: runmat_hir::FunctionId,
    name: &'a str,
}

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
        runtime: &runmat_runtime::context::RuntimeContext,
        runtime_functions: &runmat_vm::FunctionRegistry,
        preferred_function: Option<&str>,
        arguments: Vec<Value>,
        requested_outputs: usize,
    ) -> std::result::Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
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
                runtime: runtime.clone(),
                dynamic_admission: Some(self.source_evaluator.dynamic_admission_port()),
                runtime_functions,
                workspace: None,
            },
        )
        .await
        .and_then(|execution| {
            execution.try_map_owned(
                |execution| Ok(execution.outputs),
                |error| error.into_runtime_error(),
            )
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn prepare_tiered_interactive(
        &self,
        unit: &crate::ExecutableUnit,
        workspace_projection: &runmat_runtime::workspace::BaseWorkspaceProjectionService,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> std::result::Result<
        Option<(
            runmat_native_executor::execute::NativeWorkspaceInput,
            Option<runmat_jit::tiering::RepresentationProfile>,
        )>,
        RuntimeError,
    > {
        if !self.native_tiering_enabled {
            log::debug!(target: "runmat_core::native_tiering", "native interactive admission skipped reason=tiering_disabled");
            return Ok(None);
        }
        // Empty inputs and function-definition-only inputs intentionally have no
        // executable script entrypoint. They still update session state through
        // the interpreter path, but there is no native tier site to profile or
        // publish for them.
        if unit.script_entrypoint().is_none() {
            log::debug!(target: "runmat_core::native_tiering", "native interactive admission skipped reason=no_script_entrypoint");
            return Ok(None);
        }
        runmat_runtime::context::RuntimeWorkspaceService::ensure_effect_authority_with_context(
            workspace_projection,
            runtime,
        )?;
        let generation = runmat_runtime::workspace::LogicalWorkspaceGeneration::capture(
            runtime,
            &self.base_workspace,
            &runtime.session_variable_authority(),
        )?;
        let local_names = unit
            .entrypoint_workspace_bindings()
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        let bindings = local_names
            .iter()
            .filter(|(_, name)| generation.get(name).is_some())
            .map(
                |(binding, name)| runmat_native_executor::execute::NativeWorkspaceBinding {
                    binding: *binding,
                    name: name.clone(),
                },
            )
            .collect();
        let mut global_bindings = Vec::new();
        for (binding, name) in &local_names {
            if let Some(target) = generation.global_target(name) {
                global_bindings.push(
                    runmat_native_executor::execute::NativeWorkspaceGlobalBinding {
                        binding: *binding,
                        name: name.clone(),
                        target: target.clone(),
                    },
                );
            }
        }
        let workspace = runmat_native_executor::execute::NativeWorkspaceInput::from_generation(
            generation,
            local_names,
            bindings,
            global_bindings,
        );
        let profile = self
            .generic_native_cache
            .representation_profile_from_values(workspace.profile_values());
        log::debug!(target: "runmat_core::native_tiering",
            "native interactive admission workspace_values={} specialization_profile_available={}",
            workspace.iter().count(), profile.is_some());
        // Generic execution and feedback do not require a bounded specialization
        // profile. The complete workspace still reaches native execution.
        Ok(Some((workspace, profile)))
    }

    /// Execute only an already-admitted native product. The typed receipt
    /// distinguishes a cold interpreter decision from a native invocation
    /// that entered generated code and returned a language error.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) async fn invoke_tiered_interactive(
        &mut self,
        unit: &crate::ExecutableUnit,
        runtime: &runmat_runtime::context::RuntimeContext,
        runtime_functions: &runmat_vm::FunctionRegistry,
        workspace: &runmat_native_executor::execute::NativeWorkspaceInput,
        profile: Option<&runmat_jit::tiering::RepresentationProfile>,
    ) -> std::result::Result<TieredInteractiveInvocation, RuntimeError> {
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
            return Ok(TieredInteractiveInvocation::Cold);
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
                runtime: runtime.clone(),
                dynamic_admission: Some(self.source_evaluator.dynamic_admission_port()),
                runtime_functions,
                workspace: Some(workspace.clone()),
            },
        )
        .await;
        if let Ok(execution) = &execution {
            self.stats.native_osr_transfers += usize::from(execution.osr_entry.is_some());
            self.stats.vectorized_native_regions = self
                .stats
                .vectorized_native_regions
                .saturating_add(execution.vectorized_regions);
        }
        Ok(TieredInteractiveInvocation::Native(
            execution.map(Box::new).map_err(Box::new),
        ))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn observe_tiered_interactive(
        &self,
        unit: &crate::ExecutableUnit,
        profile: Option<&runmat_jit::tiering::RepresentationProfile>,
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
        runtime: &runmat_runtime::context::RuntimeContext,
        runtime_bytecode: &runmat_vm::Bytecode,
    ) -> std::result::Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
        if !self.native_tiering_enabled {
            return self
                .invoke_entrypoint_interpreted(unit, runtime, runtime_bytecode)
                .await;
        }
        let profile = self.generic_native_cache.representation_profile(&[]);
        let started = std::time::Instant::now();
        let published =
            self.generic_native_cache
                .resolve_or_schedule(unit, None, profile.as_ref())?;
        let osr = if published
            .as_ref()
            .is_some_and(|entry| matches!(entry.tier, runmat_jit::entry::PublishedTier::Generic))
        {
            self.generic_native_cache
                .resolve_or_schedule_osr(unit, None, profile.as_ref())?
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
                        runtime: runtime.clone(),
                        dynamic_admission: Some(self.source_evaluator.dynamic_admission_port()),
                        runtime_functions: &runtime_bytecode.function_registry,
                        workspace: None,
                    },
                )
                .await;
                match execution {
                    Ok(execution) => {
                        let backedges = execution.loop_backedges.clone();
                        let osr_entry = execution.osr_entry;
                        let vectorized_regions = execution.vectorized_regions;
                        (
                            execution.try_map_owned(
                                |execution| Ok(execution.outputs),
                                |error| error.into_runtime_error(),
                            ),
                            backedges,
                            osr_entry,
                            vectorized_regions,
                        )
                    }
                    Err(error) => (Err(error), BTreeMap::new(), None, 0),
                }
            }
            None => {
                self.stats.interpreter_fallback += 1;
                (
                    self.invoke_entrypoint_interpreted(unit, runtime, runtime_bytecode)
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
            None,
            profile.as_ref(),
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
        runtime: &runmat_runtime::context::RuntimeContext,
        runtime_bytecode: &runmat_vm::Bytecode,
        target: NativeProcedureTarget<'_>,
        arguments: Vec<Value>,
        requested_outputs: usize,
    ) -> std::result::Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
        let NativeProcedureTarget { function, name } = target;
        if !self.native_tiering_enabled {
            return self
                .invoke_procedure_interpreted(
                    unit,
                    runtime,
                    runtime_bytecode,
                    function,
                    arguments,
                    requested_outputs,
                )
                .await;
        }
        let profile = self.generic_native_cache.representation_profile(&arguments);
        let started = std::time::Instant::now();
        let published =
            self.generic_native_cache
                .resolve_or_schedule(unit, Some(name), profile.as_ref())?;
        let osr = if published
            .as_ref()
            .is_some_and(|entry| matches!(entry.tier, runmat_jit::entry::PublishedTier::Generic))
        {
            self.generic_native_cache
                .resolve_or_schedule_osr(unit, Some(name), profile.as_ref())?
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
                        runtime: runtime.clone(),
                        dynamic_admission: Some(self.source_evaluator.dynamic_admission_port()),
                        runtime_functions: &runtime_bytecode.function_registry,
                        workspace: None,
                    },
                )
                .await;
                match execution {
                    Ok(execution) => {
                        let backedges = execution.loop_backedges.clone();
                        let osr_entry = execution.osr_entry;
                        let vectorized_regions = execution.vectorized_regions;
                        (
                            execution.try_map_owned(
                                |execution| Ok(execution.outputs),
                                |error| error.into_runtime_error(),
                            ),
                            backedges,
                            osr_entry,
                            vectorized_regions,
                        )
                    }
                    Err(error) => (Err(error), BTreeMap::new(), None, 0),
                }
            }
            None => {
                self.stats.interpreter_fallback += 1;
                (
                    self.invoke_procedure_interpreted(
                        unit,
                        runtime,
                        runtime_bytecode,
                        function,
                        arguments,
                        requested_outputs,
                    )
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
            profile.as_ref(),
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
    pub fn invoke_executable_with_coverage<'a>(
        &'a mut self,
        unit: &'a crate::ExecutableUnit,
        invocation: crate::ProcedureInvocation,
        control: &'a crate::InvocationControl,
    ) -> impl std::future::Future<Output = CoveredInvocationResult> + 'a {
        let argument_roots = self.root_invocation_arguments(&invocation, control);
        async move {
            let argument_roots = argument_roots?;
            let coverage = runmat_runtime::coverage::CoverageSession::start(&self.runtime_context);
            let value = self
                .invoke_rooted_executable(unit, invocation, control, argument_roots)
                .await?;
            let program_revision = unit.revision().program_revision.canonical_identity();
            let fragment = unit
                .coverage_plan()
                .fragment(program_revision, coverage.counts());
            Ok((value, fragment))
        }
    }

    /// Invoke an immutable executable entrypoint or exact semantic procedure.
    ///
    /// This path does not synthesize or append source and does not publish the
    /// unit into the interactive workspace. Native-capable sessions collect
    /// bounded facts and tier hot exact procedures through the MIR/Native-IR
    /// executor; other targets retain the canonical portable path.
    pub fn invoke_executable<'a>(
        &'a mut self,
        unit: &'a crate::ExecutableUnit,
        invocation: crate::ProcedureInvocation,
        control: &'a crate::InvocationControl,
    ) -> impl std::future::Future<Output = InvocationResult> + 'a {
        // Acquire roots at construction, not at the future's first poll. An
        // embedding may retire its own roots as soon as this future owns the
        // arguments, or retain/drop that future without polling it.
        let argument_roots = self.root_invocation_arguments(&invocation, control);
        async move {
            self.invoke_rooted_executable(unit, invocation, control, argument_roots?)
                .await
        }
    }

    fn root_invocation_arguments(
        &self,
        invocation: &crate::ProcedureInvocation,
        control: &crate::InvocationControl,
    ) -> std::result::Result<runmat_runtime::execution::RootedValueLease, RunError> {
        ensure_invocation_allowed(control, &self.runtime_context)?;
        runmat_runtime::execution::RootedValueLease::for_values(invocation.arguments.iter())
            .map_err(|error| RunError::Runtime(error.into_runtime_error()))
    }

    fn invoke_rooted_executable<'a>(
        &'a mut self,
        unit: &'a crate::ExecutableUnit,
        invocation: crate::ProcedureInvocation,
        control: &'a crate::InvocationControl,
        argument_roots: runmat_runtime::execution::RootedValueLease,
    ) -> impl std::future::Future<Output = InvocationResult> + 'a {
        // Reentrant execution (including runtests) must not embed another
        // complete executable state in every caller's async frame. Pin that
        // state once, retaining the same arguments and roots without copying.
        let runtime = self.runtime_context.clone();
        let admitted = ensure_invocation_allowed(control, &runtime).and_then(|()| {
            runmat_runtime::execution::admit_executable_sequence_with_error(
                &runtime,
                executable_continuation_error,
                async move {
                    let _argument_roots = argument_roots;
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let local_tasks = self.local_tasks.clone();
                        return local_tasks
                            .participate_foreground(
                                self.invoke_executable_with_cleanup(unit, invocation, control),
                            )
                            .await
                            .map_err(RunError::Runtime)?;
                    }
                    #[cfg(target_arch = "wasm32")]
                    self.invoke_executable_with_cleanup(unit, invocation, control)
                        .await
                },
            )
        });
        async move { admitted?.await }
    }

    async fn invoke_executable_with_cleanup(
        &mut self,
        unit: &crate::ExecutableUnit,
        invocation: crate::ProcedureInvocation,
        control: &crate::InvocationControl,
    ) -> std::result::Result<runmat_runtime::execution::RootedValueSequence, RunError> {
        let profiling_root = self.admit_command_profiling();
        let receipt = self
            .runtime_context
            .command_profiling()
            .cloned()
            .expect("admitted command");
        let result = runmat_execution::profiling::observe_command_future(
            receipt,
            self.invoke_executable_and_cleanup(unit, invocation, control),
        )
        .await;
        if let Some(admission) = profiling_root {
            self.observe_command_device_time_at_seal(&admission.root);
            let _ = admission.root.seal();
        }
        result
    }

    async fn invoke_executable_and_cleanup(
        &mut self,
        unit: &crate::ExecutableUnit,
        invocation: crate::ProcedureInvocation,
        control: &crate::InvocationControl,
    ) -> std::result::Result<runmat_runtime::execution::RootedValueSequence, RunError> {
        let mut result = self
            .invoke_executable_inner(unit, invocation, control)
            .await;
        if let Err(error) = self.lifecycle_host.service_safepoint(&self.runtime_context) {
            if result.is_ok() {
                result = Err(RunError::Runtime(error));
            } else {
                tracing::warn!(%error, "cleanup safepoint failed after an invocation error");
            }
        }
        let cleanup = self
            .lifecycle_host
            .drain_gracefully(&self.runtime_context)
            .await;
        match cleanup {
            Ok(()) => result,
            Err(error) if result.is_ok() => Err(RunError::Runtime(error)),
            Err(error) => {
                tracing::warn!(%error, "cleanup callback failed after an invocation error");
                result
            }
        }
    }

    async fn invoke_executable_inner(
        &mut self,
        unit: &crate::ExecutableUnit,
        invocation: crate::ProcedureInvocation,
        control: &crate::InvocationControl,
    ) -> std::result::Result<runmat_runtime::execution::RootedValueSequence, RunError> {
        let _admission = self.admit_execution(control.cancellation());
        ensure_invocation_allowed(control, &self.runtime_context)?;
        self.ensure_user_path_hydrated().await?;
        let runtime = self.execution_runtime_context(unit.revision().program_revision.clone());
        let runtime = unit
            .bind_runtime_sources(runtime)
            .map_err(|error| runtime_error("RunMat:ExecutableSourceIdentity", error))?;
        let runtime_bytecode = unit
            .runtime_bytecode(&runtime)
            .map_err(|error| runtime_error("RunMat:ExecutableSourceIdentity", error))?;
        let current_functions = runtime_bytecode
            .function_registry
            .functions
            .keys()
            .copied()
            .collect();
        let selected = super::source_evaluation::SelectedGraphPrimary::new(
            super::source_evaluation::SelectedGraphSource {
                unit: Rc::new(unit.clone()),
                primary_source: unit
                    .source_map()
                    .entries()
                    .first()
                    .map(|entry| entry.source_id),
                already_published: None,
            },
            &runtime,
            runtime_bytecode.function_registry.clone(),
            super::source_evaluation::SelectedGraphPublication::new(
                &runtime_bytecode.function_registry,
                &std::collections::HashSet::new(),
            ),
            super::source_evaluation::SelectedGraphSelection {
                current_functions,
                interop: runmat_types::InteropManifest::empty(),
                static_targets: Vec::new(),
            },
        )
        .map_err(|error| runtime_error("RunMat:ExecutableSourceIdentity", error))?;
        let runtime = runtime
            .without_selected_graph_program()
            .with_selected_graph_program(super::source_evaluation::SelectedGraphSnapshot::program(
                Rc::new(selected),
                &self.source_evaluator,
            ));
        runtime
            .scope(self.invoke_executable_in_context(
                unit,
                invocation,
                control,
                &runtime,
                &runtime_bytecode,
            ))
            .await
    }

    async fn invoke_executable_in_context(
        &mut self,
        unit: &crate::ExecutableUnit,
        invocation: crate::ProcedureInvocation,
        control: &crate::InvocationControl,
        runtime: &runmat_runtime::context::RuntimeContext,
        runtime_bytecode: &runmat_vm::Bytecode,
    ) -> std::result::Result<runmat_runtime::execution::RootedValueSequence, RunError> {
        let (_input_guard, _input_events) = self.install_execution_input();
        let _test_services = runmat_runtime::testing::install_test_services(
            crate::testing::runtime_adapter::services(
                self.compat_mode,
                self.project_handoff.clone(),
                #[cfg(not(target_arch = "wasm32"))]
                self.local_tasks.child_scope_factory(),
            ),
        );
        // Builtin class registration is session-owned. Register the testing
        // projection after entering this session's runtime context so
        // function-based TestCase receivers resolve inherited methods here,
        // rather than only in the caller thread's fallback registry.
        runmat_runtime::testing::ensure_testing_classes();
        ensure_invocation_allowed(control, runtime)?;
        self.stats.total_executions += 1;
        self.configure_executable_runtime();
        let _interrupt_guard =
            runmat_runtime::interrupt::replace_interrupt(Some(self.runtime_context.cancellation()));
        let _source_guard = runmat_runtime::source_context::replace_current_source_context(
            Some(&unit.source().relative_path),
            Some(&unit.source().text),
        );
        let primary_source = runtime
            .program_source_id(runmat_types::ProgramSourceId(0))
            .ok_or_else(|| {
                runtime_error(
                    "RunMat:ExecutableSourceIdentity",
                    "executable primary source is absent from the invocation binding",
                )
            })?;
        let _source_id_guard =
            runmat_runtime::source_context::replace_current_source_id(Some(primary_source));

        // Construct only the selected backend outside this poll frame. Each
        // leaf transfers its future into the existing admitted owner; guards
        // above remain active until that selected continuation completes.
        let selected = self.select_executable_continuation(
            unit,
            invocation,
            runtime,
            runtime_bytecode,
            #[cfg(not(target_arch = "wasm32"))]
            control.backend(),
        )?;
        let result = selected.await;
        let outputs = result.map_err(|mut error| {
            self.populate_callstack(&mut error, runtime);
            RunError::Runtime(error)
        })?;
        ensure_invocation_allowed(control, runtime)?;
        Ok(outputs)
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
        runtime: &runmat_runtime::context::RuntimeContext,
        runtime_bytecode: &runmat_vm::Bytecode,
    ) -> std::result::Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
        let mut variables = vec![Value::Num(0.0); runtime_bytecode.var_count];
        match runmat_vm::interpret_with_workspace_and_dynamic_admission_in_context(
            runtime_bytecode,
            &mut variables,
            Some(&unit.source().relative_path),
            runtime.clone(),
            None,
            Some(self.source_evaluator.dynamic_admission_port()),
        )
        .await
        .outcome?
        {
            runmat_vm::InterpreterOutcome::Completed(_) => {
                runmat_runtime::execution::RootedValueSequence::new(
                    runmat_value::ValueSequence::empty(),
                )
                .map_err(|error| error.into_runtime_error())
            }
        }
    }

    async fn invoke_procedure_interpreted(
        &mut self,
        _unit: &crate::ExecutableUnit,
        runtime: &runmat_runtime::context::RuntimeContext,
        runtime_bytecode: &runmat_vm::Bytecode,
        function: runmat_hir::FunctionId,
        arguments: Vec<Value>,
        requested_outputs: usize,
    ) -> std::result::Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
        runmat_vm::invoke_semantic_function_value_with_dynamic_admission_in_context(
            function.0,
            &arguments,
            requested_outputs,
            &runtime_bytecode.function_registry,
            runtime.clone(),
            None,
            self.source_evaluator.dynamic_admission_port(),
        )
        .await
    }
}

fn ensure_invocation_allowed(
    control: &crate::InvocationControl,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> std::result::Result<(), RunError> {
    if control.is_cancelled() || runtime.is_cancelled() {
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

fn executable_continuation_error(
    cause: runmat_runtime::context::host_allocation::HostAllocationError,
) -> RunError {
    use runmat_runtime::context::host_allocation::HostAllocationError;
    use runmat_runtime::context::host_effect::HostEffectAllocationError;

    let cancelled = matches!(
        cause,
        HostAllocationError::Cancelled
            | HostAllocationError::Resource(HostEffectAllocationError::Cancelled)
    );
    let (identifier, message) = if cancelled {
        (
            "RunMat:ExecutionCancelled".to_owned(),
            "executable invocation was cancelled",
        )
    } else {
        (
            format!("RunMat:{}", cause.runtime_identifier_suffix()),
            "executable invocation continuation admission failed",
        )
    };
    RunError::Runtime(
        build_runtime_error(message)
            .with_identifier(identifier)
            .with_source(cause)
            .build(),
    )
}

fn runtime_error(identifier: &'static str, message: impl Into<String>) -> RunError {
    RunError::Runtime(
        build_runtime_error(message)
            .with_identifier(identifier)
            .build(),
    )
}
