use super::*;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::BTreeMap;
use std::rc::Rc;

#[cfg(not(target_arch = "wasm32"))]
mod ready_route;
mod selected_product;
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
    Cold(Option<crate::generic_native::placement::ReadyRouteSelection>),
    Native(
        std::result::Result<Box<crate::generic_native::RootedNativeExecution>, Box<RuntimeError>>,
        Option<crate::generic_native::placement::ReadyRouteSelection>,
    ),
}

#[cfg(not(target_arch = "wasm32"))]
struct NativeProcedureTarget<'a> {
    function: runmat_hir::FunctionId,
    name: &'a str,
}

struct ProcedureCall {
    arguments: runmat_runtime::call::BuiltinArguments,
    requested_outputs: usize,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
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
        call: ProcedureCall,
    ) -> std::result::Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
        let ProcedureCall {
            arguments,
            requested_outputs,
            lexical_environment,
        } = call;
        let published = self.generic_native_cache.resolve_or_compile(
            unit,
            preferred_function,
            crate::generic_native::capture_liveness_admission(runtime)?,
        )?;
        crate::generic_native::invoke(
            unit,
            crate::generic_native::NativeInvocation {
                published,
                osr: None,
                preferred_function,
                arguments,
                lexical_environment,
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
        interpreter_bytecode_unchanged: bool,
    ) -> std::result::Result<TieredInteractiveInvocation, RuntimeError> {
        let route_admission =
            if interpreter_bytecode_unchanged && self.generic_native_cache.has_retained_entries() {
                crate::generic_native::placement::ReadyRouteAdmission::prepare(runtime)?
            } else {
                None
            };
        let scheduled = self
            .generic_native_cache
            .resolve_or_schedule_with_placement(
                crate::generic_native::placement::ReadyInvocation {
                    unit,
                    preferred_function: None,
                    arguments: runmat_runtime::call::FunctionArgumentView::from_slice(&[]),
                    requested_outputs: 0,
                    profile: profile.map(|profile| profile.digest),
                },
                profile,
                runtime,
                route_admission.is_some(),
            )?;
        let published = scheduled.published;
        let ready_at_start = scheduled.ready_at_start;
        let osr = if published
            .as_ref()
            .is_some_and(|entry| matches!(entry.tier, runmat_jit::entry::PublishedTier::Generic))
        {
            self.generic_native_cache.resolve_or_schedule_osr(
                unit,
                None,
                profile,
                crate::generic_native::capture_liveness_admission(runtime)?,
            )?
        } else {
            None
        };
        let eligible = osr.is_none() && interpreter_bytecode_unchanged;
        let selected = self.select_ready_native_route(
            runtime,
            crate::generic_native::placement::ReadyInvocation {
                unit,
                preferred_function: None,
                arguments: runmat_runtime::call::FunctionArgumentView::from_slice(&[]),
                requested_outputs: 0,
                profile: profile.map(|profile| profile.digest),
            },
            published,
            eligible,
            ready_at_start,
            route_admission,
        )?;
        let published = selected.published;
        let placement = selected.receipt;
        let Some(published) = published else {
            return Ok(TieredInteractiveInvocation::Cold(placement));
        };
        self.stats.jit_compiled += 1;
        let execution = crate::generic_native::invoke(
            unit,
            crate::generic_native::NativeInvocation {
                published,
                osr,
                preferred_function: None,
                arguments: runmat_runtime::call::BuiltinArguments::from_borrowed(
                    &[],
                    Some(runtime),
                )?,
                lexical_environment: None,
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
            placement,
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
                .invoke_entrypoint_interpreted(unit, runtime, runtime_bytecode, 0)
                .await;
        }
        let profile = self.generic_native_cache.representation_profile(&[]);
        let started = std::time::Instant::now();
        let route_admission = if self.generic_native_cache.has_retained_entries() {
            crate::generic_native::placement::ReadyRouteAdmission::prepare(runtime)?
        } else {
            None
        };
        let scheduled = self
            .generic_native_cache
            .resolve_or_schedule_with_placement(
                crate::generic_native::placement::ReadyInvocation {
                    unit,
                    preferred_function: None,
                    arguments: runmat_runtime::call::FunctionArgumentView::from_slice(&[]),
                    requested_outputs: 0,
                    profile: profile.as_ref().map(|profile| profile.digest),
                },
                profile.as_ref(),
                runtime,
                route_admission.is_some(),
            )?;
        let published = scheduled.published;
        let ready_at_start = scheduled.ready_at_start;
        let osr = if published
            .as_ref()
            .is_some_and(|entry| matches!(entry.tier, runmat_jit::entry::PublishedTier::Generic))
        {
            self.generic_native_cache.resolve_or_schedule_osr(
                unit,
                None,
                profile.as_ref(),
                crate::generic_native::capture_liveness_admission(runtime)?,
            )?
        } else {
            None
        };
        // An OSR target is an additional ready product with its own route
        // identity. Until the complete pair is represented, retain the
        // existing tiering route rather than quote a one-product candidate.
        let eligible = osr.is_none();
        let selected = self.select_ready_native_route(
            runtime,
            crate::generic_native::placement::ReadyInvocation {
                unit,
                preferred_function: None,
                arguments: runmat_runtime::call::FunctionArgumentView::from_slice(&[]),
                requested_outputs: 0,
                profile: profile.as_ref().map(|profile| profile.digest),
            },
            published,
            eligible,
            ready_at_start,
            route_admission,
        )?;
        let published = selected.published;
        let placement = selected.receipt;
        let (result, backedges, osr_entry, vectorized_regions) = match published {
            Some(published) => {
                self.stats.jit_compiled += 1;
                let execution = crate::generic_native::invoke(
                    unit,
                    crate::generic_native::NativeInvocation {
                        published,
                        osr,
                        preferred_function: None,
                        arguments: runmat_runtime::call::BuiltinArguments::from_borrowed(
                            &[],
                            Some(runtime),
                        )?,
                        lexical_environment: None,
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
                    self.invoke_entrypoint_interpreted(unit, runtime, runtime_bytecode, 0)
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
        if let Some(selected) = placement {
            selected.observe(if runtime.is_cancelled() {
                runmat_execution::WholeWorkObservationOutcome::Cancelled
            } else if result.is_ok() {
                runmat_execution::WholeWorkObservationOutcome::Success
            } else {
                runmat_execution::WholeWorkObservationOutcome::OperationalFailure
            });
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
        call: ProcedureCall,
    ) -> std::result::Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
        let NativeProcedureTarget { function, name } = target;
        let ProcedureCall {
            arguments,
            requested_outputs,
            lexical_environment,
        } = call;
        if !self.native_tiering_enabled {
            return self
                .invoke_procedure_interpreted(
                    unit,
                    runtime,
                    runtime_bytecode,
                    function,
                    ProcedureCall {
                        arguments,
                        requested_outputs,
                        lexical_environment,
                    },
                )
                .await;
        }
        let profile = {
            // Specialization guards describe the callee's explicit inputs,
            // not its separately retained closure capture prefix.
            let retained = arguments.fork().into_function_arguments();
            self.generic_native_cache
                .representation_profile_from_values(retained.supplied().iter())
        };
        let started = std::time::Instant::now();
        let route_admission = if self.generic_native_cache.has_retained_entries() {
            crate::generic_native::placement::ReadyRouteAdmission::prepare(runtime)?
        } else {
            None
        };
        let scheduled = self
            .generic_native_cache
            .resolve_or_schedule_with_placement(
                crate::generic_native::placement::ReadyInvocation {
                    unit,
                    preferred_function: Some(name),
                    arguments: arguments.view(),
                    requested_outputs,
                    profile: profile.as_ref().map(|profile| profile.digest),
                },
                profile.as_ref(),
                runtime,
                route_admission.is_some(),
            )?;
        let published = scheduled.published;
        let ready_at_start = scheduled.ready_at_start;
        let osr = if published
            .as_ref()
            .is_some_and(|entry| matches!(entry.tier, runmat_jit::entry::PublishedTier::Generic))
        {
            self.generic_native_cache.resolve_or_schedule_osr(
                unit,
                Some(name),
                profile.as_ref(),
                crate::generic_native::capture_liveness_admission(runtime)?,
            )?
        } else {
            None
        };
        let eligible = osr.is_none();
        let selected = self.select_ready_native_route(
            runtime,
            crate::generic_native::placement::ReadyInvocation {
                unit,
                preferred_function: Some(name),
                arguments: arguments.view(),
                requested_outputs,
                profile: profile.as_ref().map(|profile| profile.digest),
            },
            published,
            eligible,
            ready_at_start,
            route_admission,
        )?;
        let published = selected.published;
        let placement = selected.receipt;
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
                        lexical_environment,
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
                        ProcedureCall {
                            arguments,
                            requested_outputs,
                            lexical_environment,
                        },
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
        if let Some(selected) = placement {
            selected.observe(if runtime.is_cancelled() {
                runmat_execution::WholeWorkObservationOutcome::Cancelled
            } else if result.is_ok() {
                runmat_execution::WholeWorkObservationOutcome::Success
            } else {
                runmat_execution::WholeWorkObservationOutcome::OperationalFailure
            });
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
        let coverage = runmat_runtime::coverage::CoverageSession::start(&self.runtime_context);
        let invocation = self.invoke_rooted_executable(unit, invocation, control);
        async move {
            let value = invocation.await?;
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
        self.invoke_rooted_executable(unit, invocation, control)
    }

    fn admit_invocation_arguments(
        invocation: crate::ProcedureInvocation,
        control: &crate::InvocationControl,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> std::result::Result<crate::executable::AdmittedProcedureInvocation, RunError> {
        ensure_invocation_allowed(control, runtime)?;
        let _scope = runtime.enter();
        invocation.admit(runtime).map_err(RunError::Runtime)
    }

    fn invoke_rooted_executable<'a>(
        &'a mut self,
        unit: &'a crate::ExecutableUnit,
        invocation: crate::ProcedureInvocation,
        control: &'a crate::InvocationControl,
    ) -> impl std::future::Future<Output = InvocationResult> + 'a {
        // Reentrant execution (including runtests) must not embed another
        // complete executable state in every caller's async frame. Pin that
        // state once, retaining the same arguments and roots without copying.
        // Capture the caller before this session's root binding can mask it.
        // Keep this owner through the outer receipt's publication checkpoint.
        let cancellation = control.cancellation();
        let admission = self.admit_execution(cancellation.as_ref());
        let original_cancellation = admission.token();
        let runtime = self
            .runtime_context
            .clone()
            .with_cancellation_token(admission.token());
        let invocation = Self::admit_invocation_arguments(invocation, control, &runtime);
        let admitted = invocation.and_then(|invocation| {
            runmat_runtime::execution::admit_executable_sequence_with_error(
                &runtime,
                executable_continuation_error,
                async move {
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
        async move {
            let result = match admitted {
                Ok(future) => future.await,
                Err(error) => Err(error),
            }
            .map_err(|error| present_executable_control_failure(error, &original_cancellation));
            drop(admission);
            result
        }
    }

    async fn invoke_executable_with_cleanup(
        &mut self,
        unit: &crate::ExecutableUnit,
        invocation: crate::executable::AdmittedProcedureInvocation,
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
        invocation: crate::executable::AdmittedProcedureInvocation,
        control: &crate::InvocationControl,
    ) -> std::result::Result<runmat_runtime::execution::RootedValueSequence, RunError> {
        let _input = self.install_lifecycle_input();
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
        invocation: crate::executable::AdmittedProcedureInvocation,
        control: &crate::InvocationControl,
    ) -> std::result::Result<runmat_runtime::execution::RootedValueSequence, RunError> {
        ensure_invocation_allowed(control, &self.runtime_context)?;
        self.ensure_user_path_hydrated().await?;
        let runtime = self.execution_runtime_context(unit.revision().program_revision.clone());
        let runtime = unit
            .bind_runtime_sources(runtime)
            .map_err(|error| runtime_error("RunMat:ExecutableSourceIdentity", error))?;
        let mut runtime_bytecode = unit
            .runtime_bytecode(&runtime)
            .map_err(|error| runtime_error("RunMat:ExecutableSourceIdentity", error))?;
        let selected = self.prepare_executable_graph(unit, &runtime, &mut runtime_bytecode)?;
        let selected = Rc::new(selected);
        let runtime = runtime
            .without_selected_graph_program()
            .with_selected_graph_program(super::source_evaluation::SelectedGraphSnapshot::program(
                Rc::clone(&selected),
                &self.source_evaluator,
            ));
        let runtime = super::source_evaluation::bind_function_resolvers(
            runtime,
            unit,
            Rc::new(runtime_bytecode.function_registry.clone()),
            selected.static_targets().to_vec(),
        )
        .map_err(|error| runtime_error("RunMat:ExecutableStaticImport", error))?;
        if let Some(host) = runtime.service_ports().host() {
            host.check_execution()?;
        }
        let result = runtime
            .scope(self.invoke_executable_in_context(
                unit,
                invocation,
                control,
                &runtime,
                &runtime_bytecode,
            ))
            .await;
        // Sticky host rejection cannot become successful execution through a language catch.
        if let Some(host) = runtime.service_ports().host() {
            host.check_execution()?;
        }
        result
    }

    async fn invoke_executable_in_context(
        &mut self,
        unit: &crate::ExecutableUnit,
        invocation: crate::executable::AdmittedProcedureInvocation,
        control: &crate::InvocationControl,
        runtime: &runmat_runtime::context::RuntimeContext,
        runtime_bytecode: &runmat_vm::Bytecode,
    ) -> std::result::Result<runmat_runtime::execution::RootedValueSequence, RunError> {
        let (_input_guard, _input_events) = self.install_execution_input();
        let _program_guard = self.program_service.enter(unit);
        let _dependency_guard = self
            .program_service
            .install_dependency_invoker(unit, runtime.clone());
        let _test_services = runmat_runtime::testing::install_test_services(
            crate::testing::runtime_adapter::services(
                self.compat_mode,
                self.project_handoff.clone(),
                #[cfg(not(target_arch = "wasm32"))]
                self.local_tasks.child_scope_factory(),
                #[cfg(not(target_arch = "wasm32"))]
                Some(self.mex_runtime.native_lane_child_factory()),
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
        requested_outputs: usize,
    ) -> std::result::Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
        debug_assert!(requested_outputs <= 1);
        // The VM allocates its dense storage. Supplying placeholders here
        // would declare every slot assigned before the script has run.
        match runmat_vm::interpret_workspace(
            runtime_bytecode,
            runmat_vm::InterpreterVars::from_owned(Vec::new()),
            Some(&unit.source().relative_path),
            runtime.clone(),
            None,
            Some(self.source_evaluator.dynamic_admission_port()),
        )
        .await
        .outcome?
        {
            runmat_vm::InterpreterOutcome::Completed(completion) => {
                if requested_outputs == 0 {
                    return runmat_runtime::execution::RootedValueSequence::new(
                        runmat_value::ValueSequence::empty(),
                    )
                    .map_err(|error| error.into_runtime_error());
                }
                // Dense VM storage is not an output contract. An explicit
                // script result is only the `ans` binding assigned by this
                // invocation, never the last allocated variable slot.
                let value = completion
                    .ans_slot
                    .filter(|slot| completion.assigned_slots.contains(slot))
                    .and_then(|slot| completion.values.get(slot).map(|value| value.clone()))
                    .ok_or_else(|| {
                        runmat_runtime::build_runtime_error(
                            "script did not produce the requested ans value",
                        )
                        .with_identifier("RunMat:ExecutableOutput")
                        .build()
                    })?;
                let sequence = runmat_value::ValueSequence::single(value)
                    .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?;
                runmat_runtime::execution::RootedValueSequence::new(sequence)
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
        call: ProcedureCall,
    ) -> std::result::Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
        let ProcedureCall {
            arguments,
            requested_outputs,
            lexical_environment,
        } = call;
        runmat_vm::invoke_semantic_function_with_rooted_arguments_and_dynamic_admission_in_context(
            function.0,
            arguments.into_function_arguments(),
            requested_outputs,
            &runtime_bytecode.function_registry,
            runtime.clone(),
            lexical_environment,
            self.source_evaluator.dynamic_admission_port(),
        )
        .await
    }
}

fn ensure_invocation_allowed(
    control: &crate::InvocationControl,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> std::result::Result<(), RunError> {
    let mut cancellation = runtime.cancellation();
    if let Some(control) = control.cancellation() {
        cancellation = cancellation.or(&control);
    }
    if let Some(parent) = runmat_runtime::interrupt::current_interrupt() {
        cancellation = cancellation.or(&parent);
    }
    match cancellation.stop_reason() {
        Some(runmat_execution::CancellationReason::Deadline) => Err(runtime_error(
            "RunMat:ExecutionDeadline",
            "executable invocation deadline elapsed",
        )),
        Some(_) => Err(runtime_error(
            "RunMat:ExecutionCancelled",
            "executable invocation was cancelled",
        )),
        None => Ok(()),
    }
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
    let mut builder = build_runtime_error(message).with_identifier(identifier);
    if cancelled {
        builder = builder.with_failure_origin(
            runmat_runtime::runtime_error::RuntimeFailureOrigin::ExecutionControlCancellation,
        );
    }
    RunError::Runtime(builder.with_source(cause).build())
}

/// Execution and allocation checkpoints carry a typed cancellation cause but
/// not its invocation-local reason. Present the admitted token's reason without
/// rewriting a language failure or dropping its retained diagnostic owner.
fn present_executable_control_failure(
    error: RunError,
    original: &runmat_execution::CancellationToken,
) -> RunError {
    let RunError::Runtime(cause) = error else {
        return error;
    };
    if let Some(reason) = original
        .stop_reason()
        .filter(|_| runmat_runtime::runtime_error::error_has_control_cancellation_origin(&cause))
    {
        let (identifier, message) = match reason {
            runmat_execution::CancellationReason::Deadline => (
                "RunMat:ExecutionDeadline",
                "executable invocation deadline elapsed",
            ),
            _ => (
                "RunMat:ExecutionCancelled",
                "executable invocation was cancelled",
            ),
        };
        RunError::Runtime(
            build_runtime_error(message)
                .with_identifier(identifier)
                .with_failure_origin(
                    runmat_runtime::runtime_error::RuntimeFailureOrigin::ExecutionControlCancellation,
                )
                .with_gpu_gather_retry(runmat_runtime::runtime_error::GpuGatherRetry::Never)
                .with_source(cause)
                .build(),
        )
    } else {
        RunError::Runtime(cause)
    }
}

fn runtime_error(identifier: &'static str, message: impl Into<String>) -> RunError {
    RunError::Runtime(
        build_runtime_error(message)
            .with_identifier(identifier)
            .build(),
    )
}

#[cfg(test)]
mod control_failure_tests {
    use super::*;

    #[test]
    fn expired_deadline_does_not_promote_language_failure_with_typed_cancelled_source() {
        let language = build_runtime_error("language evaluation failed")
            .with_identifier("RunMat:LanguageFailure")
            .with_source(runmat_runtime::execution::ExecutionServiceError::Cancelled)
            .build();
        let deadline = runmat_execution::CancellationToken::default().with_deadline(Some(0));
        let RunError::Runtime(presented) =
            present_executable_control_failure(RunError::Runtime(language), &deadline)
        else {
            panic!("runtime failure changed domains");
        };
        assert_eq!(presented.identifier(), Some("RunMat:LanguageFailure"));
        assert_eq!(presented.message(), "language evaluation failed");
    }
}
