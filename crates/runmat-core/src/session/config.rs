use super::*;
mod foreign;

impl RunMatSession {
    pub(crate) fn runtime_context(&self) -> &runmat_runtime::context::RuntimeContext {
        &self.runtime_context
    }

    pub(crate) fn configure_runtime_context(&self) {
        self.runtime_context
            .set_language_mode(runtime_language_mode(self.compat_mode));
        self.runtime_context
            .set_runmat_extensions_enabled(self.compat_mode.allows_runmat_extensions());
        self.runtime_context
            .set_top_level_await_enabled(self.top_level_await_enabled);
        self.runtime_context
            .set_dynamic_eval_enabled(self.dynamic_eval_enabled);
    }

    /// Install an exact environment owned by this session and its nested invocations.
    /// Missing names remain absent; this never changes the worker process environment.
    pub fn install_environment_snapshot(
        &mut self,
        entries: Vec<(String, String)>,
    ) -> std::io::Result<()> {
        self.runtime_context.install_environment_snapshot(entries)?;
        let path = self
            .environment_snapshot()
            .into_iter()
            .find_map(|(name, value)| (name == "RUNMAT_PATH").then_some(value))
            .unwrap_or_default();
        self.runtime_context = self
            .runtime_context
            .clone()
            .with_search_path(std::sync::Arc::new(
                runmat_runtime::builtins::common::path_state::SearchPath::new(path),
            ));
        self.dynamic_function_cache.clear();
        Ok(())
    }

    pub fn environment_snapshot(&self) -> Vec<(String, String)> {
        self.runtime_context.environment_snapshot()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn install_native_interface_artifact(
        &self,
        library_path: &std::path::Path,
        manifest_path: &std::path::Path,
    ) -> Result<String, RuntimeError> {
        self.native_ffi_adapter
            .install_prepared_artifact(library_path, manifest_path)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn install_java_project_artifacts(
        &self,
        artifacts: &[(runmat_java::JavaArtifactIdentity, std::path::PathBuf)],
    ) -> Result<(), RuntimeError> {
        self.java_adapter.install_project_artifacts(artifacts)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn install_python_artifact_bundle(
        &self,
        bundle: &runmat_python::PythonArtifactBundle,
    ) -> Result<(), RuntimeError> {
        self.python_adapter.install_artifact_bundle(bundle)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn install_portable_python_artifact_bundle(
        &self,
        bundle: &runmat_python::PythonArtifactBundle,
    ) -> Result<(), RuntimeError> {
        self.python_adapter.install_portable_artifact_bundle(bundle)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn install_mex_artifact(
        &self,
        module_path: &std::path::Path,
    ) -> Result<String, RuntimeError> {
        self.mex_runtime.install_artifact(module_path)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn admit_interop_manifest(
        &self,
        manifest: &runmat_types::InteropManifest,
    ) -> Result<runmat_runtime::foreign::InteropAdmissionPlan, RuntimeError> {
        self.foreign_runtime.admit(manifest)
    }

    pub fn execution_runtime_context(
        &self,
        revision: runmat_execution::ProgramRevision,
    ) -> runmat_runtime::context::RuntimeContext {
        self.configure_runtime_context();
        self.configure_dynamic_function_services();
        self.runtime_context
            .clone()
            .with_cancellation_token(self.runtime_context.cancellation())
            .with_program_revision(Some(revision))
    }

    /// Replace the root execution service used by this session and every
    /// nested invocation compiled from it.
    pub fn install_execution_services(
        &mut self,
        services: std::rc::Rc<dyn runmat_runtime::execution::RuntimeExecutionServices>,
    ) {
        self.runtime_context = self.runtime_context.clone().with_execution(services);
    }

    /// Install a session-local async FEA host. Ordinary scripts do not contact it.
    pub fn install_fea_run_admission_service(
        &mut self,
        service: std::rc::Rc<dyn runmat_runtime::analysis::fea_run_admission::AsyncFeaRunAdmission>,
    ) {
        let ports = self
            .runtime_context
            .service_ports()
            .clone()
            .with_fea_run_admission(service);
        self.runtime_context = self.runtime_context.clone().with_service_ports(ports);
    }

    /// Install an exact invocation-owned builtin worker port supplied by the
    /// host. Ordinary serial sessions do not gain worker authority.
    pub fn install_owned_builtin_work_service(
        &mut self,
        service: std::rc::Rc<dyn runmat_runtime::execution::RuntimeOwnedBuiltinWorkService>,
    ) {
        let ports = self
            .runtime_context
            .service_ports()
            .clone()
            .with_owned_builtin_work(service);
        self.runtime_context = self.runtime_context.clone().with_service_ports(ports);
    }

    /// Install the host's parallel/resource authority. Placement consumes the
    /// service's current allocation lease; it does not reserve or schedule
    /// work independently of that authority.
    pub fn install_parallel_service(
        &mut self,
        service: std::rc::Rc<dyn runmat_runtime::context::RuntimeParallelService>,
    ) {
        let old_managed = self
            .runtime_context
            .service_ports()
            .parallel()
            .and_then(|owner| owner.managed_host_effect_ledger());
        let next_managed = service.managed_host_effect_ledger();
        if let Some(old) = old_managed {
            let same_epoch_owner = next_managed.as_ref().is_some_and(|next| {
                old.same_owner(next) && old.snapshot().epoch == service.placement_resources().epoch
            });
            if !same_epoch_owner {
                // The ledger may be shared with another session. Invalidate
                // only this session's outstanding staged calls; existing
                // returned Values retain their global charges until drop.
                self.runtime_context
                    .invalidate_host_effect_session_authority();
            }
        }
        let ports = self
            .runtime_context
            .service_ports()
            .clone()
            .with_parallel(service);
        self.runtime_context = self.runtime_context.clone().with_service_ports(ports);
    }

    /// Install the host that executes admitted generated analysis inputs.
    pub fn install_analysis_service(
        &mut self,
        service: std::rc::Rc<dyn runmat_runtime::context::RuntimeAnalysisService>,
    ) {
        let ports = self
            .runtime_context
            .service_ports()
            .clone()
            .with_analysis(service);
        self.runtime_context = self.runtime_context.clone().with_service_ports(ports);
    }

    /// Preallocate finite in-flight placement observation slots on the same
    /// managed ledger as the installed parallel service. This does not create
    /// a host-effect resource certificate or change ordinary unbounded calls.
    pub fn install_managed_host_effect_placement(
        &mut self,
        capacity: runmat_accelerate::placement::ManagedPlacementCapacity,
    ) -> Result<(), RuntimeError> {
        let parallel = self
            .runtime_context
            .service_ports()
            .parallel()
            .ok_or_else(|| managed_placement_error("parallel allocation service is absent"))?;
        let ledger = parallel.managed_host_effect_ledger().ok_or_else(|| {
            managed_placement_error("parallel service has no managed allocation owner")
        })?;
        let before = ledger.snapshot();
        if !before.current || before.epoch != parallel.placement_resources().epoch {
            return Err(managed_placement_error(
                "parallel allocation epoch is stale",
            ));
        }
        let replacement = std::rc::Rc::new(
            runmat_accelerate::placement::PlacementSession::new_with_managed_observations(
                runmat_accelerate::placement::PlacementSessionConfig::default(),
                &ledger,
                capacity,
            )
            .map_err(|source| {
                runmat_runtime::build_runtime_error(
                    "managed placement backing could not be reserved",
                )
                .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
                .with_source(source)
                .build()
            })?,
        );
        let profile = self.placement_session.profile_snapshot();
        replacement.restore_profile(profile).map_err(|source| {
            runmat_runtime::build_runtime_error("placement profile could not be transferred")
                .with_identifier(source.code)
                .with_source(source)
                .build()
        })?;
        let after = ledger.snapshot();
        if !after.current
            || after.epoch != before.epoch
            || parallel.placement_resources().epoch != after.epoch
        {
            return Err(managed_placement_error(
                "parallel allocation owner changed during setup",
            ));
        }
        self.placement_session
            .invalidate(runmat_execution::PlacementInvalidation::All);
        let ports = self
            .runtime_context
            .service_ports()
            .clone()
            .with_placement(replacement.clone());
        self.runtime_context = self.runtime_context.clone().with_service_ports(ports);
        self.placement_session = replacement;
        Ok(())
    }

    /// Install one host-owned complete-call resource evidence service. The
    /// service must use the same managed allocation authority exposed by the
    /// installed parallel port; this does not create a separate quota.
    pub fn install_host_effect_resource_service(
        &mut self,
        service: std::rc::Rc<dyn runmat_runtime::context::RuntimeHostEffectResourceService>,
    ) {
        let ports = self
            .runtime_context
            .service_ports()
            .clone()
            .with_host_effect_resources(service);
        self.runtime_context = self.runtime_context.clone().with_service_ports(ports);
    }

    /// Return the bounded, portable feedback profile owned by this session.
    /// The profile contains operation/candidate identities, digests, and
    /// aggregate counts/timings only; persistence remains an explicit host
    /// decision.
    pub fn placement_profile_snapshot(
        &self,
    ) -> runmat_accelerate::placement::PlacementProfileSnapshot {
        self.placement_session.profile_snapshot()
    }

    /// Restore a compatible bounded placement profile into this session.
    pub fn restore_placement_profile(
        &self,
        profile: runmat_accelerate::placement::PlacementProfileSnapshot,
    ) -> Result<(), runmat_accelerate::placement::PlacementPlanError> {
        self.placement_session.restore_profile(profile)
    }

    /// Install an async stdin handler (Phase 2). This is the preferred input path for
    /// poll-driven execution (`ExecuteFuture`).
    ///
    /// The handler is invoked when `input()` / `pause()` needs a line or keypress, and the
    /// returned future is awaited by the runtime.
    pub fn install_async_input_handler<F, Fut>(&mut self, handler: F)
    where
        F: Fn(InputRequest) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<InputResponse, String>> + 'static,
    {
        self.async_input_handler = Some(Arc::new(move |req: InputRequest| {
            let fut = handler(req);
            Box::pin(fut)
        }));
    }

    pub fn clear_async_input_handler(&mut self) {
        self.async_input_handler = None;
    }

    pub fn telemetry_consent(&self) -> bool {
        self.telemetry_consent
    }

    pub fn set_telemetry_consent(&mut self, consent: bool) {
        self.telemetry_consent = consent;
    }

    pub fn telemetry_client_id(&self) -> Option<&str> {
        self.telemetry_client_id.as_deref()
    }

    pub fn set_telemetry_client_id(&mut self, cid: Option<String>) {
        self.telemetry_client_id = cid;
    }

    /// Request cooperative cancellation for the currently running execution.
    pub fn cancel_execution(&self) {
        self.cancellation
            .handle()
            .cancel(runmat_execution::CancellationReason::User);
        self.runtime_context
            .execution()
            .drain_scope(runmat_execution::CancellationReason::User);
    }

    /// Cancellation authority for the current or next admitted execution.
    /// Obtain a fresh handle for each top-level request. A retained handle can
    /// never cancel a later execution generation, even after this request ends.
    pub fn interrupt_handle(&self) -> runmat_execution::cancellation::CancellationSource {
        self.cancellation.handle()
    }

    /// Get execution statistics
    pub fn stats(&self) -> &ExecutionStats {
        &self.stats
    }

    /// Reset execution statistics
    pub fn reset_stats(&mut self) {
        self.stats = ExecutionStats::default();
    }

    /// Control whether fusion plan snapshots are emitted in [`crate::abi::ExecutionOutcome`].
    pub fn set_emit_fusion_plan(&mut self, enabled: bool) {
        self.emit_fusion_plan = enabled;
    }

    /// Return the active language compatibility mode.
    pub fn compat_mode(&self) -> CompatMode {
        self.compat_mode
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn execution_host_inventory(
        &self,
        trust_tier: runmat_execution::security::ExecutionTrustTier,
    ) -> Result<runmat_execution::host::ExecutionHostInventory> {
        let environment = crate::program_environment(self.compat_mode);
        let mut adapters = self.foreign_runtime.execution_host_inventory();
        adapters.sort_by(|left, right| left.adapter.cmp(&right.adapter));
        let inventory = runmat_execution::host::ExecutionHostInventory {
            schema_version: runmat_execution::host::EXECUTION_HOST_SCHEMA_VERSION,
            semantic_schema: environment.semantic_schema,
            compiler_schema: environment.compiler_schema,
            runtime_fingerprint: environment.runtime_fingerprint,
            catalog_fingerprint: environment.catalog_fingerprint,
            compatibility_modes: [
                runmat_execution::LanguageCompatibilityMode::RunMat,
                runmat_execution::LanguageCompatibilityMode::Matlab,
                runmat_execution::LanguageCompatibilityMode::Strict,
            ]
            .into_iter()
            .collect(),
            target: runmat_execution::host::ExecutionHostTarget::Native(
                runmat_native_codegen::NativeTarget::current()
                    .execution_identity()
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?,
            ),
            capabilities: runmat_types::CapabilitySet(
                [
                    runmat_types::CapabilityRequirement::HostRuntime,
                    runmat_types::CapabilityRequirement::Filesystem,
                    runmat_types::CapabilityRequirement::Network,
                    runmat_types::CapabilityRequirement::UserInterface,
                    runmat_types::CapabilityRequirement::Accelerator,
                    runmat_types::CapabilityRequirement::NativeCode,
                    runmat_types::CapabilityRequirement::ForeignRuntime,
                    runmat_types::CapabilityRequirement::ParallelRuntime,
                    runmat_types::CapabilityRequirement::DistributedRuntime,
                ]
                .into_iter()
                .collect(),
            ),
            adapters,
            process_stack_available: true,
            host_bridge_available: false,
            trust_tier,
        };
        inventory
            .validate()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        Ok(inventory)
    }

    /// Set the language compatibility mode (`runmat`, `matlab`, or `strict`).
    pub fn set_compat_mode(&mut self, mode: CompatMode) {
        self.compat_mode = mode;
        self.runtime_context
            .set_language_mode(runtime_language_mode(mode));
        self.runtime_context
            .set_runmat_extensions_enabled(mode.allows_runmat_extensions());
    }

    pub fn set_callstack_limit(&mut self, limit: usize) {
        self.callstack_limit = limit;
        self.runtime_context.set_callstack_limit(limit);
    }

    pub fn set_error_namespace(&mut self, namespace: impl Into<String>) {
        let namespace = namespace.into();
        let namespace = if namespace.trim().is_empty() {
            runmat_runtime::context::DEFAULT_ERROR_NAMESPACE.to_string()
        } else {
            namespace
        };
        self.error_namespace = namespace.clone();
        self.runtime_context.set_error_namespace(namespace.clone());
        runmat_hir::set_error_namespace(&namespace);
    }

    /// Configure garbage collector
    pub fn configure_gc(&self, config: GcConfig) -> Result<()> {
        gc_configure(config)
            .map_err(|e| anyhow::anyhow!("Failed to configure garbage collector: {}", e))
    }

    /// Get GC statistics
    pub fn gc_stats(&self) -> runmat_gc::GcStats {
        gc_stats()
    }

    /// Show detailed system information
    pub fn show_system_info(&self) {
        let gc_stats = self.gc_stats();
        let workspace_vars = match self.base_workspace.snapshot() {
            Ok(snapshot) => snapshot.len(),
            Err(error) => {
                tracing::warn!(%error, "could not inspect base workspace for session status");
                0
            }
        };
        info!(
            jit = %if self.has_jit() { "available" } else { "disabled/failed" },
            verbose = self.verbose,
            total_executions = self.stats.total_executions,
            jit_compiled = self.stats.jit_compiled,
            interpreter_fallback = self.stats.interpreter_fallback,
            native_osr_transfers = self.stats.native_osr_transfers,
            vectorized_native_regions = self.stats.vectorized_native_regions,
            avg_time_ms = self.stats.average_execution_time_ms,
            total_allocations = gc_stats
                .total_allocations
                .load(std::sync::atomic::Ordering::Relaxed),
            minor_collections = gc_stats
                .minor_collections
                .load(std::sync::atomic::Ordering::Relaxed),
            major_collections = gc_stats
                .major_collections
                .load(std::sync::atomic::Ordering::Relaxed),
            current_memory_mb = gc_stats
                .current_memory_usage
                .load(std::sync::atomic::Ordering::Relaxed) as f64
                / 1024.0
                / 1024.0,
            workspace_vars,
            "RunMat Session Status"
        );
    }

    /// Complete foreign-extension lifecycle while the originating runtime
    /// context and callback services are still available.
    pub async fn shutdown_foreign_runtime(&mut self) -> Result<(), RuntimeError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.source_evaluator
                .retire_selected_foreign_hosts(&self.runtime_context, &self.local_tasks)
                .await?;
            // Keep the canonical session MEX callback scope and retirement
            // ordering after selected hosts have retired. A retained MEX
            // callback must not drop native services or their byte roots.
            let mex_retirement = self
                .local_tasks
                .participate_foreground(
                    self.mex_runtime
                        .shutdown_gracefully_retirement(self.runtime_context.clone()),
                )
                .await?;
            match mex_retirement {
                runmat_runtime::foreign::MexSessionRetirement::Retained(error) => {
                    return Err(error);
                }
                runmat_runtime::foreign::MexSessionRetirement::Retired(mex_result) => {
                    let native_result = self.native_ffi_adapter.shutdown().await;
                    mex_result?;
                    native_result?;
                }
            }
        }
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn has_jit(&self) -> bool {
        self.native_tiering_enabled
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn has_jit(&self) -> bool {
        false
    }
}

fn managed_placement_error(message: &'static str) -> RuntimeError {
    runmat_runtime::build_runtime_error(message)
        .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
        .build()
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use runmat_runtime::context::host_effect::ManagedHostEffectLedger;
    use runmat_runtime::context::{
        ParallelCapability, RuntimeHostEffectResourceService, RuntimeParallelResources,
        RuntimeParallelService,
    };

    use super::RunMatSession;

    struct FixedAllocation;

    struct ManagedAllocation(ManagedHostEffectLedger);

    struct NoEvidence;

    impl RuntimeHostEffectResourceService for NoEvidence {
        fn admission(
            &self,
            _: &runmat_runtime::context::host_effect::HostEffectRequest<'_>,
            _: &runmat_runtime::context::RuntimeContext,
        ) -> Option<runmat_runtime::context::host_effect::HostEffectResourceAdmission> {
            None
        }
    }

    impl RuntimeParallelService for ManagedAllocation {
        fn supports(&self, _capability: ParallelCapability) -> bool {
            false
        }

        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..RuntimeParallelResources::default()
            }
        }

        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }

    impl RuntimeParallelService for FixedAllocation {
        fn supports(&self, _capability: ParallelCapability) -> bool {
            false
        }

        fn placement_resources(&self) -> RuntimeParallelResources {
            RuntimeParallelResources {
                cpu_millicores_available: 250,
                memory_available_bytes: Some(4_096),
                epoch: 7,
            }
        }
    }

    #[test]
    fn session_exposes_explicit_profile_and_scheduler_composition() {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        assert!(session.placement_profile_snapshot().feedback.is_empty());

        session.install_parallel_service(Rc::new(FixedAllocation));
        let resources = session
            .runtime_context()
            .service_ports()
            .parallel()
            .unwrap()
            .placement_resources();
        assert_eq!(resources.cpu_millicores_available, 250);
        assert_eq!(resources.memory_available_bytes, Some(4_096));
        assert_eq!(resources.epoch, 7);
    }

    #[test]
    fn managed_placement_setup_uses_installed_parallel_ledger_and_is_failure_atomic() {
        let ledger = ManagedHostEffectLedger::new_with_epoch(16 * 1024, 9);
        let mut first = RunMatSession::with_options(false, false).unwrap();
        let mut second = RunMatSession::with_options(false, false).unwrap();
        first.install_parallel_service(Rc::new(ManagedAllocation(ledger.clone())));
        second.install_parallel_service(Rc::new(ManagedAllocation(ledger.clone())));
        let capacity = runmat_accelerate::placement::ManagedPlacementCapacity {
            in_flight_slots: 1,
            feedback_series: 1,
            signature_bytes_per_series: 1024,
        };
        first
            .install_managed_host_effect_placement(capacity)
            .expect("one preallocated slot on first session");
        let charged = ledger.snapshot().live_reserved_bytes;
        assert!(charged > 0);
        second
            .install_managed_host_effect_placement(capacity)
            .expect("independent session shares the ledger, not slot identity");
        assert_eq!(ledger.snapshot().live_reserved_bytes, charged * 2);
        let prior = first.placement_session.clone();
        let error = first
            .install_managed_host_effect_placement(
                runmat_accelerate::placement::ManagedPlacementCapacity {
                    in_flight_slots: usize::MAX,
                    ..capacity
                },
            )
            .expect_err("unrepresentable backing must not replace placement");
        assert_eq!(
            error.identifier(),
            Some("RunMat:Placement:InsufficientResourceEvidence")
        );
        assert!(Rc::ptr_eq(&prior, &first.placement_session));
        assert_eq!(ledger.snapshot().live_reserved_bytes, charged * 2);
        drop(prior);
        drop(first);
        assert_eq!(ledger.snapshot().live_reserved_bytes, charged);
        drop(second);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }

    #[test]
    fn host_resource_installation_preserves_parallel_and_placement_ports() {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let ledger = ManagedHostEffectLedger::new_with_epoch(16, 5);
        let parallel: Rc<dyn RuntimeParallelService> = Rc::new(ManagedAllocation(ledger.clone()));
        session.install_parallel_service(parallel.clone());
        let previous_placement = session
            .runtime_context()
            .service_ports()
            .placement()
            .cloned();
        let resources: Rc<dyn RuntimeHostEffectResourceService> = Rc::new(NoEvidence);
        session.install_host_effect_resource_service(resources.clone());
        let ports = session.runtime_context().service_ports();
        assert!(Rc::ptr_eq(ports.parallel().unwrap(), &parallel));
        assert!(Rc::ptr_eq(
            ports.host_effect_resources().unwrap(),
            &resources
        ));
        assert_eq!(
            ports
                .parallel()
                .unwrap()
                .managed_host_effect_ledger()
                .unwrap()
                .snapshot()
                .capacity_bytes,
            16
        );
        if let Some(previous) = previous_placement {
            assert!(Rc::ptr_eq(ports.placement().unwrap(), &previous));
        }
    }

    #[test]
    fn rebind_preserves_same_managed_owner_and_invalidates_only_this_session() {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let first = ManagedHostEffectLedger::new_with_epoch(16, 5);
        session.install_parallel_service(Rc::new(ManagedAllocation(first.clone())));
        let claim = first.reserve_stage(8).expect("original session owner");
        session.install_parallel_service(Rc::new(ManagedAllocation(first.clone())));
        assert!(first.snapshot().current);
        assert_eq!(first.snapshot().free_bytes(), 8);

        let replacement = ManagedHostEffectLedger::new_with_epoch(16, 6);
        session.install_parallel_service(Rc::new(ManagedAllocation(replacement.clone())));
        assert!(first.snapshot().current);
        assert_eq!(first.snapshot().live_reserved_bytes, 8);
        assert!(first.reserve_stage(1).is_ok());
        assert!(replacement.reserve_stage(16).is_ok());
        drop(claim);
        assert_eq!(first.snapshot().live_reserved_bytes, 0);
    }

    #[test]
    fn one_session_rebind_does_not_revoke_another_sessions_shared_ledger() {
        let shared = ManagedHostEffectLedger::new_with_epoch(16, 5);
        let mut first = RunMatSession::with_options(false, false).unwrap();
        let mut second = RunMatSession::with_options(false, false).unwrap();
        first.install_parallel_service(Rc::new(ManagedAllocation(shared.clone())));
        second.install_parallel_service(Rc::new(ManagedAllocation(shared.clone())));
        let second_claim = shared.reserve_stage(8).expect("other session claim");

        first.install_parallel_service(Rc::new(ManagedAllocation(
            ManagedHostEffectLedger::new_with_epoch(16, 6),
        )));
        assert!(shared.snapshot().current);
        assert_eq!(
            second
                .runtime_context()
                .service_ports()
                .parallel()
                .expect("second session owner")
                .placement_resources()
                .memory_available_bytes,
            Some(8)
        );
        let following = shared.reserve_stage(8).expect("second issuer remains live");
        drop(following);
        drop(second_claim);
        assert_eq!(shared.snapshot().live_reserved_bytes, 0);
    }
}
