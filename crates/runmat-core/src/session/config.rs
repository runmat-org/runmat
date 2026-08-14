use super::*;

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

    /// Replace the root execution service used by this session and every
    /// nested invocation compiled from it.
    pub fn install_execution_services(
        &mut self,
        services: std::rc::Rc<dyn runmat_runtime::execution::RuntimeExecutionServices>,
    ) {
        self.runtime_context = self.runtime_context.clone().with_execution(services);
    }

    /// Install the host's parallel/resource authority. Placement consumes the
    /// service's current allocation lease; it does not reserve or schedule
    /// work independently of that authority.
    pub fn install_parallel_service(
        &mut self,
        service: std::rc::Rc<dyn runmat_runtime::context::RuntimeParallelService>,
    ) {
        let ports = self
            .runtime_context
            .service_ports()
            .clone()
            .with_parallel(service);
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
        self.interrupt_flag.store(true, Ordering::Relaxed);
        self.runtime_context
            .execution()
            .drain_scope(runmat_execution::CancellationReason::User);
    }

    /// Shared interrupt flag used by the VM to implement cooperative cancellation.
    pub fn interrupt_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.interrupt_flag)
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

    /// Set the language compatibility mode (`matlab` or `strict`).
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
        info!(
            jit = %if self.has_jit() { "available" } else { "disabled/failed" },
            verbose = self.verbose,
            total_executions = self.stats.total_executions,
            jit_compiled = self.stats.jit_compiled,
            interpreter_fallback = self.stats.interpreter_fallback,
            native_osr_transfers = self.stats.native_osr_transfers,
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
            workspace_vars = self.workspace_values.len(),
            "RunMat Session Status"
        );
    }

    #[cfg(feature = "jit")]
    pub(crate) fn has_jit(&self) -> bool {
        self.jit_engine.is_some()
    }

    #[cfg(not(feature = "jit"))]
    pub(crate) fn has_jit(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use runmat_runtime::context::{
        ParallelCapability, RuntimeParallelResources, RuntimeParallelService,
    };

    use super::RunMatSession;

    struct FixedAllocation;

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
}
