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
        runmat_vm::set_call_stack_limit(limit);
    }

    pub fn set_error_namespace(&mut self, namespace: impl Into<String>) {
        let namespace = namespace.into();
        let namespace = if namespace.trim().is_empty() {
            runmat_vm::DEFAULT_ERROR_NAMESPACE.to_string()
        } else {
            namespace
        };
        self.error_namespace = namespace.clone();
        self.runtime_context.set_error_namespace(namespace.clone());
        runmat_vm::set_error_namespace(&namespace);
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
