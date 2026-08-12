use super::*;

impl RunMatSession {
    /// Create a new session
    pub fn new() -> Result<Self> {
        Self::with_options(true, false) // JIT enabled, verbose disabled
    }

    /// Create a new session with specific options
    pub fn with_options(enable_jit: bool, verbose: bool) -> Result<Self> {
        Self::initialize(enable_jit, verbose)
    }

    fn initialize(enable_jit: bool, verbose: bool) -> Result<Self> {
        #[cfg(feature = "jit")]
        let jit_engine = if enable_jit {
            match TurbineEngine::new() {
                Ok(engine) => {
                    info!("JIT compiler initialized successfully");
                    Some(engine)
                }
                Err(e) => {
                    warn!("JIT compiler initialization failed: {e}, falling back to interpreter");
                    None
                }
            }
        } else {
            info!("JIT compiler disabled, using interpreter only");
            None
        };

        #[cfg(not(feature = "jit"))]
        if enable_jit {
            info!(
                "JIT support was requested but the 'jit' feature is disabled; running interpreter-only."
            );
        }

        let session = Self {
            #[cfg(feature = "jit")]
            jit_engine,
            verbose,
            stats: ExecutionStats::default(),
            variable_array: Vec::new(),
            workspace_bindings: HashMap::new(),
            workspace_values: HashMap::new(),
            abi_workspace_handle: crate::abi::WorkspaceHandle(Uuid::new_v4()),
            active_source_identity: None,
            function_registry: runmat_vm::FunctionRegistry::default(),
            next_semantic_function_id: 0,
            search_path: Arc::new(
                runmat_runtime::builtins::common::path_state::SearchPath::new(
                    runmat_runtime::builtins::common::path_state::current_path_string(),
                ),
            ),
            dynamic_function_cache: Arc::new(Mutex::new(HashMap::new())),
            project_handoff: None,
            source_pool: SourcePool::default(),
            interrupt_flag: Arc::new(AtomicBool::new(false)),
            execution_context: runmat_runtime::execution::InvocationExecutionContext::new(
                std::rc::Rc::new(runmat_runtime::execution::RuntimeExecutionService::new()),
            ),
            is_executing: false,
            async_input_handler: None,
            callstack_limit: runmat_vm::DEFAULT_CALLSTACK_LIMIT,
            error_namespace: runmat_vm::DEFAULT_ERROR_NAMESPACE.to_string(),
            active_source_name: "<repl>".to_string(),
            active_source_fullpath_name: None,
            telemetry_consent: true,
            telemetry_client_id: None,
            telemetry_platform: TelemetryPlatformInfo::default(),
            telemetry_sink: None,
            workspace_preview_tokens: HashMap::new(),
            workspace_version: 0,
            emit_fusion_plan: false,
            compat_mode: CompatMode::Matlab,
            top_level_await_enabled: true,
            dynamic_eval_enabled: true,
            format_mode: runmat_value::FormatMode::default(),
            diary_state: runmat_runtime::console::DiaryStateSnapshot::default(),
            pending_companion_source_discovery: None,
        };

        runmat_vm::set_call_stack_limit(session.callstack_limit);

        // Cache the shared plotting context (if a GPU provider is active) so the
        // runtime can wire zero-copy render paths without instantiating another
        // WebGPU device.
        #[cfg(any(target_arch = "wasm32", not(target_arch = "wasm32")))]
        {
            if let Err(err) =
                runmat_runtime::builtins::plotting::context::ensure_context_from_provider()
            {
                debug!("Plotting context unavailable during session init: {err}");
            }
        }

        Ok(session)
    }

    pub(crate) fn current_source_name(&self) -> &str {
        &self.active_source_name
    }

    pub(crate) fn current_source_fullpath_name(&self) -> Option<&str> {
        self.active_source_fullpath_name.as_deref()
    }
}
