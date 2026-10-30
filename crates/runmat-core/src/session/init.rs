use super::*;
mod child_settings;

impl RunMatSession {
    /// Create a new session
    pub fn new() -> Result<Self> {
        Self::with_options(true, false) // JIT enabled, verbose disabled
    }

    /// Create a new session with specific options
    pub fn with_options(enable_jit: bool, verbose: bool) -> Result<Self> {
        Self::initialize(
            enable_jit,
            verbose,
            #[cfg(not(target_arch = "wasm32"))]
            std::rc::Rc::new(SessionLocalTaskSupervisor::new()),
            None,
            #[cfg(not(target_arch = "wasm32"))]
            None,
            None,
        )
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn with_options_on_parent_driver(
        enable_jit: bool,
        verbose: bool,
        parent: &runmat_runtime::context::RuntimeLocalTaskChildFactory,
        parent_context: &runmat_runtime::context::RuntimeContext,
    ) -> Result<Self> {
        Self::with_options_on_parent_driver_with_mex(
            enable_jit,
            verbose,
            parent,
            parent_context,
            None,
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn with_options_on_parent_driver_with_mex(
        enable_jit: bool,
        verbose: bool,
        parent: &runmat_runtime::context::RuntimeLocalTaskChildFactory,
        parent_context: &runmat_runtime::context::RuntimeContext,
        parent_mex_lane: Option<runmat_runtime::foreign::MexNativeLaneChildFactory>,
    ) -> Result<Self> {
        Self::initialize(
            enable_jit,
            verbose,
            std::rc::Rc::new(parent.create()?),
            Some(parent_context),
            parent_mex_lane,
            None,
        )
    }

    /// Create a session with exactly the supplied environment. No ambient variables or
    /// source search path are collected before the frozen context is installed.
    pub fn with_frozen_environment(
        enable_jit: bool,
        verbose: bool,
        entries: Vec<(String, String)>,
    ) -> Result<Self> {
        Self::initialize(
            enable_jit,
            verbose,
            #[cfg(not(target_arch = "wasm32"))]
            std::rc::Rc::new(SessionLocalTaskSupervisor::new()),
            None,
            #[cfg(not(target_arch = "wasm32"))]
            None,
            Some(entries),
        )
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn with_options_for_test_child(
        enable_jit: bool,
        verbose: bool,
        parent_context: &runmat_runtime::context::RuntimeContext,
    ) -> Result<Self> {
        Self::initialize(enable_jit, verbose, Some(parent_context), None)
    }

    fn initialize(
        enable_jit: bool,
        verbose: bool,
        #[cfg(not(target_arch = "wasm32"))] local_tasks: std::rc::Rc<SessionLocalTaskSupervisor>,
        parent_context: Option<&runmat_runtime::context::RuntimeContext>,
        #[cfg(not(target_arch = "wasm32"))] parent_mex_lane: Option<
            runmat_runtime::foreign::MexNativeLaneChildFactory,
        >,
        environment: Option<Vec<(String, String)>>,
    ) -> Result<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        let native_tiering_enabled = enable_jit && cfg!(feature = "jit");
        #[cfg(all(not(target_arch = "wasm32"), not(feature = "jit")))]
        if enable_jit {
            info!("JIT support was requested but the 'jit' feature is disabled; running interpreter-only.");
        }
        #[cfg(target_arch = "wasm32")]
        let _ = enable_jit;

        let cancellation = cancellation::SessionCancellation::default();
        let startup_user_path_authority =
            runmat_runtime::builtins::common::path_state::CapturedUserPathAuthority::capture()?;
        let search_path =
            Arc::new(runmat_runtime::builtins::common::path_state::SearchPath::new(String::new()));
        let runtime_context = runmat_runtime::context::RuntimeContext::with_cancellation(
            std::rc::Rc::new(runmat_runtime::execution::RuntimeExecutionService::new()),
            cancellation.handle().token(),
        )
        .with_search_path(search_path);
        // A frozen environment replaces ambient variables before any path is derived.
        let startup_environment_path = match environment {
            Some(entries) => {
                runtime_context.install_environment_snapshot(entries)?;
                runtime_context
                    .environment_snapshot()
                    .into_iter()
                    .find_map(|(key, value)| (key == "RUNMAT_PATH").then_some(value))
                    .unwrap_or_default()
            }
            None => runmat_runtime::builtins::common::path_state::environment_path_string(),
        };
        let inherited_command_profiling = parent_context
            .and_then(runmat_runtime::context::RuntimeContext::command_profiling)
            .cloned();
        let runtime_context =
            if let Some(policy) = parent_context.and_then(|parent| parent.acceleration_policy()) {
                runtime_context.with_acceleration_policy(policy)
            } else {
                runtime_context
            };
        let runtime_context = if let Some(origin) = inherited_command_profiling.as_ref() {
            runtime_context.with_command_profiling(origin.clone())
        } else {
            runtime_context
        };
        #[cfg(not(target_arch = "wasm32"))]
        let runtime_context = if let Some(parent) = parent_context {
            runtime_context
                .try_inherit_foreign_callback_lineage_for_child(parent)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "child session callback lineage requires independent Runtime owners"
                    )
                })?
        } else {
            runtime_context
        };
        let placement = std::rc::Rc::new(runmat_accelerate::placement::PlacementSession::default());
        let parallel_services = crate::parallel_runtime::CoreParallelServices::new();
        #[cfg(target_arch = "wasm32")]
        let local_tasks = std::rc::Rc::new(SessionLocalTaskSupervisor::new());
        let base_workspace = runmat_runtime::workspace::BaseWorkspaceOwner::new();
        let source_evaluator = super::source_evaluation::SessionSourceEvaluator::new(
            base_workspace.clone(),
            runtime_context.local_identity(),
        );
        let base_workspace_service = std::rc::Rc::new(
            runmat_runtime::workspace::BaseWorkspaceService::new(base_workspace.clone()),
        );
        let runtime_services = runmat_runtime::context::RuntimeServicePorts::default()
            .with_workspace(base_workspace_service.clone())
            .with_base_workspace(base_workspace_service)
            .with_source_evaluation(source_evaluator.clone())
            .with_callable_artifacts(super::source_evaluation::callable_artifact_service(
                &source_evaluator,
            ))
            .with_local_tasks(local_tasks.submission_port(runtime_context.local_identity()))
            .with_call(std::rc::Rc::new(runmat_runtime::context::RuntimeCallRouter))
            .with_http(super::http_host::default_http_service())
            .with_placement(placement.clone())
            .with_acceleration(std::rc::Rc::new(
                runmat_accelerate::native_auto::AdmittedAcceleration,
            ))
            .with_spmd(parallel_services.spmd)
            .with_distributed(parallel_services.distributed);
        #[cfg(not(target_arch = "wasm32"))]
        let runtime_services =
            runtime_services.with_process(super::process_host::default_process_service());
        #[cfg(not(target_arch = "wasm32"))]
        let runtime_services = if let Some(parent) = parent_context {
            child_settings::inherit_caller_settings(runtime_services, parent)
        } else if let Some(directory) =
            runmat_config::runtime::ApplicationSettingsDirectory::discover()
        {
            let driver = super::host_driver::host_driver()?;
            let settings = std::rc::Rc::new(runmat_runtime::context::NativePreferenceService::new(
                directory.settings_file(),
                driver.handle().clone(),
            )?);
            runtime_services
                .with_preferences(settings.clone())
                .with_user_path(settings)
        } else {
            // The first async hydration returns CapabilityUnavailable before
            // source resolution; no session-memory fallback can hide storage.
            runtime_services
        };
        #[cfg(target_arch = "wasm32")]
        let runtime_services = {
            if let Some(parent) = parent_context {
                child_settings::inherit_caller_settings(runtime_services, parent)
            } else {
                // IndexedDB is origin-scoped and the explicit browser host can
                // replace this namespace with an application-specific one.
                let settings = std::rc::Rc::new(
                    runmat_runtime::context::BrowserPreferenceService::anonymous_origin_local(),
                );
                runtime_services
                    .with_preferences(settings.clone())
                    .with_user_path(settings)
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        let runtime_services = runtime_services.with_tcp(super::tcp_host::default_tcp_service());
        #[cfg(not(target_arch = "wasm32"))]
        let (
            runtime_services,
            foreign_runtime,
            native_ffi_adapter,
            java_adapter,
            python_adapter,
            mex_runtime,
        ) = {
            let foreign_runtime = std::rc::Rc::new(runmat_runtime::foreign::ForeignRuntime::new(
                runmat_runtime::foreign::ForeignPlatform::Native,
            ));
            let adapter =
                runmat_runtime::foreign::NativeFfiAdapter::new(foreign_runtime.handles().clone())?;
            foreign_runtime.register_adapter(adapter.clone())?;
            let java =
                runmat_runtime::foreign::JavaAdapter::new(foreign_runtime.handles().clone())?;
            foreign_runtime.register_adapter(java.clone())?;
            let python =
                runmat_runtime::foreign::PythonAdapter::new(foreign_runtime.handles().clone())?;
            foreign_runtime.register_adapter(python.clone())?;
            let mex = std::rc::Rc::new(match parent_mex_lane {
                Some(parent) => {
                    runmat_runtime::foreign::MexRuntimeSession::new_on_parent_native_lane(parent)
                }
                None => runmat_runtime::foreign::MexRuntimeSession::new(),
            });
            foreign_runtime.register_adapter(mex.clone())?;
            (
                runtime_services.with_foreign(foreign_runtime.clone()),
                foreign_runtime,
                adapter,
                java,
                python,
                mex,
            )
        };
        let program_service = std::rc::Rc::new(super::program::SessionProgramService::default());
        let runtime_services = runtime_services.with_program(program_service.clone());
        let canonical_device_route_ports = runtime_services.clone();
        let runtime_context = runtime_context.with_service_ports(runtime_services);
        let lifecycle_host = runmat_runtime::context::RuntimeLifecycleHost::new(
            &runtime_context,
            local_tasks.clone(),
        )
        .map_err(super::error_conversion::into_anyhow)?;
        super::source_evaluation::install_bound_invoker(&runtime_context, &source_evaluator);
        let session = Self {
            verbose,
            stats: ExecutionStats::default(),
            workspace_bindings: workspace::binding_control::BindingControl::new_for_context(
                &runtime_context,
                workspace::binding_table::BindingTable::new_for_context(&runtime_context)?,
            )?,
            base_workspace,
            abi_workspace_handle: crate::abi::WorkspaceHandle(Uuid::new_v4()),
            active_source_identity: None,
            active_primary_source: None,
            function_registry: runmat_vm::FunctionRegistry::default(),
            next_semantic_function_id: 0,
            dynamic_function_cache: Rc::new(dynamic_cache::DynamicFunctionCache::default()),
            startup_user_path_authority,
            startup_environment_path,
            user_path_hydrated: false,
            #[cfg(not(target_arch = "wasm32"))]
            mex_runtime,
            #[cfg(not(target_arch = "wasm32"))]
            foreign_runtime,
            #[cfg(not(target_arch = "wasm32"))]
            native_ffi_adapter,
            #[cfg(not(target_arch = "wasm32"))]
            java_adapter,
            #[cfg(not(target_arch = "wasm32"))]
            python_adapter,
            #[cfg(not(target_arch = "wasm32"))]
            generic_native_cache: crate::generic_native::GenericNativeCache::default(),
            #[cfg(not(target_arch = "wasm32"))]
            native_tiering_enabled,
            project_handoff: None,
            checked_fea_sources: Default::default(),
            pinned_inputs: None,
            captured_project: None,
            #[cfg(not(target_arch = "wasm32"))]
            private_project_root: None,
            #[cfg(not(target_arch = "wasm32"))]
            prepared_project_foreign: None,
            #[cfg(not(target_arch = "wasm32"))]
            prepared_selected_foreign_input: None,
            #[cfg(not(target_arch = "wasm32"))]
            selected_foreign_config: Some(Arc::new({
                let mut config = runmat_config::runtime::RunMatRuntimeConfig::default();
                // Embedding sessions retain the native adapter's in-process
                // default. Application configuration may choose isolation.
                config.foreign.native.isolation =
                    runmat_config::runtime::NativeFfiIsolation::InProcess;
                config
            })),
            source_pool: SourcePool::default(),
            source_evaluator,
            cancellation: cancellation.clone(),
            runtime_context,
            canonical_device_route_ports,
            inherited_command_profiling,
            graphics_callbacks: None,
            local_tasks,
            lifecycle_host,
            program_service,
            placement_session: placement,
            is_executing: false,
            async_input_handler: None,
            callstack_limit: runmat_runtime::context::DEFAULT_CALLSTACK_LIMIT,
            error_namespace: runmat_runtime::context::DEFAULT_ERROR_NAMESPACE.to_string(),
            active_source_name: "<repl>".into(),
            active_source_fullpath_name: None,
            telemetry_consent: true,
            telemetry_client_id: None,
            telemetry_platform: TelemetryPlatformInfo::default(),
            telemetry_sink: None,
            workspace_preview_tokens: HashMap::new(),
            emit_fusion_plan: false,
            compat_mode: CompatMode::Matlab,
            top_level_await_enabled: true,
            dynamic_eval_enabled: true,
            pending_companion_source_discovery: None,
        };

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
