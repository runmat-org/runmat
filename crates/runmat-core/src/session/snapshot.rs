use super::*;

impl RunMatSession {
    /// Create a new session
    pub fn new() -> Result<Self> {
        Self::with_options(true, false) // JIT enabled, verbose disabled
    }

    /// Create a new session with specific options
    pub fn with_options(enable_jit: bool, verbose: bool) -> Result<Self> {
        Self::from_snapshot(enable_jit, verbose, None)
    }

    /// Create a new session with snapshot loading
    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_snapshot<P: AsRef<Path>>(
        enable_jit: bool,
        verbose: bool,
        snapshot_path: Option<P>,
    ) -> Result<Self> {
        let snapshot = snapshot_path.and_then(|path| match Self::load_snapshot(path.as_ref()) {
            Ok(snapshot) => {
                info!(
                    "Snapshot loaded successfully from {}",
                    path.as_ref().display()
                );
                Some(Arc::new(snapshot))
            }
            Err(e) => {
                warn!(
                    "Failed to load snapshot from {}: {}, continuing without snapshot",
                    path.as_ref().display(),
                    e
                );
                None
            }
        });
        Self::from_snapshot(enable_jit, verbose, snapshot)
    }

    /// Create a session using snapshot bytes (already fetched from disk or network)
    pub fn with_snapshot_bytes(
        enable_jit: bool,
        verbose: bool,
        snapshot_bytes: Option<&[u8]>,
    ) -> Result<Self> {
        let snapshot =
            snapshot_bytes.and_then(|bytes| match Self::load_snapshot_from_bytes(bytes) {
                Ok(snapshot) => {
                    info!("Snapshot loaded successfully from in-memory bytes");
                    Some(Arc::new(snapshot))
                }
                Err(e) => {
                    warn!("Failed to load snapshot from bytes: {e}, continuing without snapshot");
                    None
                }
            });
        Self::from_snapshot(enable_jit, verbose, snapshot)
    }

    fn from_snapshot(
        enable_jit: bool,
        verbose: bool,
        snapshot: Option<Arc<Snapshot>>,
    ) -> Result<Self> {
        #[cfg(target_arch = "wasm32")]
        let snapshot = {
            match snapshot {
                some @ Some(_) => some,
                None => Self::build_wasm_snapshot(),
            }
        };

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
            info!("JIT support was requested but the 'jit' feature is disabled; running interpreter-only.");
        }

        let session = Self {
            #[cfg(feature = "jit")]
            jit_engine,
            verbose,
            stats: ExecutionStats::default(),
            variables: HashMap::new(),
            variable_array: Vec::new(),
            variable_names: HashMap::new(),
            workspace_values: HashMap::new(),
            function_definitions: HashMap::new(),
            source_pool: SourcePool::default(),
            function_source_ids: HashMap::new(),
            snapshot,
            interrupt_flag: Arc::new(AtomicBool::new(false)),
            is_executing: false,
            async_input_handler: None,
            callstack_limit: runmat_vm::DEFAULT_CALLSTACK_LIMIT,
            error_namespace: runmat_vm::DEFAULT_ERROR_NAMESPACE.to_string(),
            default_source_name: "<repl>".to_string(),
            source_name_override: None,
            telemetry_consent: true,
            telemetry_client_id: None,
            telemetry_platform: TelemetryPlatformInfo::default(),
            telemetry_sink: None,
            workspace_preview_tokens: HashMap::new(),
            workspace_version: 0,
            emit_fusion_plan: false,
            compat_mode: CompatMode::Matlab,
            format_mode: runmat_builtins::FormatMode::default(),
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
        self.source_name_override
            .as_deref()
            .unwrap_or(&self.default_source_name)
    }

    #[cfg(target_arch = "wasm32")]
    fn build_wasm_snapshot() -> Option<Arc<Snapshot>> {
        use log::{info, warn};

        info!("No snapshot provided; building stdlib snapshot inside wasm runtime");
        let config = SnapshotConfig {
            compression_enabled: false,
            validation_enabled: false,
            memory_mapping_enabled: false,
            parallel_loading: false,
            progress_reporting: false,
            ..Default::default()
        };

        match SnapshotBuilder::new(config).build() {
            Ok(snapshot) => {
                info!("WASM snapshot build completed successfully");
                Some(Arc::new(snapshot))
            }
            Err(err) => {
                warn!("Failed to build stdlib snapshot in wasm runtime: {err}");
                None
            }
        }
    }

    /// Load a snapshot from disk
    #[cfg(not(target_arch = "wasm32"))]
    fn load_snapshot(path: &Path) -> Result<Snapshot> {
        let mut loader = SnapshotLoader::new(SnapshotConfig::default());
        let (snapshot, _stats) = loader
            .load(path)
            .map_err(|e| anyhow::anyhow!("Failed to load snapshot: {}", e))?;
        Ok(snapshot)
    }

    /// Load a snapshot from in-memory bytes
    fn load_snapshot_from_bytes(bytes: &[u8]) -> Result<Snapshot> {
        let mut loader = SnapshotLoader::new(SnapshotConfig::default());
        let (snapshot, _stats) = loader
            .load_from_bytes(bytes)
            .map_err(|e| anyhow::anyhow!("Failed to load snapshot: {}", e))?;
        Ok(snapshot)
    }

    /// Get snapshot information
    pub fn snapshot_info(&self) -> Option<String> {
        self.snapshot.as_ref().map(|snapshot| {
            format!(
                "Snapshot loaded: {} builtins, {} HIR functions, {} bytecode entries",
                snapshot.builtins.functions.len(),
                snapshot.hir_cache.functions.len(),
                snapshot.bytecode_cache.stdlib_bytecode.len()
            )
        })
    }

    /// Check if a snapshot is loaded
    pub fn has_snapshot(&self) -> bool {
        self.snapshot.is_some()
    }
}
