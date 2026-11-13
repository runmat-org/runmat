//! Typed host policy application. File/document precedence belongs to Config;
//! this module connects resolved settings to the actual session mechanisms.
use super::*;
#[cfg(not(target_arch = "wasm32"))]
use runmat_config::runtime::JitOptLevel;
use runmat_config::runtime::{GcPreset, JitConfig, LanguageCompatMode, RunMatRuntimeConfig};

impl RunMatSession {
    /// Cluster sessions never hydrate ambient application preferences/userpath.
    pub fn apply_cluster_runtime_config(&mut self, config: &RunMatRuntimeConfig) -> Result<()> {
        self.install_application_settings(Rc::new(
            runmat_runtime::context::IsolatedMemoryPreferenceService::new(),
        ))?;
        self.apply_runtime_config(config)?;
        Ok(())
    }

    /// Apply session-scoped settings without consulting ambient files or env.
    /// Hosts resolve the correct local/cluster policy before calling this.
    pub fn apply_runtime_config(&mut self, config: &RunMatRuntimeConfig) -> Result<()> {
        if config.jit.threshold == 0 {
            anyhow::bail!("JIT threshold must be positive");
        }
        #[cfg(target_arch = "wasm32")]
        if config.foreign != runmat_config::runtime::ForeignConfig::default() {
            anyhow::bail!("native Python, Java and MEX settings require a native execution host");
        }
        // Foreign adapters become immutable after startup. Replaying the same
        // resolved policy is not a request to recreate those adapters. Admit
        // real changes before mutating the rest of the session policy.
        #[cfg(not(target_arch = "wasm32"))]
        {
            let unchanged = match self.selected_foreign_config.as_ref() {
                Some(previous) => previous.foreign == config.foreign,
                None => false,
            };
            if !unchanged {
                self.set_foreign_config(&config.foreign)?;
            }
        }
        self.set_jit_config(&config.jit)?;
        self.set_compat_mode(match config.language.compat {
            LanguageCompatMode::RunMat => CompatMode::RunMat,
            LanguageCompatMode::Matlab => CompatMode::Matlab,
            LanguageCompatMode::Strict => CompatMode::Strict,
        });
        self.set_callstack_limit(config.runtime.callstack_limit);
        self.set_error_namespace(if config.runtime.error_namespace.trim().is_empty() {
            runmat_config::runtime::error_namespace_for_language_compat(config.language.compat)
                .into()
        } else {
            config.runtime.error_namespace.clone()
        });
        self.set_telemetry_consent(config.telemetry.enabled);
        self.runtime_context = self
            .runtime_context
            .clone()
            .with_acceleration_policy(crate::cluster_settings::acceleration_policy(config));
        #[cfg(not(target_arch = "wasm32"))]
        self.set_selected_foreign_config(Arc::new(config.clone()));
        Ok(())
    }

    /// Reconfiguration creates a fresh policy-owned compiler cache generation.
    /// Existing invocation leases retain their executors; old background jobs
    /// lose only their publication receiver, never a running invocation/result.
    pub fn set_jit_config(&mut self, config: &JitConfig) -> Result<()> {
        if config.threshold == 0 {
            anyhow::bail!("JIT threshold must be positive");
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (tiering, optimization) = native_policy(config);
            self.generic_native_cache
                .reconfigure_compilation_policy(tiering, optimization)
                .map_err(anyhow::Error::msg)?;
            self.native_tiering_enabled = config.enabled && cfg!(feature = "jit");
        }
        // WebAssembly has no native JIT backend. Serialized policy remains
        // available to the host, but is never advertised as applied native code.
        Ok(())
    }

    /// Validate collector capabilities without changing process-owned state.
    pub fn validate_gc_runtime_config(config: &RunMatRuntimeConfig) -> Result<()> {
        resolved_gc_config(config).map(|_| ())
    }

    /// Configure process-owned GC before execution. A worker host must first
    /// clamp/reject this policy against its admitted resources.
    pub fn configure_gc_from_runtime_config(config: &RunMatRuntimeConfig) -> Result<()> {
        gc_configure(resolved_gc_config(config)?).map_err(anyhow::Error::msg)
    }

    /// Enforce allocation limits after resolving project policy. Explicit
    /// over-budget requests fail; unspecified GC defaults are bounded by the
    /// allocation and any existing process heap ceiling.
    pub fn configure_gc_with_resource_limits(
        config: &RunMatRuntimeConfig,
        memory_bytes: usize,
        threads: usize,
    ) -> Result<()> {
        if memory_bytes == 0 || threads == 0 {
            anyhow::bail!("worker GC resource limits must be positive");
        }
        let mut gc = resolved_gc_config(config)?;
        let process = runmat_gc::gc_get_config();
        let ceiling = if process.max_heap_size == 0 {
            memory_bytes
        } else {
            memory_bytes.min(process.max_heap_size)
        };
        if config
            .gc
            .threads
            .is_some_and(|requested| requested > threads)
            || (config.gc.young_size_mb.is_some() && gc.young_generation_size > ceiling)
        {
            anyhow::bail!(
                "project GC settings exceed the worker's admitted CPU or memory resources"
            );
        }
        gc.max_gc_threads = gc.max_gc_threads.min(threads);
        gc.max_heap_size = if gc.max_heap_size == 0 {
            ceiling
        } else {
            gc.max_heap_size.min(ceiling)
        };
        gc.min_heap_size = gc.min_heap_size.min(gc.max_heap_size);
        gc.young_generation_size = gc.young_generation_size.min(gc.max_heap_size);
        gc_configure(gc).map_err(anyhow::Error::msg)
    }
}

fn resolved_gc_config(config: &RunMatRuntimeConfig) -> Result<GcConfig> {
    if config.gc.threads.is_some_and(|threads| threads != 1) {
        anyhow::bail!("gc.threads supports only 1: the current collector is serial; parallel GC workers are unavailable");
    }
    let mut gc = match config.gc.preset {
        Some(GcPreset::LowLatency) => GcConfig::low_latency(),
        Some(GcPreset::HighThroughput) => GcConfig::high_throughput(),
        Some(GcPreset::LowMemory) => GcConfig::low_memory(),
        Some(GcPreset::Debug) => GcConfig::debug(),
        None => GcConfig::default(),
    };
    if let Some(megabytes) = config.gc.young_size_mb {
        gc.young_generation_size = megabytes.checked_mul(1024 * 1024).ok_or_else(|| {
            anyhow::anyhow!("GC young-generation size exceeds addressable memory")
        })?;
    }
    if let Some(threads) = config.gc.threads {
        gc.max_gc_threads = threads;
    }
    gc.collect_statistics = config.gc.collect_stats;
    gc.verbose_logging = config.logging.debug || config.runtime.verbose;
    Ok(gc)
}

#[cfg(not(target_arch = "wasm32"))]
fn native_policy(
    config: &JitConfig,
) -> (
    runmat_jit::tiering::TieringConfig,
    runmat_jit::CompilerOptimization,
) {
    let threshold = u64::from(config.threshold);
    let eager = config.optimization_level == JitOptLevel::Aggressive;
    let tiering = runmat_jit::tiering::TieringConfig {
        generic_hot_threshold: threshold,
        specialized_hot_threshold: if eager { threshold } else { threshold * 2 },
        loop_hot_threshold: threshold * 4,
        ..Default::default()
    };
    let optimization = match config.optimization_level {
        JitOptLevel::None => runmat_jit::CompilerOptimization::None,
        JitOptLevel::Size => runmat_jit::CompilerOptimization::Size,
        JitOptLevel::Speed | JitOptLevel::Aggressive => runmat_jit::CompilerOptimization::Speed,
    };
    (tiering, optimization)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn compiler_policy_matrix_and_cache_generation_are_real() {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let unit = futures::executor::block_on(session.compile_executable_unit(
            crate::ExecutableSource::new(
                "settings-policy@1",
                "settingsPolicy.m",
                "function y = settingsPolicy(x)\ny = x + 1;\nend\n",
            ),
            None,
        ))
        .unwrap();
        let retained = session
            .generic_native_cache
            .resolve_or_compile(
                &unit,
                Some("settingsPolicy"),
                runmat_native_codegen::NativeLivenessAdmission::AuthorizedUnbounded,
            )
            .unwrap();
        assert!(session.generic_native_cache.has_retained_entries());
        for (level, backend, specialized) in [
            (JitOptLevel::None, "none", 20),
            (JitOptLevel::Size, "speed_and_size", 20),
            (JitOptLevel::Speed, "speed", 20),
            (JitOptLevel::Aggressive, "speed", 10),
        ] {
            let config = JitConfig {
                enabled: true,
                threshold: 10,
                optimization_level: level,
            };
            session.set_jit_config(&config).unwrap();
            let (tiering, optimization) = session.generic_native_cache.compilation_policy();
            assert_eq!(tiering.generic_hot_threshold, 10);
            assert_eq!(tiering.specialized_hot_threshold, specialized);
            assert_eq!(tiering.loop_hot_threshold, 40);
            assert_eq!(optimization.cranelift_opt_level(false), backend);
            assert!(!session.generic_native_cache.has_retained_entries());
        }
        assert!(session
            .set_jit_config(&JitConfig {
                threshold: 0,
                ..Default::default()
            })
            .is_err());
        assert!(retained.executor.retained_code_bytes() > 0);
        let mut applied = RunMatRuntimeConfig::default();
        session.apply_runtime_config(&applied).unwrap();
        session.apply_runtime_config(&applied).unwrap();
        let accepted = Arc::clone(session.selected_foreign_config.as_ref().unwrap());
        applied.jit.threshold = 0;
        applied
            .foreign
            .java
            .options
            .push("-Dmust-not-apply=true".into());
        assert!(session.apply_runtime_config(&applied).is_err());
        assert!(Arc::ptr_eq(
            &accepted,
            session.selected_foreign_config.as_ref().unwrap()
        ));
        let mut over_budget = RunMatRuntimeConfig::default();
        over_budget.gc.threads = Some(1);
        assert!(RunMatSession::validate_gc_runtime_config(&over_budget).is_ok());
        over_budget.gc.threads = Some(2);
        assert!(RunMatSession::validate_gc_runtime_config(&over_budget).is_err());
        over_budget.gc.threads = Some(4);
        assert!(RunMatSession::configure_gc_with_resource_limits(
            &over_budget,
            16 * 1024 * 1024,
            1
        )
        .is_err());
        over_budget.gc.threads = None;
        over_budget.gc.young_size_mb = Some(128);
        assert!(RunMatSession::configure_gc_with_resource_limits(
            &over_budget,
            16 * 1024 * 1024,
            1
        )
        .is_err());
    }
}
