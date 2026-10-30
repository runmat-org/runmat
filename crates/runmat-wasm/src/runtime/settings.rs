use runmat_config::runtime::{FeaArtifactStoreMode, RunMatRuntimeConfig};

/// This browser host installs no native foreign adapters or generic host bridge.
/// Use the canonical typed admission owner rather than a family-name/Python test.
pub(crate) fn validate_browser_interop(
    manifest: &runmat_types::InteropManifest,
) -> Result<(), String> {
    runmat_runtime::foreign::admit_interop_manifest(
        manifest,
        &Default::default(),
        runmat_runtime::foreign::ForeignPlatform::Wasm {
            host_bridge_available: false,
        },
    )
    .map(|_| ())
    .map_err(|error| error.to_string())
}

/// Pure admission, before logging, GC, plots, filesystem or provider effects.
/// Native paths have no browser storage authority; resolving them is not proof
/// that this host can honor them.
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) fn validate_browser_capabilities(config: &RunMatRuntimeConfig) -> Result<(), String> {
    if config.logging.file.is_some() {
        return Err("logging.file requires a native host; browser logging does not write native filesystem paths".into());
    }
    if config.fea.artifact_store == Some(FeaArtifactStoreMode::Filesystem)
        || config.fea.artifact_root.is_some()
        || config.fea.study_artifact_root.is_some()
    {
        return Err("native FEA filesystem storage and artifact/study roots require a native host; use in_memory without native roots in this browser".into());
    }
    if config.foreign != runmat_config::runtime::ForeignConfig::default() {
        return Err("native Python, Java and MEX settings require a native execution host".into());
    }
    runmat_core::RunMatSession::validate_gc_runtime_config(config)
        .map_err(|error| error.to_string())
}

/// Apply process-owned browser mechanisms only after typed layer resolution.
pub(crate) fn apply_resources(config: &RunMatRuntimeConfig) -> Result<(), String> {
    runmat_core::RunMatSession::configure_gc_from_runtime_config(config)
        .map_err(|error| error.to_string())?;
    use runmat_runtime::builtins::plotting::{
        set_scatter_target_points, set_scene_export_budget_bytes, set_surface_vertex_budget,
    };
    if let Some(points) = config.plotting.scatter_target_points {
        set_scatter_target_points(points);
    }
    if let Some(vertices) = config.plotting.surface_vertex_budget {
        set_surface_vertex_budget(vertices);
    }
    if let Some(export) = &config.plotting.export {
        set_scene_export_budget_bytes(export.scene_budget_bytes);
    }
    use runmat_runtime::analysis::storage;
    if let Some(mode) = config.fea.artifact_store {
        let store = match mode {
            FeaArtifactStoreMode::InMemory => storage::AnalysisArtifactStoreConfig::InMemory,
            FeaArtifactStoreMode::Filesystem => {
                #[cfg(target_arch = "wasm32")]
                return Err("filesystem FEA artifact storage requires a native host; use in_memory in this browser".into());
                #[cfg(not(target_arch = "wasm32"))]
                {
                    storage::AnalysisArtifactStoreConfig::Filesystem {
                        root: config
                            .fea
                            .artifact_root
                            .clone()
                            .unwrap_or_else(storage::default_filesystem_artifact_root),
                    }
                }
            }
        };
        storage::configure_artifact_store(store)?;
    }
    storage::configure_artifact_retention(storage::AnalysisArtifactRetentionConfig {
        max_runs: config.fea.artifact_max_runs,
        max_runs_per_kind: config.fea.artifact_max_runs_per_kind,
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_capability_matrix_rejects_native_only_settings_without_effects() {
        let base = RunMatRuntimeConfig::default();
        assert!(validate_browser_capabilities(&base).is_ok());
        let changes: [fn(&mut RunMatRuntimeConfig); 6] = [
            |config: &mut RunMatRuntimeConfig| config.logging.file = Some("runtime.log".into()),
            |config: &mut RunMatRuntimeConfig| config.fea.artifact_root = Some("results".into()),
            |config: &mut RunMatRuntimeConfig| {
                config.fea.study_artifact_root = Some("studies".into())
            },
            |config: &mut RunMatRuntimeConfig| {
                config.fea.artifact_store = Some(FeaArtifactStoreMode::Filesystem)
            },
            |config: &mut RunMatRuntimeConfig| config.foreign.java.options.push("-Xmx1g".into()),
            |config: &mut RunMatRuntimeConfig| config.gc.threads = Some(2),
        ];
        for change in changes {
            let mut config = base.clone();
            change(&mut config);
            assert!(validate_browser_capabilities(&config).is_err());
        }
        let mut supported = base;
        supported.gc.threads = Some(1);
        supported.fea.artifact_store = Some(FeaArtifactStoreMode::InMemory);
        supported.fea.artifact_max_runs = Some(3);
        assert!(validate_browser_capabilities(&supported).is_ok());
        assert!(validate_browser_interop(&runmat_types::InteropManifest::empty()).is_ok());
        use runmat_builtins::ForeignAdapterFamily;
        for family in [
            ForeignAdapterFamily::Python,
            ForeignAdapterFamily::Java,
            ForeignAdapterFamily::Mex,
            ForeignAdapterFamily::NativeFfi,
        ] {
            let mut interop = runmat_types::InteropManifest::empty();
            interop
                .adapters
                .push(runmat_types::ForeignAdapterRequirement {
                    adapter: runmat_types::ForeignAdapterId::new(family.adapter_id()).unwrap(),
                    minimum_version: 1,
                    capabilities: runmat_types::CapabilitySet(std::collections::BTreeSet::from([
                        runmat_types::CapabilityRequirement::ForeignRuntime,
                    ])),
                    execution_stack: runmat_types::ExecutionStackRequirement::Process,
                    artifact_identities: Vec::new(),
                });
            assert!(validate_browser_interop(&interop).is_err(), "{family:?}");
        }
    }
}
