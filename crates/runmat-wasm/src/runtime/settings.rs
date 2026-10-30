use runmat_config::runtime::{FeaArtifactStoreMode, RunMatRuntimeConfig};

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
