//! The process-wide analysis artifact store and retention policy, and the
//! configuration entry points that replace them.
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, RwLock};
#[cfg(test)]
use std::sync::{Mutex, MutexGuard};

use super::{
    AnalysisArtifactRetentionConfig, AnalysisArtifactStore, AnalysisArtifactStoreConfig,
    FilesystemAnalysisArtifactStore, InMemoryAnalysisArtifactStore,
};

pub(super) fn global_store() -> &'static RwLock<Arc<dyn AnalysisArtifactStore>> {
    static STORE: OnceLock<RwLock<Arc<dyn AnalysisArtifactStore>>> = OnceLock::new();
    STORE.get_or_init(|| {
        let default = store_from_config(config_from_env());
        RwLock::new(default)
    })
}

pub(super) fn retention_config() -> &'static RwLock<AnalysisArtifactRetentionConfig> {
    static CONFIG: OnceLock<RwLock<AnalysisArtifactRetentionConfig>> = OnceLock::new();
    CONFIG.get_or_init(|| RwLock::new(AnalysisArtifactRetentionConfig::default()))
}

pub fn configure_artifact_store(config: AnalysisArtifactStoreConfig) -> Result<(), String> {
    let mut guard = global_store()
        .write()
        .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
    *guard = store_from_config(config);
    Ok(())
}

/// Install a retained host-owned store while preserving the exact previous
/// owner. Embedded hosts must serialize access and restore it on scope exit;
/// constructing a fresh in-memory store per command would discard results.
pub fn replace_artifact_store(
    store: Arc<dyn AnalysisArtifactStore>,
) -> Result<Arc<dyn AnalysisArtifactStore>, String> {
    let mut guard = global_store()
        .write()
        .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
    Ok(std::mem::replace(&mut *guard, store))
}

/// Scoped host counterpart of `configure_artifact_retention`.
pub fn replace_artifact_retention(
    config: AnalysisArtifactRetentionConfig,
) -> Result<AnalysisArtifactRetentionConfig, String> {
    let mut guard = retention_config()
        .write()
        .map_err(|_| "analysis artifact retention config lock poisoned".to_string())?;
    Ok(std::mem::replace(&mut *guard, config))
}

pub fn configure_artifact_retention(config: AnalysisArtifactRetentionConfig) -> Result<(), String> {
    let mut guard = retention_config()
        .write()
        .map_err(|_| "analysis artifact retention config lock poisoned".to_string())?;
    *guard = config;
    Ok(())
}

pub fn configure_artifact_store_from_env() -> Result<(), String> {
    configure_artifact_store(config_from_env())
}

fn store_from_config(config: AnalysisArtifactStoreConfig) -> Arc<dyn AnalysisArtifactStore> {
    match config {
        AnalysisArtifactStoreConfig::InMemory => Arc::new(InMemoryAnalysisArtifactStore::new()),
        AnalysisArtifactStoreConfig::Filesystem { root } => {
            Arc::new(FilesystemAnalysisArtifactStore::new(root))
        }
    }
}

fn config_from_env() -> AnalysisArtifactStoreConfig {
    let mode = std::env::var("RUNMAT_FEA_ARTIFACT_STORE")
        .or_else(|_| std::env::var("RUNMAT_ANALYSIS_ARTIFACT_STORE"))
        .unwrap_or_else(|_| "filesystem".to_string())
        .to_lowercase();
    if mode == "filesystem" {
        let root = std::env::var("RUNMAT_FEA_ARTIFACT_ROOT")
            .or_else(|_| std::env::var("RUNMAT_ANALYSIS_ARTIFACT_ROOT"))
            .map(PathBuf::from)
            .unwrap_or_else(|_| default_filesystem_artifact_root());
        AnalysisArtifactStoreConfig::Filesystem { root }
    } else {
        AnalysisArtifactStoreConfig::InMemory
    }
}

pub fn default_filesystem_artifact_root() -> PathBuf {
    PathBuf::from(".artifacts")
}

#[cfg(test)]
pub fn set_artifact_store_for_tests(store: Arc<dyn AnalysisArtifactStore>) {
    let mut guard = global_store()
        .write()
        .expect("analysis artifact store lock poisoned");
    *guard = store;
}

#[cfg(test)]
pub fn reset_artifact_store_for_tests() {
    let mut guard = global_store()
        .write()
        .expect("analysis artifact store lock poisoned");
    *guard = Arc::new(InMemoryAnalysisArtifactStore::new());
    *retention_config()
        .write()
        .expect("analysis artifact retention config lock poisoned") =
        AnalysisArtifactRetentionConfig::default();
}

#[cfg(test)]
pub(crate) fn artifact_store_test_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
