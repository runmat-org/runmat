//! Host artifact placement and additional publication storage admission.
use std::path::PathBuf;
use std::sync::{OnceLock, RwLock};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeaRuntimeConfig {
    pub artifact_root: Option<PathBuf>,
    pub study_artifact_root: Option<PathBuf>,
    /// Additional host storage envelope for privately prepared whole-study publication.
    pub publication_staging_bytes: u64,
}
impl Default for FeaRuntimeConfig {
    fn default() -> Self {
        Self {
            artifact_root: None,
            study_artifact_root: None,
            publication_staging_bytes: 128 * 1024 * 1024,
        }
    }
}

fn fea_runtime_config() -> &'static RwLock<FeaRuntimeConfig> {
    static CONFIG: OnceLock<RwLock<FeaRuntimeConfig>> = OnceLock::new();
    CONFIG.get_or_init(|| RwLock::new(FeaRuntimeConfig::default()))
}

pub(super) fn current_fea_runtime_config() -> FeaRuntimeConfig {
    fea_runtime_config()
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_default()
}

/// Resource authority cannot silently revert to a default ceiling after lock poisoning.
pub(super) fn publication_staging_budget() -> Result<u64, String> {
    fea_runtime_config()
        .read()
        .map(|guard| guard.publication_staging_bytes)
        .map_err(|_| "FEA publication resource config lock poisoned".to_string())
}

pub fn default_fea_artifact_root() -> PathBuf {
    PathBuf::from(".artifacts")
}

pub fn configure_fea_runtime(config: FeaRuntimeConfig) -> Result<(), String> {
    replace_fea_runtime(config).map(|_| ())
}

/// Embedded hosts serialize runtime commands and restore the returned exact
/// configuration when their execution scope exits.
pub fn replace_fea_runtime(config: FeaRuntimeConfig) -> Result<FeaRuntimeConfig, String> {
    if config.publication_staging_bytes == 0 {
        return Err("publication staging budget must be positive".into());
    }
    let mut guard = fea_runtime_config()
        .write()
        .map_err(|_| "FEA runtime config lock poisoned".to_string())?;
    Ok(std::mem::replace(&mut *guard, config))
}
