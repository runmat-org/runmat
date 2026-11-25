//! Host artifact placement and additional publication storage admission.
use std::path::PathBuf;
use std::sync::{OnceLock, RwLock};

mod publication_memory;
pub(super) use publication_memory::over_budget_message;
pub use publication_memory::{
    clear_publication_memory_authority, install_publication_memory_authority, PublicationMemory,
    PublicationMemoryAuthority,
};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FeaRuntimeConfig {
    pub artifact_root: Option<PathBuf>,
    pub study_artifact_root: Option<PathBuf>,
    /// An optional host cap, in bytes, on the memory one whole-study publication may stage.
    /// `None` (the default) leaves the budget to the run's admitted memory authority: the memory
    /// admitted for the run less what the publishing process holds (see
    /// [`install_publication_memory_authority`]), or this machine's admissible memory when no
    /// host installed an authority. A cap lowers that budget and never raises it.
    pub publication_staging_cap_bytes: Option<u64>,
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

/// The memory one whole-study publication may stage, read when a publication starts. Resource
/// authority cannot silently revert to a default ceiling after lock poisoning.
pub(super) fn publication_staging_budget() -> Result<u64, String> {
    let cap = fea_runtime_config()
        .read()
        .map(|guard| guard.publication_staging_cap_bytes)
        .map_err(|_| "FEA publication resource config lock poisoned".to_string())?;
    publication_memory::staging_budget(cap)
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
    if config.publication_staging_cap_bytes == Some(0) {
        return Err("publication staging cap must be positive".into());
    }
    let mut guard = fea_runtime_config()
        .write()
        .map_err(|_| "FEA runtime config lock poisoned".to_string())?;
    Ok(std::mem::replace(&mut *guard, config))
}
