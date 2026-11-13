mod discovery;
mod env;
mod file;
mod layers;
mod settings;
pub use discovery::ApplicationSettingsDirectory;
pub use layers::{
    ConfigLoadOptions, ConfigSource, ConfigurationPolicy, HostConfigDocument, HostRuntimeSettings,
    ResolvedRuntimeConfig, UserSettingsSource,
};
pub use settings::UserSettingsDocument;

use anyhow::Result;
use std::path::{Path, PathBuf};

use crate::runtime::RunMatRuntimeConfig;

/// Configuration loader with multiple source support
pub struct ConfigLoader;

impl ConfigLoader {
    /// Load configuration from all sources with proper precedence
    pub fn load() -> Result<RunMatRuntimeConfig> {
        Ok(Self::load_with_options(&ConfigLoadOptions::native())?.config)
    }

    /// Resolve recursive layers and exact leaf provenance using a host policy.
    pub fn load_with_options(options: &ConfigLoadOptions) -> Result<ResolvedRuntimeConfig> {
        layers::resolve(options)
    }

    /// Walk up from the provided directory looking for the first config file.
    pub fn discover_config_path_from(start: &Path) -> Option<PathBuf> {
        discovery::discover_project_config_path_from(start)
    }

    /// Load configuration from a specific file
    pub fn load_from_file(path: &Path) -> Result<RunMatRuntimeConfig> {
        file::load_from_file(path)
    }

    /// Save configuration to a file
    pub fn save_to_file(config: &RunMatRuntimeConfig, path: &Path) -> Result<()> {
        file::save_to_file(config, path)
    }

    /// Render runtime config into TOML/JSON text matching the output file contract.
    pub fn render_runtime_config(config: &RunMatRuntimeConfig, path: &Path) -> Result<String> {
        file::render_runtime_config(config, path)
    }

    /// Generate a sample configuration file
    pub fn generate_sample_config() -> String {
        file::generate_sample_config()
    }
}
