mod discovery;
mod env;
mod file;

use anyhow::Result;
use log::{debug, info};
use std::path::{Path, PathBuf};

use crate::RunMatConfig;

/// Configuration loader with multiple source support
pub struct ConfigLoader;

impl ConfigLoader {
    /// Load configuration from all sources with proper precedence
    pub fn load() -> Result<RunMatConfig> {
        let mut config = Self::load_from_files()?;
        Self::apply_environment_variables(&mut config)?;
        Ok(config)
    }

    /// Find and load configuration from files
    fn load_from_files() -> Result<RunMatConfig> {
        if let Ok(config_path) = std::env::var("RUNMAT_CONFIG") {
            let path = PathBuf::from(config_path);
            if path.is_dir() {
                return Err(anyhow::anyhow!(
                    "RUNMAT_CONFIG points to a directory, expected a file: {}",
                    path.display()
                ));
            }
            if !path.is_file() {
                return Err(anyhow::anyhow!(
                    "RUNMAT_CONFIG points to a missing file: {}",
                    path.display()
                ));
            }
            info!(
                "Loading configuration from RUNMAT_CONFIG: {}",
                path.display()
            );
            return Self::load_from_file(&path);
        }

        if let Ok(current_dir) = std::env::current_dir() {
            if let Some(path) = discovery::discover_project_config_path_from(&current_dir) {
                info!("Loading configuration from: {}", path.display());
                return Self::load_from_file(&path);
            }
        }

        if let Some(path) = discovery::user_config_candidates()
            .into_iter()
            .find(|candidate| candidate.is_file())
        {
            info!("Loading configuration from: {}", path.display());
            return Self::load_from_file(&path);
        }

        debug!("No configuration file found, using defaults");
        Ok(RunMatConfig::default())
    }

    /// Walk up from the provided directory looking for the first config file.
    pub fn discover_config_path_from(start: &Path) -> Option<PathBuf> {
        discovery::discover_project_config_path_from(start)
    }

    /// Load configuration from a specific file
    pub fn load_from_file(path: &Path) -> Result<RunMatConfig> {
        file::load_from_file(path)
    }

    /// Apply environment variable overrides
    fn apply_environment_variables(config: &mut RunMatConfig) -> Result<()> {
        env::apply_environment_variables(config)
    }

    /// Save configuration to a file
    pub fn save_to_file(config: &RunMatConfig, path: &Path) -> Result<()> {
        file::save_to_file(config, path)
    }

    /// Render runtime config into TOML/JSON text matching the output file contract.
    pub fn render_runtime_config(config: &RunMatConfig, path: &Path) -> Result<String> {
        let format = file::format_from_path(path)?;
        file::render_runtime_config(config, format)
    }

    /// Generate a sample configuration file
    pub fn generate_sample_config() -> String {
        file::generate_sample_config()
    }
}
