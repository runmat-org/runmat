mod discovery;
mod env;
mod file;
mod parse;

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
        // Try to find config file in order of preference
        let config_paths = discovery::find_config_files();

        for path in config_paths {
            if path.is_dir() {
                info!(
                    "Ignoring config directory path (expected file): {}",
                    path.display()
                );
                continue;
            }
            if path.exists() {
                info!("Loading configuration from: {}", path.display());
                return Self::load_from_file(&path);
            }
        }

        debug!("No configuration file found, using defaults");
        Ok(RunMatConfig::default())
    }

    /// Walk up from the provided directory looking for the first config file.
    pub fn discover_config_path_from(start: &Path) -> Option<PathBuf> {
        discovery::discover_config_path_from(start)
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

    /// Generate a sample configuration file
    pub fn generate_sample_config() -> String {
        file::generate_sample_config()
    }
}
