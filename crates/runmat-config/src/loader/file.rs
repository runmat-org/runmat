use anyhow::{Context, Result};
use log::info;
use std::fs;
use std::path::Path;

use crate::RunMatConfig;

/// Load configuration from a specific file.
pub(crate) fn load_from_file(path: &Path) -> Result<RunMatConfig> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {}", path.display()))?;

    let config = match path.extension().and_then(|ext| ext.to_str()) {
        // `.runmat` is a TOML alias by default (single canonical format)
        None if path.file_name().and_then(|n| n.to_str()) == Some(".runmat") => {
            toml::from_str(&content).with_context(|| {
                format!("Failed to parse .runmat (TOML) config: {}", path.display())
            })?
        }
        Some("runmat") => toml::from_str(&content).with_context(|| {
            format!("Failed to parse .runmat (TOML) config: {}", path.display())
        })?,
        Some("yaml") | Some("yml") => serde_yaml::from_str(&content)
            .with_context(|| format!("Failed to parse YAML config: {}", path.display()))?,
        Some("json") => serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse JSON config: {}", path.display()))?,
        Some("toml") => toml::from_str(&content)
            .with_context(|| format!("Failed to parse TOML config: {}", path.display()))?,
        _ => {
            // Try auto-detect (prefer TOML for unknown/no extension)
            if let Ok(config) = toml::from_str(&content) {
                config
            } else if let Ok(config) = serde_yaml::from_str(&content) {
                config
            } else if let Ok(config) = serde_json::from_str(&content) {
                config
            } else {
                return Err(anyhow::anyhow!(
                    "Could not parse config file {} (tried TOML, YAML, JSON)",
                    path.display()
                ));
            }
        }
    };

    Ok(config)
}

/// Save configuration to a file.
pub(crate) fn save_to_file(config: &RunMatConfig, path: &Path) -> Result<()> {
    let content = match path.extension().and_then(|ext| ext.to_str()) {
        Some("yaml") | Some("yml") => {
            serde_yaml::to_string(config).context("Failed to serialize config to YAML")?
        }
        Some("json") => {
            serde_json::to_string_pretty(config).context("Failed to serialize config to JSON")?
        }
        Some("toml") => {
            toml::to_string_pretty(config).context("Failed to serialize config to TOML")?
        }
        _ => {
            // Default to YAML
            serde_yaml::to_string(config).context("Failed to serialize config to YAML")?
        }
    };

    fs::write(path, content)
        .with_context(|| format!("Failed to write config file: {}", path.display()))?;

    info!("Configuration saved to: {}", path.display());
    Ok(())
}

/// Generate a sample configuration file.
pub(crate) fn generate_sample_config() -> String {
    let config = RunMatConfig::default();
    serde_yaml::to_string(&config).unwrap_or_else(|_| "# Failed to generate config".to_string())
}
