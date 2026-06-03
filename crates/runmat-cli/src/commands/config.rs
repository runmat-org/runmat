use anyhow::{Context, Result};
use log::error;
use runmat_config::runtime::{ConfigLoader, RunMatRuntimeConfig};
use std::path::Path;

use crate::cli::{ConfigCommand, ConfigFormat};

pub async fn execute_config_command(
    config_command: ConfigCommand,
    config: &RunMatRuntimeConfig,
) -> Result<()> {
    match config_command {
        ConfigCommand::Show { format } => {
            let rendered = render_runtime_config(config, format)?;
            println!("{rendered}");
        }
        ConfigCommand::Generate { output, format } => {
            let format = format.unwrap_or_else(|| infer_format_from_path(&output));
            let sample = render_sample_config(format)?;
            std::fs::write(&output, sample)
                .with_context(|| format!("Failed to write config to {}", output.display()))?;

            println!("Sample RunMat config generated: {}", output.display());
            println!("This file includes project and runtime sections.");
        }
        ConfigCommand::Validate { config_file } => match ConfigLoader::load_from_file(&config_file)
        {
            Ok(_) => {
                println!("Configuration file is valid: {}", config_file.display());
            }
            Err(e) => {
                error!("Configuration validation failed: {e}");
                std::process::exit(1);
            }
        },
        ConfigCommand::Paths => {
            println!("RunMat Configuration File Locations:");
            println!("====================================");
            println!();

            if let Ok(config_path) = std::env::var("RUNMAT_CONFIG") {
                println!("Environment override: {config_path}");
            }

            println!("Project discovery (walk-up): runmat.toml, runmat.json");

            println!();
            println!("User config:");
            if let Some(home_dir) = dirs::home_dir() {
                let config_dir = home_dir.join(".config/runmat");
                for name in &["config.toml", "config.json"] {
                    let path = config_dir.join(name);
                    let exists = if path.exists() { " (exists)" } else { "" };
                    println!("  {}{}", path.display(), exists);
                }
            }
        }
    }
    Ok(())
}

fn infer_format_from_path(path: &Path) -> ConfigFormat {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("json") => ConfigFormat::Json,
        _ => ConfigFormat::Toml,
    }
}

fn render_runtime_config(config: &RunMatRuntimeConfig, format: ConfigFormat) -> Result<String> {
    let virtual_path = match format {
        ConfigFormat::Toml => Path::new("resolved.runmat.toml"),
        ConfigFormat::Json => Path::new("resolved.runmat.json"),
    };
    ConfigLoader::render_runtime_config(config, virtual_path)
}

fn render_sample_config(format: ConfigFormat) -> Result<String> {
    let sample_toml = ConfigLoader::generate_sample_config();
    match format {
        ConfigFormat::Toml => Ok(sample_toml),
        ConfigFormat::Json => {
            let value: toml::Value = toml::from_str(&sample_toml)
                .context("Failed to parse sample TOML for JSON conversion")?;
            serde_json::to_string_pretty(&value)
                .context("Failed to serialize sample config as JSON")
        }
    }
}
