use anyhow::{Context, Result};
use log::error;
use runmat_config::{ConfigLoader, RunMatConfig};

use crate::cli::ConfigCommand;

pub async fn execute_config_command(
    config_command: ConfigCommand,
    config: &RunMatConfig,
) -> Result<()> {
    match config_command {
        ConfigCommand::Show => {
            println!("Current RunMat Configuration:");
            println!("==============================");

            let yaml =
                serde_yaml::to_string(config).context("Failed to serialize configuration")?;
            println!("{yaml}");
        }
        ConfigCommand::Generate { output } => {
            let sample_config = RunMatConfig::default();
            ConfigLoader::save_to_file(&sample_config, &output)
                .with_context(|| format!("Failed to write config to {}", output.display()))?;

            println!("Sample configuration generated: {}", output.display());
            println!("Edit this file to customize your RunMat settings.");
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
