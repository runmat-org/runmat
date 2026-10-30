use anyhow::{Context, Result};
use log::error;
use runmat_config::runtime::{
    ApplicationSettingsDirectory, ConfigLoader, RunMatRuntimeConfig, UserSettingsDocument,
};
use std::path::Path;

use crate::cli::{ConfigCommand, ConfigFormat, ConfigUserCommand};
use crate::presentation;

pub async fn execute_config_command(
    config_command: ConfigCommand,
    config: &RunMatRuntimeConfig,
) -> Result<()> {
    match config_command {
        ConfigCommand::Show { format, .. } => {
            let rendered = render_runtime_config(config, format)?;
            println!("{rendered}");
        }
        ConfigCommand::Generate { output, format } => {
            let format = format.unwrap_or_else(|| infer_format_from_path(&output));
            let sample = render_sample_config(format)?;
            std::fs::write(&output, sample)
                .with_context(|| format!("Failed to write config to {}", output.display()))?;

            let styles = presentation::stdout();
            println!(
                "{}: {}",
                styles.success("Sample RunMat config generated"),
                styles.path(output.display())
            );
            println!(
                "{}",
                styles.muted("This file includes project and runtime sections.")
            );
        }
        ConfigCommand::Validate { config_file } => match ConfigLoader::load_from_file(&config_file)
        {
            Ok(_) => {
                println!(
                    "{}: {}",
                    presentation::stdout().success("Configuration file is valid"),
                    presentation::stdout().path(config_file.display())
                );
            }
            Err(e) => {
                error!("Configuration validation failed: {e}");
                std::process::exit(1);
            }
        },
        ConfigCommand::Paths => {
            let styles = presentation::stdout();
            println!("{}", styles.heading("RunMat Configuration File Locations:"));
            println!("{}", styles.muted("===================================="));
            println!();

            if let Ok(config_path) = std::env::var("RUNMAT_CONFIG") {
                println!(
                    "{} {}",
                    styles.label("Environment override:"),
                    styles.path(config_path)
                );
            }

            println!("Project discovery (walk-up): runmat.toml, runmat.json");

            println!();
            println!("{}", styles.heading("User config:"));
            if let Some(home_dir) = dirs::home_dir() {
                let config_dir = home_dir.join(".config/runmat");
                for name in &["settings-v1.json", "config.toml", "config.json"] {
                    let path = config_dir.join(name);
                    let exists = if path.exists() { " (exists)" } else { "" };
                    println!(
                        "  {}{}",
                        styles.path(path.display()),
                        styles.success(exists)
                    );
                }
            }
        }
        ConfigCommand::User { command } => execute_user_command(command)?,
    }
    Ok(())
}

fn execute_user_command(command: ConfigUserCommand) -> Result<()> {
    let file = match &command {
        ConfigUserCommand::Show { file }
        | ConfigUserCommand::Set { file, .. }
        | ConfigUserCommand::Reset { file, .. } => file,
    };
    let path = file
        .clone()
        .or_else(|| {
            ApplicationSettingsDirectory::discover().map(|directory| directory.settings_file())
        })
        .context("user settings directory unavailable; supply --file")?;
    let mut document = UserSettingsDocument::load(&path)?;
    match command {
        ConfigUserCommand::Show { .. } => print!("{}", document.source()),
        ConfigUserCommand::Set { field, value, .. } => {
            let value: serde_json::Value =
                serde_json::from_str(&value).context("--value must be a typed JSON value")?;
            document.set(&field.iter().map(String::as_str).collect::<Vec<_>>(), value)?;
            document.save(&path)?;
        }
        ConfigUserCommand::Reset { field, .. } => {
            document.reset(&field.iter().map(String::as_str).collect::<Vec<_>>())?;
            document.save(&path)?;
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
