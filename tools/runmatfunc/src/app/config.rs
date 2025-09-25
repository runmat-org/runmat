use std::path::PathBuf;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::cli::CliArgs;

/// User configuration file (loaded from $HOME/.runmatfunc/config.toml by default).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AppConfig {
    pub default_model: Option<String>,
    pub artifacts_dir: Option<PathBuf>,
}

impl AppConfig {
    pub fn load(_args: &CliArgs) -> Result<Self> {
        // TODO: implement config discovery (env + default path)
        Ok(Self::default())
    }
}
