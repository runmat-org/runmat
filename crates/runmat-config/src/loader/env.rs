use anyhow::Result;
use std::env;

use crate::RunMatConfig;

/// Runtime config environment overrides are intentionally disabled in the
/// hard-cutover model. Configuration should come from runmat.toml/runmat.json
/// (or RUNMAT_CONFIG pointing to one of those files), with CLI flags as the
/// highest precedence runtime layer.
pub(crate) fn apply_environment_variables(_config: &mut RunMatConfig) -> Result<()> {
    Ok(())
}

pub(crate) fn env_value(primary: &str, aliases: &[&str]) -> Option<String> {
    env::var(primary)
        .ok()
        .or_else(|| aliases.iter().find_map(|alias| env::var(alias).ok()))
}
