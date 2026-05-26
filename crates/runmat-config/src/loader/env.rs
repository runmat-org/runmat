use anyhow::Result;

use crate::RunMatConfig;

/// Runtime config environment overrides are intentionally disabled in the
/// hard-cutover model. Configuration should come from runmat.toml/runmat.json
/// (or RUNMAT_CONFIG pointing to one of those files), with CLI flags as the
/// highest precedence runtime layer.
pub(crate) fn apply_environment_variables(_config: &mut RunMatConfig) -> Result<()> {
    Ok(())
}
