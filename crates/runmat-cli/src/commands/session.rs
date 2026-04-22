use anyhow::{Context, Result};
use runmat_config::RunMatConfig;
use runmat_core::RunMatSession;
use std::path::PathBuf;

use crate::diagnostics::{parser_compat, resolved_error_namespace};
use crate::telemetry::{sink as runtime_sink, telemetry_client_id};

pub(crate) fn create_session(
    enable_jit: bool,
    verbose: bool,
    snapshot_path: Option<&PathBuf>,
    config: &RunMatConfig,
    create_error_context: &'static str,
) -> Result<RunMatSession> {
    let mut engine = RunMatSession::with_snapshot(enable_jit, verbose, snapshot_path)
        .context(create_error_context)?;
    engine.set_telemetry_consent(config.telemetry.enabled);
    engine.set_telemetry_sink(runtime_sink());
    engine.set_compat_mode(parser_compat(config.language.compat));
    engine.set_callstack_limit(config.runtime.callstack_limit);
    engine.set_error_namespace(resolved_error_namespace(config));
    if let Some(client_id) = telemetry_client_id() {
        engine.set_telemetry_client_id(Some(client_id));
    }
    Ok(engine)
}
