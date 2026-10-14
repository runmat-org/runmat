use anyhow::{Context, Result};
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_core::RunMatSession;
use runmat_runtime::context::RuntimeGraphicsCallbackQueue;
use std::future::Future;

use crate::diagnostics::{parser_compat, resolved_error_namespace};
use crate::telemetry::{sink as runtime_sink, telemetry_client_id};

pub(crate) fn create_session(
    enable_jit: bool,
    verbose: bool,
    config: &RunMatRuntimeConfig,
    create_error_context: &'static str,
) -> Result<RunMatSession> {
    let mut engine =
        RunMatSession::with_options(enable_jit, verbose).context(create_error_context)?;
    let execution = runmat_execution_runner_native::NativeExecutionService::new(
        runmat_execution_runner_native::NativeExecutionConfig::for_current_executable()
            .context("failed to locate the RunMat execution worker executable")?,
    )
    .context("failed to initialize local process execution")?;
    engine.install_execution_services(std::rc::Rc::new(execution));
    engine.set_telemetry_consent(config.telemetry.enabled);
    engine.set_telemetry_sink(runtime_sink());
    engine.set_compat_mode(parser_compat(config.language.compat));
    engine.set_callstack_limit(config.runtime.callstack_limit);
    engine.set_error_namespace(resolved_error_namespace(config));
    engine.set_mex_config(&config.foreign.mex);
    engine
        .set_native_ffi_config(&config.foreign.native)
        .context("failed to configure native-library isolation")?;
    engine
        .set_java_config(&config.foreign.java)
        .context("failed to configure Java runtime")?;
    engine
        .set_python_config(&config.foreign.python)
        .context("failed to configure Python runtime")?;
    engine.set_selected_foreign_config(std::sync::Arc::new(config.clone()));
    if let Some(client_id) = telemetry_client_id() {
        engine.set_telemetry_client_id(Some(client_id));
    }
    Ok(engine)
}

/// Drive admitted GUI callbacks on the selected session lane while an
/// execution future is suspended. The queue serializes this with drawnow.
pub(crate) async fn drive_with_graphics_callbacks<F: Future>(
    callbacks: Option<RuntimeGraphicsCallbackQueue>,
    execution: F,
) -> Result<F::Output> {
    match callbacks {
        Some(callbacks) => callbacks
            .drive_execution_with_host_callbacks(execution)
            .await
            .context("session graphics callback failed"),
        None => Ok(execution.await),
    }
}
