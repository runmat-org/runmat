use anyhow::{Context, Result};
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_core::RunMatSession;
use runmat_runtime::context::RuntimeGraphicsCallbackQueue;
use std::future::Future;

pub(crate) mod future;

use crate::diagnostics::{parser_compat, resolved_error_namespace};
use crate::telemetry::{sink as runtime_sink, telemetry_client_id};

pub(crate) fn create_session(
    enable_jit: bool,
    verbose: bool,
    config: &RunMatRuntimeConfig,
    create_error_context: &'static str,
) -> Result<RunMatSession> {
    create_session_with_graphics(
        enable_jit,
        verbose,
        config,
        create_error_context,
        &runmat_core::SessionGraphicsHost::Unavailable,
    )
}

pub(crate) fn create_session_with_graphics(
    enable_jit: bool,
    verbose: bool,
    config: &RunMatRuntimeConfig,
    create_error_context: &'static str,
    graphics: &runmat_core::SessionGraphicsHost,
) -> Result<RunMatSession> {
    let mut engine =
        RunMatSession::with_options(enable_jit, verbose).context(create_error_context)?;
    let mut effective = config.clone();
    effective.jit.enabled = enable_jit;
    engine
        .apply_runtime_config(&effective)
        .context("failed to apply runtime settings")?;
    let execution = runmat_execution_runner_native::NativeExecutionService::new(
        runmat_execution_runner_native::NativeExecutionConfig::for_current_executable()
            .context("failed to configure local process execution")?,
    )
    .context("failed to initialize local process execution")?;
    engine.install_execution_services(std::rc::Rc::new(execution));
    engine.install_analysis_service(std::rc::Rc::new(
        crate::commands::fea::NativeAnalysisService::new(parser_compat(config.language.compat)),
    ));
    engine.set_telemetry_consent(config.telemetry.enabled);
    engine.set_telemetry_sink(runtime_sink());
    engine.set_error_namespace(resolved_error_namespace(config));
    engine
        .attach_graphics_host(graphics)
        .context("failed to attach native graphics host")?;
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

/// A local callback diagnostic must not drop/recreate the REPL command or
/// its input future. Fatal pump authority failures still stop the caller.
pub(crate) async fn drive_repl_with_graphics_callbacks<F: Future>(
    callbacks: Option<RuntimeGraphicsCallbackQueue>,
    execution: F,
) -> Result<F::Output> {
    let Some(callbacks) = callbacks else {
        return Ok(execution.await);
    };
    let mut execution = Box::pin(execution);
    loop {
        match callbacks
            .drive_execution_with_host_callbacks_classified(execution.as_mut())
            .await
        {
            Ok(result) => return Ok(result),
            Err(runmat_runtime::context::GraphicsCallbackDriveError::Callback(error)) => {
                eprintln!(
                    "{}: {error}",
                    crate::presentation::stderr().error("Graphics callback error")
                );
            }
            Err(runmat_runtime::context::GraphicsCallbackDriveError::Service(error)) => {
                return Err(error).context("session graphics callback service failed");
            }
        }
    }
}
