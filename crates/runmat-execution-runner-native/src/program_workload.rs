//! Interpreter unit requests enter core's full session/program context.
use runmat_execution_artifact::{ProgramExecutionRequest, ProgramExecutionResponse};
use std::sync::{atomic::AtomicBool, Arc};

pub(crate) async fn execute_unit(
    request: &ProgramExecutionRequest,
    project: Option<&runmat_package::FrozenProjectHandoff>,
    environment: Option<&[(String, String)]>,
    cancellation: Option<Arc<AtomicBool>>,
) -> ProgramExecutionResponse {
    if request.console.is_some() {
        return ProgramExecutionResponse::Failure {
            message: "requested console capture requires an owning transport".into(),
        };
    }
    execute_unit_with_console(request, project, environment, cancellation, None)
        .await
        .0
}

pub(crate) async fn execute_unit_with_console(
    request: &ProgramExecutionRequest,
    project: Option<&runmat_package::FrozenProjectHandoff>,
    environment: Option<&[(String, String)]>,
    cancellation: Option<Arc<AtomicBool>>,
    sink: Option<std::rc::Rc<dyn runmat_runtime::context::PortableConsoleSink>>,
) -> (
    ProgramExecutionResponse,
    Option<runmat_execution::console::ConsoleReceipt>,
) {
    let (response, console) =
        execute_unit_with_transcript(request, project, environment, cancellation, sink).await;
    let receipt = match console.map(|console| console.receipt()).transpose() {
        Ok(receipt) => receipt,
        Err(error) => {
            return (
                ProgramExecutionResponse::Failure {
                    message: error.to_string(),
                },
                None,
            )
        }
    };
    (response, receipt)
}

/// Terminal transports take the original bounded capture, rather than another
/// full-size receiver. Streaming process transports continue to use its receipt.
pub(crate) async fn execute_unit_with_transcript(
    request: &ProgramExecutionRequest,
    project: Option<&runmat_package::FrozenProjectHandoff>,
    environment: Option<&[(String, String)]>,
    cancellation: Option<Arc<AtomicBool>>,
    sink: Option<std::rc::Rc<dyn runmat_runtime::context::PortableConsoleSink>>,
) -> (
    ProgramExecutionResponse,
    Option<runmat_execution::console::ConsoleTranscript>,
) {
    let host = match request
        .console
        .map(|limits| runmat_runtime::context::PortableConsoleHost::with_sink(limits, sink))
        .transpose()
    {
        Ok(host) => host.map(std::rc::Rc::new),
        Err(error) => {
            return (
                ProgramExecutionResponse::Failure {
                    message: error.to_string(),
                },
                None,
            )
        }
    };
    let result = invoke_unit(
        request,
        project,
        environment,
        cancellation.clone(),
        host.as_ref(),
    )
    .await;
    let response = match result {
        Ok(outputs) => ProgramExecutionResponse::from_success_outputs(outputs, vec![]),
        Err(message) => ProgramExecutionResponse::Failure { message },
    };
    let completed = cancellation
        .as_ref()
        .is_none_or(|signal| !signal.load(std::sync::atomic::Ordering::Acquire));
    let console = match host.map(|host| host.finish(completed)).transpose() {
        Ok(transcript) => transcript,
        Err(error) => {
            return (
                ProgramExecutionResponse::Failure {
                    message: error.to_string(),
                },
                None,
            )
        }
    };
    (response, console)
}

async fn invoke_unit(
    request: &ProgramExecutionRequest,
    project: Option<&runmat_package::FrozenProjectHandoff>,
    environment: Option<&[(String, String)]>,
    cancellation: Option<Arc<AtomicBool>>,
    host: Option<&std::rc::Rc<runmat_runtime::context::PortableConsoleHost>>,
) -> Result<Vec<runmat_execution::value::ValuePayload>, String> {
    let mut session = prepare_session(request, project, environment).await?;
    if let Some(host) = host {
        session.install_host_service(host.clone());
    } else {
        session.install_host_service(std::rc::Rc::new(
            runmat_runtime::context::NonCapturingPortableHost,
        ));
    }
    // Core derives invocation views from this same session state. Retain the
    // original scheduler assignment/job across execution and result encoding;
    // an initialized provider does not grant this CPU-only attempt a device.
    let runtime = session.execution_runtime_context(request.recipe.program_revision.clone());
    let _assignment = runtime.enter_execution_assignment(request.assignment.clone());
    let _job = runtime.enter_execution_job(request.job_id);
    let mut control = runmat_core::InvocationControl::default();
    if let Some(cancellation) = cancellation {
        control = control.with_cancellation(
            runmat_execution::cancellation::CancellationToken::from_shared(cancellation),
        );
    }
    let envelope = request
        .artifact
        .executable_unit()
        .map_err(|error| error.to_string())?
        .ok_or("portable worker requires a complete executable unit")?;
    let arguments = request
        .arguments
        .iter()
        .map(runmat_runtime::execution::value_codec::decode_inline_value)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    let outputs = session
        .invoke_portable_unit(
            &envelope,
            arguments,
            usize::from(request.requested_outputs),
            &control,
        )
        .await?;
    runmat_runtime::execution::value_codec::encode_inline_sequence(
        outputs.value(),
        usize::from(request.requested_outputs),
    )
    .map_err(|error| error.to_string())
}

async fn prepare_session(
    request: &ProgramExecutionRequest,
    project: Option<&runmat_package::FrozenProjectHandoff>,
    environment: Option<&[(String, String)]>,
) -> Result<runmat_core::RunMatSession, String> {
    request
        .validate_for_portable_host()
        .map_err(|error| error.to_string())?;
    let mode: runmat_core::CompatMode = serde_json::from_value(serde_json::Value::String(
        request
            .recipe
            .program_revision
            .environment()
            .compatibility_mode
            .as_ref()
            .to_string(),
    ))
    .map_err(|error| format!("unsupported portable language mode: {error}"))?;
    if environment.is_some_and(|entries| entries != request.environment.entries()) {
        return Err("portable invocation environment differs from admitted bundle context".into());
    }
    let mut session = runmat_core::RunMatSession::with_frozen_environment(
        false,
        false,
        request.environment.entries().to_vec(),
    )
    .map_err(|error| format!("failed to initialize portable execution session: {error}"))?;
    session.set_compat_mode(mode);
    if let Some(project) = project {
        session
            .install_project_handoff(project.clone())
            .await
            .map_err(|error| format!("failed to install exact portable project: {error}"))?;
    }
    Ok(session)
}

#[cfg(test)]
mod tests;
