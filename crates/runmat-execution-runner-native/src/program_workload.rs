//! Interpreter unit requests enter core's full session/program context.
use runmat_execution_artifact::{ProgramExecutionRequest, ProgramExecutionResponse};
use std::sync::{atomic::AtomicBool, Arc};

pub(crate) async fn execute_unit(
    request: &ProgramExecutionRequest,
    project: Option<&runmat_package::FrozenProjectHandoff>,
    environment: Option<&[(String, String)]>,
    cancellation: Option<Arc<AtomicBool>>,
) -> ProgramExecutionResponse {
    match invoke_unit(request, project, environment, cancellation).await {
        Ok(outputs) => ProgramExecutionResponse::from_success_outputs(outputs, vec![]),
        Err(message) => ProgramExecutionResponse::Failure { message },
    }
}

async fn invoke_unit(
    request: &ProgramExecutionRequest,
    project: Option<&runmat_package::FrozenProjectHandoff>,
    environment: Option<&[(String, String)]>,
    cancellation: Option<Arc<AtomicBool>>,
) -> Result<Vec<runmat_execution::value::ValuePayload>, String> {
    let mut session = prepare_session(request, project, environment)?;
    let mut control = runmat_core::InvocationControl::default();
    if let Some(cancellation) = cancellation {
        control = control.with_cancellation(cancellation);
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
    let value = session
        .invoke_portable_unit(
            &envelope,
            arguments,
            usize::from(request.requested_outputs),
            &control,
        )
        .await?;
    runmat_runtime::execution::value_codec::encode_inline_outputs(
        &value,
        usize::from(request.requested_outputs),
    )
    .map_err(|error| error.to_string())
}

fn prepare_session(
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
            .clone(),
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
            .map_err(|error| format!("failed to install exact portable project: {error}"))?;
    }
    Ok(session)
}

#[cfg(test)]
mod tests;
