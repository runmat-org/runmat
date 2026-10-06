//! Browser artifact transport delegates complete compiler products to core.
use runmat_execution_artifact::{
    ExecutableForm, ProgramExecutionRequest, ProgramExecutionResponse,
};
use runmat_runtime::execution::RuntimeExecutionServices;
use std::rc::Rc;

pub(crate) async fn execute(
    request: ProgramExecutionRequest,
    execution: Option<Rc<dyn RuntimeExecutionServices>>,
) -> ProgramExecutionResponse {
    if request.artifact.form != ExecutableForm::ExecutableUnitV3 {
        return runmat_vm::execute_program_request(request).await;
    }
    match invoke(&request, execution).await {
        Ok(outputs) => ProgramExecutionResponse::from_success_outputs(outputs, vec![]),
        Err(message) => ProgramExecutionResponse::Failure { message },
    }
}

async fn invoke(
    request: &ProgramExecutionRequest,
    execution: Option<Rc<dyn RuntimeExecutionServices>>,
) -> Result<Vec<runmat_execution::value::ValuePayload>, String> {
    request
        .validate_for_portable_host()
        .map_err(|error| error.to_string())?;
    let envelope = request
        .artifact
        .executable_unit()
        .map_err(|error| error.to_string())?
        .ok_or("browser session requires a complete executable unit")?;
    let mode: runmat_core::CompatMode = serde_json::from_value(serde_json::Value::String(
        envelope
            .manifest
            .identity
            .program
            .environment()
            .compatibility_mode
            .clone(),
    ))
    .map_err(|error| format!("unsupported portable language mode: {error}"))?;
    let mut session = runmat_core::RunMatSession::with_options(false, false)
        .map_err(|error| format!("failed to initialize portable browser session: {error}"))?;
    session.set_compat_mode(mode);
    if let Some(execution) = execution {
        session.install_execution_services(execution);
    }
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
            &runmat_core::InvocationControl::default(),
        )
        .await?;
    runmat_runtime::execution::value_codec::encode_inline_outputs(
        &value,
        usize::from(request.requested_outputs),
    )
    .map_err(|error| error.to_string())
}
