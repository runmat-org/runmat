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
    if request.console.is_some() {
        return ProgramExecutionResponse::Failure {
            message: "requested console capture requires an owning transport".into(),
        };
    }
    if request.artifact.form != ExecutableForm::ExecutableUnitV4 {
        return super::execute_local_program(request).await;
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
    super::admission::validate_browser_admission(request)?;
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
            .as_ref()
            .to_string(),
    ))
    .map_err(|error| format!("unsupported portable language mode: {error}"))?;
    let mut session = runmat_core::RunMatSession::with_frozen_environment(
        false,
        false,
        request.environment.entries().to_vec(),
    )
    .map_err(|error| format!("failed to initialize portable browser session: {error}"))?;
    session.set_compat_mode(mode);
    session.install_host_service(Rc::new(runmat_runtime::context::NonCapturingPortableHost));
    if let Some(execution) = execution {
        session.install_execution_services(execution);
    }
    // Core session projections share this actual invocation state. Retain the
    // original assignment/job through execution AND portable output encoding.
    let selected_runtime =
        session.execution_runtime_context(request.recipe.program_revision.clone());
    let _assignment = selected_runtime.enter_execution_assignment(request.assignment.clone());
    let _job = selected_runtime.enter_execution_job(request.job_id);
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
            &runmat_core::InvocationControl::default(),
        )
        .await?;
    runmat_runtime::execution::value_codec::encode_inline_sequence(
        outputs.value(),
        usize::from(request.requested_outputs),
    )
    .map_err(|error| error.to_string())
}
