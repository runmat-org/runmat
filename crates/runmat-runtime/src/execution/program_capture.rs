//! Complete program capture through the active session's typed authority.
use crate::{context::RuntimeContext, RuntimeError};
use runmat_types::ProgramFunctionId;

pub fn capture_deferred_program(
    runtime: &RuntimeContext,
    function: ProgramFunctionId,
) -> Result<Option<Vec<u8>>, RuntimeError> {
    if !runtime.execution().requires_program_capture() {
        return Ok(None);
    }
    let program = runtime
        .service_ports()
        .require_program("capture deferred program")
        .map_err(|error| capture_error(error.to_string()))?
        .freeze_executable(function)?;
    if runtime.program_revision() != Some(&program.manifest.identity.program) {
        return Err(capture_error(
            "active program revision differs from deferred callable",
        ));
    }
    program
        .canonical_bytes()
        .map(Some)
        .map_err(|error| capture_error(error.to_string()))
}

fn capture_error(message: impl Into<String>) -> RuntimeError {
    crate::build_runtime_error(message.into())
        .with_identifier("RunMat:ExecutionProgram")
        .build()
}
