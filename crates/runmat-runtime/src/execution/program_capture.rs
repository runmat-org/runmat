//! Complete program capture through the active session's typed authority.
use crate::{context::RuntimeContext, RuntimeError};
use runmat_execution::FunctionBindingOrigin;

/// Inseparable original callable revision and its complete portable product.
pub struct DeferredProgramCapture {
    pub revision: runmat_execution::ProgramRevision,
    pub bytes: Vec<u8>,
}

pub fn capture_deferred_program(
    runtime: &RuntimeContext,
    origin: FunctionBindingOrigin,
) -> Result<Option<DeferredProgramCapture>, RuntimeError> {
    if !runtime.execution().requires_program_capture() {
        return Ok(None);
    }
    let invocation_revision = runtime
        .program_revision()
        .ok_or_else(|| capture_error("active invocation has no original program revision"))?;
    let program = runtime
        .service_ports()
        .require_program("capture deferred program")
        .map_err(|error| capture_error(error.to_string()))?
        .freeze_executable(origin, invocation_revision)?;
    let bytes = program
        .canonical_bytes()
        .map_err(|error| capture_error(error.to_string()))?;
    Ok(Some(DeferredProgramCapture {
        revision: program.manifest.identity.program,
        bytes,
    }))
}

fn capture_error(message: impl Into<String>) -> RuntimeError {
    crate::build_runtime_error(message.into())
        .with_identifier("RunMat:ExecutionProgram")
        .build()
}

/// Freeze invocation state separately from the original compiler product.
pub fn capture_deferred_environment(
    runtime: &RuntimeContext,
) -> Result<runmat_execution::EnvironmentSnapshot, RuntimeError> {
    runmat_execution::EnvironmentSnapshot::new(runtime.environment_snapshot())
        .map_err(|error| capture_error(error.to_string()))
}
