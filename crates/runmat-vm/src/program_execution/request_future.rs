//! Prepare one request route before its child is polled.
//!
//! Construction and decoding return before entering the interpreter poll chain.
//! The existing single branch allocation owns moved requests and decoded metadata;
//! no workspace or numeric payload is copied. Process routes remain on the native
//! stack required by their foreign hosts.

use super::*;
use std::future::Future;
use std::pin::Pin;

#[path = "request_future/unit.rs"]
mod unit;

pub(super) type PreparedRequest = Pin<Box<dyn Future<Output = ProgramExecutionResponse>>>;

pub(super) struct ProgramPreparationError {
    message: String,
}

impl ProgramPreparationError {
    pub(super) fn into_response(self) -> ProgramExecutionResponse {
        ProgramExecutionResponse::Failure {
            message: self.message,
        }
    }
}

#[inline(never)]
pub(super) fn prepare(
    request: ProgramExecutionRequest,
    runtime: runmat_runtime::context::RuntimeContext,
    bundle: Option<Box<runmat_execution_artifact::ExecutionBundle>>,
) -> Result<PreparedRequest, ProgramPreparationError> {
    if bundle.is_some() && request.artifact.form != ExecutableForm::ExecutableUnitV4 {
        return Err(ProgramPreparationError {
            message: "worker rejected selected foreign bundle for a non-V3 program".into(),
        });
    }
    if request.artifact.form == ExecutableForm::InterpreterScriptV1 {
        return Err(ProgramPreparationError {
            message: format!(
                "worker rejected legacy script bytecode; rebuild the program for bytecode schema {}",
                crate::BYTECODE_SCHEMA_VERSION
            ),
        });
    }
    if request.artifact.form == ExecutableForm::InterpreterScriptV2
        && !matches!(
            &request.callable,
            ProgramCallable::ParallelRegion { .. } | ProgramCallable::SpmdRegion { .. }
        )
    {
        return Ok(Box::pin(execute_script_request(request, runtime)));
    }
    if request.artifact.form == ExecutableForm::ExecutableUnitV4 {
        return unit::prepare(request, runtime, bundle);
    }
    if matches!(
        &request.callable,
        ProgramCallable::ParallelRegion { .. } | ProgramCallable::SpmdRegion { .. }
    ) && request.artifact.form != ExecutableForm::InterpreterRegionSourcesV1
    {
        return Err(ProgramPreparationError {
            message: "worker rejected source-less region bytecode; rebuild the parallel program"
                .into(),
        });
    }
    if request.artifact.form == ExecutableForm::InterpreterBytecodeV1 {
        return Err(ProgramPreparationError {
            message: format!(
                "worker rejected legacy interpreter bytecode; expected schema {}. Rebuild the program with this RunMat version",
                crate::BYTECODE_SCHEMA_VERSION
            ),
        });
    }
    if request.artifact.form != ExecutableForm::InterpreterBytecodeV2
        && request.artifact.form != ExecutableForm::InterpreterScriptV2
        && request.artifact.form != ExecutableForm::InterpreterRegionSourcesV1
    {
        return Err(ProgramPreparationError {
            message:
                "worker rejected an unsupported interpreter artifact form; rebuild the program"
                    .into(),
        });
    }
    if matches!(request.callable, ProgramCallable::ParallelRegion { .. }) {
        return Ok(Box::pin(execute_parallel_region_request(request, runtime)));
    }
    if matches!(request.callable, ProgramCallable::SpmdRegion { .. }) {
        return Ok(Box::pin(execute_spmd_region_request(request, runtime)));
    }
    let registry: crate::FunctionRegistry =
        match crate::decode_interpreter_program_v2(&request.artifact.executable_bytes) {
            Ok(registry) => registry,
            Err(error) => {
                return Err(ProgramPreparationError {
                    message: format!("worker rejected an invalid program: {error}"),
                });
            }
        };
    let origin_program = semantic_identity::revision_owner(
        &request.recipe.program_revision,
        &request.artifact.executable_bytes,
    )
    .map_err(|message| ProgramPreparationError { message })?;
    let capabilities = match callable_capabilities(&registry, &request.callable, origin_program) {
        Ok(capabilities) => capabilities,
        Err(message) => return Err(ProgramPreparationError { message }),
    };
    if !recipe_supports_capabilities(&request.recipe, &capabilities) {
        return Err(ProgramPreparationError {
            message: "worker rejected a recipe that weakens callable accelerator requirements"
                .into(),
        });
    }
    Ok(Box::pin(execute_function_request(
        request, registry, runtime, None,
    )))
}
