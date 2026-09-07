use runmat_builtins::{BuiltinErrorDescriptor, MTIMES_ERROR_INTERNAL, MTIMES_ERROR_INVALID_INPUT};

use crate::{build_runtime_error, RuntimeError};

use super::NAME;

pub(super) fn descriptor(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    with_message(error.message, error)
}

pub(super) fn with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn invalid_input(message: impl Into<String>) -> RuntimeError {
    with_message(message, &MTIMES_ERROR_INVALID_INPUT)
}

pub(super) fn internal(message: impl Into<String>) -> RuntimeError {
    with_message(message, &MTIMES_ERROR_INTERNAL)
}

pub(super) fn map_control_flow(error: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(error.message()).with_builtin(NAME);
    if let Some(identifier) = error.identifier() {
        builder = builder.with_identifier(identifier.to_string());
    }
    if let Some(task_id) = error.context.task_id.clone() {
        builder = builder.with_task_id(task_id);
    }
    if !error.context.call_stack.is_empty() {
        builder = builder.with_call_stack(error.context.call_stack.clone());
    }
    if let Some(phase) = error.context.phase.clone() {
        builder = builder.with_phase(phase);
    }
    builder.with_source(error).build()
}
