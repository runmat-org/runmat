use runmat_builtins::{
    BuiltinErrorDescriptor, MLDIVIDE_ERROR_INTERNAL, MLDIVIDE_ERROR_INVALID_INPUT,
    MRDIVIDE_ERROR_INTERNAL, MRDIVIDE_ERROR_INVALID_INPUT,
};

use crate::{build_runtime_error, RuntimeError};

use super::SolveOrientation;

pub(super) fn invalid_input(
    orientation: SolveOrientation,
    message: impl Into<String>,
) -> RuntimeError {
    described(orientation, message, invalid_descriptor(orientation))
}

pub(super) fn internal(orientation: SolveOrientation, message: impl Into<String>) -> RuntimeError {
    described(orientation, message, internal_descriptor(orientation))
}

pub(super) fn map_control_flow(orientation: SolveOrientation, error: RuntimeError) -> RuntimeError {
    if error.message() == "interaction pending..." {
        return build_runtime_error("interaction pending...")
            .with_builtin(orientation.name())
            .build();
    }
    let mut builder = build_runtime_error(error.message()).with_builtin(orientation.name());
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

fn described(
    orientation: SolveOrientation,
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(orientation.name());
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

const fn invalid_descriptor(orientation: SolveOrientation) -> &'static BuiltinErrorDescriptor {
    match orientation {
        SolveOrientation::Left => &MLDIVIDE_ERROR_INVALID_INPUT,
        SolveOrientation::Right => &MRDIVIDE_ERROR_INVALID_INPUT,
    }
}

const fn internal_descriptor(orientation: SolveOrientation) -> &'static BuiltinErrorDescriptor {
    match orientation {
        SolveOrientation::Left => &MLDIVIDE_ERROR_INTERNAL,
        SolveOrientation::Right => &MRDIVIDE_ERROR_INTERNAL,
    }
}
