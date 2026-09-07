use runmat_builtins::{
    BuiltinErrorDescriptor, MPOWER_ERROR_INTERNAL, MPOWER_ERROR_INVALID_ARGUMENT,
    MPOWER_ERROR_INVALID_INPUT,
};

use crate::{build_runtime_error, RuntimeError};

use super::NAME;

fn with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn invalid_argument(message: impl Into<String>) -> RuntimeError {
    with_message(message, &MPOWER_ERROR_INVALID_ARGUMENT)
}

pub(super) fn invalid_input(message: impl Into<String>) -> RuntimeError {
    with_message(message, &MPOWER_ERROR_INVALID_INPUT)
}

pub(super) fn internal(message: impl Into<String>) -> RuntimeError {
    with_message(message, &MPOWER_ERROR_INTERNAL)
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

pub(super) fn map_host_power(
    error: crate::builtins::common::elementwise::MatrixPowerEvaluationError,
) -> RuntimeError {
    use crate::builtins::common::elementwise::MatrixPowerEvaluationError;
    use crate::builtins::common::matrix::MatrixPowerError;

    let message = error.to_string();
    match error {
        MatrixPowerEvaluationError::InvalidExponent(_)
        | MatrixPowerEvaluationError::IntegerArithmetic(_)
        | MatrixPowerEvaluationError::Matrix(MatrixPowerError::NegativeExponent) => {
            invalid_argument(message)
        }
        MatrixPowerEvaluationError::Matrix(MatrixPowerError::IdentitySizeOverflow)
        | MatrixPowerEvaluationError::Matrix(MatrixPowerError::StorageClassChanged { .. })
        | MatrixPowerEvaluationError::Matrix(MatrixPowerError::Storage(_)) => internal(message),
        MatrixPowerEvaluationError::Matrix(MatrixPowerError::NonSquare { .. })
        | MatrixPowerEvaluationError::UnsupportedOperands(_) => invalid_input(message),
    }
}
