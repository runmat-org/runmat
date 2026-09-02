use runmat_builtins::{
    BuiltinErrorDescriptor, RelationalOperator, EQ_ERROR_INVALID_INPUT, EQ_ERROR_SIZE_MISMATCH,
    GE_ERROR_INVALID_INPUT, GE_ERROR_SIZE_MISMATCH, GT_ERROR_INVALID_INPUT, GT_ERROR_SIZE_MISMATCH,
    LE_ERROR_INVALID_INPUT, LE_ERROR_SIZE_MISMATCH, LT_ERROR_INVALID_INPUT, LT_ERROR_SIZE_MISMATCH,
    NE_ERROR_INVALID_INPUT, NE_ERROR_SIZE_MISMATCH,
};

use crate::{build_runtime_error, RuntimeError};

#[derive(Clone, Copy)]
pub(super) enum ComparisonError {
    InvalidInput,
    SizeMismatch,
}

pub(super) fn runtime_error(operator: RelationalOperator, kind: ComparisonError) -> RuntimeError {
    let descriptor = descriptor(operator, kind);
    let mut builder = build_runtime_error(descriptor.message).with_builtin(operator.name());
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn descriptor(
    operator: RelationalOperator,
    kind: ComparisonError,
) -> &'static BuiltinErrorDescriptor {
    use ComparisonError::{InvalidInput, SizeMismatch};
    use RelationalOperator::{
        Equal, GreaterThan, GreaterThanOrEqual, LessThan, LessThanOrEqual, NotEqual,
    };

    match (operator, kind) {
        (Equal, InvalidInput) => &EQ_ERROR_INVALID_INPUT,
        (Equal, SizeMismatch) => &EQ_ERROR_SIZE_MISMATCH,
        (NotEqual, InvalidInput) => &NE_ERROR_INVALID_INPUT,
        (NotEqual, SizeMismatch) => &NE_ERROR_SIZE_MISMATCH,
        (LessThan, InvalidInput) => &LT_ERROR_INVALID_INPUT,
        (LessThan, SizeMismatch) => &LT_ERROR_SIZE_MISMATCH,
        (LessThanOrEqual, InvalidInput) => &LE_ERROR_INVALID_INPUT,
        (LessThanOrEqual, SizeMismatch) => &LE_ERROR_SIZE_MISMATCH,
        (GreaterThan, InvalidInput) => &GT_ERROR_INVALID_INPUT,
        (GreaterThan, SizeMismatch) => &GT_ERROR_SIZE_MISMATCH,
        (GreaterThanOrEqual, InvalidInput) => &GE_ERROR_INVALID_INPUT,
        (GreaterThanOrEqual, SizeMismatch) => &GE_ERROR_SIZE_MISMATCH,
    }
}
