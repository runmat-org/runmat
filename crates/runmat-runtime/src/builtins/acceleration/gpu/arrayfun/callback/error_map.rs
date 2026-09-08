use crate::builtins::common::mapped_callable::{CallableCallError, CallableParseError};
use runmat_builtins::{ARRAYFUN_ERROR_CALLBACK_FAILED, ARRAYFUN_ERROR_UNDEFINED_FUNCTION};

use super::super::error::{arrayfun_error_with_detail, arrayfun_error_with_message, arrayfun_flow};

pub(super) fn parse(error: CallableParseError) -> crate::RuntimeError {
    match error {
        CallableParseError::EmptyText => {
            arrayfun_flow("arrayfun: expected function handle or builtin name, got empty string")
        }
        CallableParseError::EmptyHandle => arrayfun_flow("arrayfun: empty function handle"),
        CallableParseError::CharacterNameMustBeRow | CallableParseError::StringNameMustBeScalar => {
            arrayfun_flow("arrayfun: function name must be a character vector or string scalar")
        }
        CallableParseError::ScalarValue => {
            arrayfun_flow("arrayfun: expected function handle or builtin name, not a scalar value")
        }
        CallableParseError::UnsupportedValue(value) => arrayfun_flow(format!(
            "arrayfun: expected function handle or builtin name, got {value:?}"
        )),
    }
}

pub(super) fn call(error: CallableCallError) -> crate::RuntimeError {
    match error {
        CallableCallError::Runtime(error) => error,
        CallableCallError::UndefinedExternal { identity } => arrayfun_error_with_message(
            format!("Undefined function for callable identity {identity:?}"),
            &ARRAYFUN_ERROR_UNDEFINED_FUNCTION,
        ),
        CallableCallError::SemanticUnavailable {
            function_name,
            function,
        } => arrayfun_error_with_detail(
            &ARRAYFUN_ERROR_CALLBACK_FAILED,
            format!("semantic closure '{function_name}' ({function}) is unavailable"),
        ),
    }
}
