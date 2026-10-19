use crate::builtins::common::mapped_callable::{CallableCallError, CallableParseError};

use super::super::error;

pub(super) fn parse(parse_error: CallableParseError) -> crate::RuntimeError {
    match parse_error {
        CallableParseError::EmptyText => {
            error::invalid("cellfun: expected function handle or builtin name, got empty string")
        }
        CallableParseError::EmptyHandle => error::invalid("cellfun: empty function handle"),
        CallableParseError::CharacterNameMustBeRow | CallableParseError::StringNameMustBeScalar => {
            error::invalid("cellfun: function name must be a character vector or string scalar")
        }
        CallableParseError::ScalarValue => {
            error::invalid("cellfun: expected function handle or builtin name, not a scalar value")
        }
        CallableParseError::UnsupportedValue(value) => error::invalid(format!(
            "cellfun: expected function handle or builtin name, got {value:?}"
        )),
    }
}

pub(super) fn call(call_error: CallableCallError) -> crate::RuntimeError {
    match call_error {
        CallableCallError::Runtime(error) => error,
        CallableCallError::UndefinedExternal { identity } => error::undefined(format!(
            "Undefined function for callable identity {identity:?}"
        )),
        CallableCallError::SemanticUnavailable {
            function_name,
            function,
        } => error::invalid(format!(
            "cellfun: semantic closure '{function_name}' ({function}) is unavailable"
        )),
    }
}
