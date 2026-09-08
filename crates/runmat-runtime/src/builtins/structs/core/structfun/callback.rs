use crate::builtins::common::mapped_callable::{
    CallableCallError, CallableParseError, MappedCallable,
};
use runmat_value::Value;
use std::borrow::Cow;

use super::error;

#[derive(Clone)]
pub(super) struct Callable(MappedCallable);

impl Callable {
    pub(super) fn parse(value: Value) -> crate::BuiltinResult<Self> {
        MappedCallable::parse(value).map(Self).map_err(parse_error)
    }
    pub(super) async fn call(
        &self,
        arguments: &[Value],
        outputs: usize,
    ) -> Result<Value, CallbackFailure> {
        self.0
            .invoke_with_outputs(arguments, outputs)
            .await
            .map_err(CallbackFailure::from)
    }
}

pub(super) enum CallbackFailure {
    Runtime(crate::RuntimeError),
    UndefinedExternal {
        identity: runmat_types::CallableIdentity,
    },
    SemanticUnavailable {
        function_name: String,
        function: String,
    },
}

impl CallbackFailure {
    pub(super) fn identifier(&self) -> Option<&str> {
        match self {
            Self::Runtime(error) => error.identifier(),
            Self::UndefinedExternal { .. } => {
                runmat_builtins::STRUCTFUN_ERROR_UNDEFINED_FUNCTION.identifier
            }
            Self::SemanticUnavailable { .. } => {
                runmat_builtins::STRUCTFUN_ERROR_FUNCTION_ERROR.identifier
            }
        }
    }

    pub(super) fn message(&self) -> Cow<'_, str> {
        match self {
            Self::Runtime(error) => Cow::Borrowed(error.message()),
            Self::UndefinedExternal { identity } => Cow::Owned(format!(
                "Undefined function for callable identity {identity:?}"
            )),
            Self::SemanticUnavailable {
                function_name,
                function,
            } => Cow::Owned(format!(
                "semantic closure '{function_name}' ({function}) is unavailable"
            )),
        }
    }

    pub(super) fn into_runtime_error(self) -> crate::RuntimeError {
        match self {
            Self::Runtime(error) => error::function(error.message()),
            Self::UndefinedExternal { identity } => error::undefined(format!(
                "Undefined function for callable identity {identity:?}"
            )),
            Self::SemanticUnavailable {
                function_name,
                function,
            } => error::function(format!(
                "semantic closure '{function_name}' ({function}) is unavailable"
            )),
        }
    }
}

impl From<CallableCallError> for CallbackFailure {
    fn from(value: CallableCallError) -> Self {
        match value {
            CallableCallError::Runtime(error) => Self::Runtime(error),
            CallableCallError::UndefinedExternal { identity } => {
                Self::UndefinedExternal { identity }
            }
            CallableCallError::SemanticUnavailable {
                function_name,
                function,
            } => Self::SemanticUnavailable {
                function_name,
                function,
            },
        }
    }
}

fn parse_error(value: CallableParseError) -> crate::RuntimeError {
    match value {
        CallableParseError::EmptyText | CallableParseError::EmptyHandle => {
            error::invalid("structfun: function name must not be empty")
        }
        CallableParseError::CharacterNameMustBeRow | CallableParseError::StringNameMustBeScalar => {
            error::invalid("structfun: function name must be a character row or string scalar")
        }
        CallableParseError::ScalarValue => error::invalid(
            "structfun: expected a function handle or function name, not a scalar value",
        ),
        CallableParseError::UnsupportedValue(value) => error::invalid(format!(
            "structfun: expected a function handle or function name, got {value:?}"
        )),
    }
}
