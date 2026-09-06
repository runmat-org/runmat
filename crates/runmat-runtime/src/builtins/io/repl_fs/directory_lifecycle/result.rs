use runmat_value::{CharArray, Value};

use crate::warnings::{self, WarningRequest};
use crate::{build_runtime_error, BuiltinResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DirectoryOutcome {
    Success,
    Notice { message: String, identifier: String },
    Failure { message: String, identifier: String },
}

impl DirectoryOutcome {
    pub(crate) fn notice(message: impl Into<String>, identifier: impl Into<String>) -> Self {
        Self::Notice {
            message: message.into(),
            identifier: identifier.into(),
        }
    }

    pub(crate) fn failure(message: impl Into<String>, identifier: impl Into<String>) -> Self {
        Self::Failure {
            message: message.into(),
            identifier: identifier.into(),
        }
    }

    pub(crate) fn status(&self) -> bool {
        !matches!(self, Self::Failure { .. })
    }

    pub(crate) fn message(&self) -> &str {
        match self {
            Self::Success => "",
            Self::Notice { message, .. } | Self::Failure { message, .. } => message,
        }
    }

    pub(crate) fn identifier(&self) -> &str {
        match self {
            Self::Success => "",
            Self::Notice { identifier, .. } | Self::Failure { identifier, .. } => identifier,
        }
    }

    fn outputs(&self) -> Vec<Value> {
        vec![
            Value::Bool(self.status()),
            Value::CharArray(CharArray::new_row(self.message())),
            Value::CharArray(CharArray::new_row(self.identifier())),
        ]
    }
}

pub(crate) fn complete(outcome: DirectoryOutcome, builtin: &'static str) -> BuiltinResult<Value> {
    match crate::output_count::current_output_count() {
        Some(0) => complete_without_outputs(outcome, builtin),
        Some(count) => Ok(crate::output_count::output_list_with_padding(
            count,
            outcome.outputs(),
        )),
        None => Ok(Value::Bool(outcome.status())),
    }
}

fn complete_without_outputs(
    outcome: DirectoryOutcome,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match outcome {
        DirectoryOutcome::Success => {}
        DirectoryOutcome::Notice {
            message,
            identifier,
        } => warnings::emit(WarningRequest {
            builtin,
            identifier: &identifier,
            message: &message,
            show_identifier: false,
        })?,
        DirectoryOutcome::Failure {
            message,
            identifier,
        } => {
            return Err(build_runtime_error(message)
                .with_builtin(builtin)
                .with_identifier(identifier)
                .build())
        }
    }
    Ok(Value::OutputList(Vec::new()))
}
