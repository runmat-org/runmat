#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpreterPayloadForm {
    ScriptV2,
    ProgramV2,
}

impl std::fmt::Display for InterpreterPayloadForm {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::ScriptV2 => "interpreter script V2",
            Self::ProgramV2 => "interpreter program V2",
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum InterpreterPayloadError {
    LegacyPayload {
        expected: InterpreterPayloadForm,
    },
    InvalidEnvelope {
        form: InterpreterPayloadForm,
        reason: String,
    },
    UnsupportedRevision {
        field: InterpreterRevisionField,
        actual: u16,
        expected: u16,
    },
}

impl std::fmt::Display for InterpreterPayloadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LegacyPayload { expected } => {
                write!(
                    formatter,
                    "raw legacy interpreter payload; expected {expected} envelope"
                )
            }
            Self::InvalidEnvelope { form, reason } => {
                write!(formatter, "invalid {form} envelope: {reason}")
            }
            Self::UnsupportedRevision {
                field,
                actual,
                expected,
            } => write!(
                formatter,
                "unsupported {field} revision: actual {actual}, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for InterpreterPayloadError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpreterRevisionField {
    Bytecode,
    FunctionRegistry,
}

impl std::fmt::Display for InterpreterRevisionField {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Bytecode => "bytecode schema",
            Self::FunctionRegistry => "function registry schema",
        })
    }
}

pub(super) fn invalid(
    form: InterpreterPayloadForm,
    reason: impl Into<String>,
) -> InterpreterPayloadError {
    InterpreterPayloadError::InvalidEnvelope {
        form,
        reason: reason.into(),
    }
}
