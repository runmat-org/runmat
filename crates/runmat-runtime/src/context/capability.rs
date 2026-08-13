use crate::{build_runtime_error, RuntimeError};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Stable families of optional runtime capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeCapability {
    Call,
    Workspace,
    Object,
    Host,
    Error,
    Cancellation,
    Acceleration,
    Native,
    Foreign,
    Parallel,
}

impl RuntimeCapability {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Call => "call",
            Self::Workspace => "workspace",
            Self::Object => "object",
            Self::Host => "host",
            Self::Error => "error",
            Self::Cancellation => "cancellation",
            Self::Acceleration => "acceleration",
            Self::Native => "native",
            Self::Foreign => "foreign",
            Self::Parallel => "parallel",
        }
    }
}

/// Stable failure returned when an invocation requests a service its host did
/// not compose. This is intentionally target-neutral: browser/WASM reports the
/// same capability failure as a native host with that service disabled.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeCapabilityError {
    pub capability: RuntimeCapability,
    pub operation: String,
}

impl RuntimeCapabilityError {
    pub const IDENTIFIER: &'static str = "RunMat:RuntimeContext:CapabilityUnavailable";

    pub fn new(capability: RuntimeCapability, operation: impl Into<String>) -> Self {
        Self {
            capability,
            operation: operation.into(),
        }
    }

    pub fn into_runtime_error(self) -> RuntimeError {
        build_runtime_error(self.to_string())
            .with_identifier(Self::IDENTIFIER)
            .build()
    }
}

impl fmt::Display for RuntimeCapabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "runtime capability '{}' is unavailable for {}",
            self.capability.as_str(),
            self.operation
        )
    }
}

impl std::error::Error for RuntimeCapabilityError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_failure_has_stable_identifier_and_payload() {
        let error = RuntimeCapabilityError::new(RuntimeCapability::Foreign, "invoke JNI callback");
        let json = serde_json::to_string(&error).expect("serialize capability error");
        assert_eq!(
            json,
            r#"{"capability":"foreign","operation":"invoke JNI callback"}"#
        );
        let runtime = error.into_runtime_error();
        assert_eq!(
            runtime.identifier(),
            Some(RuntimeCapabilityError::IDENTIFIER)
        );
        assert!(runtime.to_string().contains("foreign"));
    }
}
