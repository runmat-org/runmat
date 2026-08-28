use thiserror::Error;

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ContractError {
    #[error("invalid {field}: {reason}")]
    Invalid { field: &'static str, reason: String },
    #[error("{field} exceeds its limit of {limit}")]
    Limit { field: &'static str, limit: u64 },
    #[error("unsupported schema {actual}; supported schema is {supported}")]
    UnsupportedSchema { actual: u16, supported: u16 },
    #[error("protocol message is malformed: {0}")]
    MalformedProtocol(String),
}

impl ContractError {
    pub(crate) fn invalid(field: &'static str, reason: impl Into<String>) -> Self {
        Self::Invalid {
            field,
            reason: reason.into(),
        }
    }
}
