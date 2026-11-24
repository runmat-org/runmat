//! Native FEA execution failures, carried whole across the host boundary.
use runmat_runtime::analysis::storage::AnalysisPublicationError;
use runmat_runtime::operations::OperationErrorEnvelope;

/// A failed native FEA execution.
///
/// An operation failure keeps its complete envelope (error code, type, details,
/// trace and request ids). A publication failure keeps its complete storage
/// outcome, including durability receipts. Hosts read these fields directly.
/// `Display` gives `error_code: message` for an operation failure, which is the
/// text Desktop shows.
#[derive(Debug, Clone, PartialEq)]
pub enum FeaExecutionError {
    Operation(Box<OperationErrorEnvelope>),
    Publication(Box<AnalysisPublicationError>),
    Message(String),
}

impl FeaExecutionError {
    /// The operation envelope, when an analysis operation failed.
    pub fn operation_error(&self) -> Option<&OperationErrorEnvelope> {
        match self {
            Self::Operation(error) => Some(error),
            _ => None,
        }
    }

    /// The storage outcome, when saving the results failed.
    pub fn publication_error(&self) -> Option<&AnalysisPublicationError> {
        match self {
            Self::Publication(error) => Some(error),
            _ => None,
        }
    }
}

impl std::fmt::Display for FeaExecutionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Operation(error) => write!(formatter, "{}: {}", error.error_code, error.message),
            Self::Publication(error) => write!(formatter, "{error}"),
            Self::Message(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for FeaExecutionError {}

impl From<OperationErrorEnvelope> for FeaExecutionError {
    fn from(error: OperationErrorEnvelope) -> Self {
        Self::Operation(Box::new(error))
    }
}

impl From<AnalysisPublicationError> for FeaExecutionError {
    fn from(error: AnalysisPublicationError) -> Self {
        Self::Publication(Box::new(error))
    }
}

impl From<String> for FeaExecutionError {
    fn from(message: String) -> Self {
        Self::Message(message)
    }
}

impl From<&str> for FeaExecutionError {
    fn from(message: &str) -> Self {
        Self::Message(message.to_owned())
    }
}

/// Hosts that report failures as text keep compiling; the text is `Display`.
impl From<FeaExecutionError> for String {
    fn from(error: FeaExecutionError) -> Self {
        error.to_string()
    }
}
