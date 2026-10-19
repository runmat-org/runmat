//! Publication failure retains uncertainty without turning it into numerical rejection.
use super::AnalysisPublicationReceipt;
use runmat_filesystem::workspace::WorkspacePublicationReceipt;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum AnalysisPublicationError {
    Rejected {
        message: String,
    },
    ResourceLimit {
        message: String,
    },
    /// An immutable sidecar exists but fresh file/directory synchronization failed.
    /// There is no whole-study commit receipt at this preparation boundary.
    EvidenceDurabilityIndeterminate {
        path: std::path::PathBuf,
        cause: String,
    },
    DurabilityIndeterminate {
        workspace: WorkspacePublicationReceipt,
        /// None when access is blocked by an earlier workspace transaction.
        publication: Option<Box<AnalysisPublicationReceipt>>,
    },
}
impl AnalysisPublicationError {
    pub(crate) fn context(self, context: &str) -> Self {
        match self {
            Self::Rejected { message } => Self::Rejected {
                message: format!("{context}: {message}"),
            },
            Self::ResourceLimit { message } => Self::ResourceLimit {
                message: format!("{context}: {message}"),
            },
            uncertain => uncertain,
        }
    }
    pub(crate) fn from_io(error: std::io::Error) -> Self {
        if runmat_filesystem::workspace_access::resource_limit(&error).is_some() {
            return Self::ResourceLimit {
                message: error.to_string(),
            };
        }
        if let Some(codec) = error
            .get_ref()
            .and_then(|cause| cause.downcast_ref::<runmat_canonical_codec::CanonicalCodecError>())
        {
            return codec.clone().into();
        }
        match runmat_filesystem::workspace::indeterminate_receipt(&error) {
            Some(receipt) => Self::DurabilityIndeterminate {
                workspace: receipt.clone(),
                publication: None,
            },
            None => error.to_string().into(),
        }
    }
    pub(super) fn with_publication(self, receipt: AnalysisPublicationReceipt) -> Self {
        match self {
            Self::DurabilityIndeterminate { workspace, .. } => Self::DurabilityIndeterminate {
                workspace,
                publication: Some(Box::new(receipt)),
            },
            other => other,
        }
    }
}
impl From<String> for AnalysisPublicationError {
    fn from(message: String) -> Self {
        Self::Rejected { message }
    }
}
impl From<&str> for AnalysisPublicationError {
    fn from(message: &str) -> Self {
        message.to_owned().into()
    }
}
impl std::fmt::Display for AnalysisPublicationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Native hosts still have textual error envelopes. Preserve the complete typed
        // outcome there rather than dropping its original receipt into a short message.
        match self {
            Self::Rejected { message } => f.write_str(message),
            Self::ResourceLimit { .. }
            | Self::DurabilityIndeterminate { .. }
            | Self::EvidenceDurabilityIndeterminate { .. } => {
                let value = serde_json::to_string(self).map_err(|_| std::fmt::Error)?;
                f.write_str(&value)
            }
        }
    }
}
impl std::error::Error for AnalysisPublicationError {}

impl From<runmat_canonical_codec::CanonicalCodecError> for AnalysisPublicationError {
    fn from(error: runmat_canonical_codec::CanonicalCodecError) -> Self {
        match error.kind {
            runmat_canonical_codec::CanonicalCodecErrorKind::ResourceLimit => Self::ResourceLimit {
                message: error.to_string(),
            },
            runmat_canonical_codec::CanonicalCodecErrorKind::Invalid => error.to_string().into(),
        }
    }
}
impl From<serde_json::Error> for AnalysisPublicationError {
    fn from(error: serde_json::Error) -> Self {
        error.to_string().into()
    }
}
impl From<std::io::Error> for AnalysisPublicationError {
    fn from(error: std::io::Error) -> Self {
        Self::from_io(error)
    }
}

impl From<super::super::AssessmentStoreError> for AnalysisPublicationError {
    fn from(error: super::super::AssessmentStoreError) -> Self {
        match error {
            super::super::AssessmentStoreError::ResourceLimit => Self::ResourceLimit {
                message: error.to_string(),
            },
            super::super::AssessmentStoreError::Invalid(message) => Self::Rejected { message },
        }
    }
}
impl From<AnalysisPublicationError> for super::super::AssessmentStoreError {
    fn from(error: AnalysisPublicationError) -> Self {
        match error {
            AnalysisPublicationError::ResourceLimit { .. } => Self::ResourceLimit,
            other => Self::Invalid(other.to_string()),
        }
    }
}
