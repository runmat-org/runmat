//! Native operational failures own their causes; diagnostics are display data.
use crate::MexDiagnostic;
use runmat_accelerate_api::NumericUploadError;
use std::{error::Error, fmt};

#[derive(Debug)]
pub enum MexGpuError {
    Message(String),
    Provider(anyhow::Error),
    Upload {
        context: &'static str,
        source: NumericUploadError,
    },
}
impl From<String> for MexGpuError {
    fn from(message: String) -> Self {
        Self::Message(message)
    }
}
impl fmt::Display for MexGpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Message(message) => f.write_str(message),
            Self::Provider(error) => fmt::Display::fmt(error, f),
            Self::Upload { context, source } => write!(f, "{context}: {source}"),
        }
    }
}
impl Error for MexGpuError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Message(_) => None,
            Self::Provider(error) => Some(error.as_ref()),
            Self::Upload { source, .. } => Some(source),
        }
    }
}

/// Invocation-local failure carrier. It is not serializable or equatable:
/// only its diagnostic may cross a diagnostic-only transport boundary.
#[derive(Debug)]
pub struct MexInvocationFailure {
    pub diagnostic: MexDiagnostic,
    causes: Vec<MexGpuError>,
}
impl MexInvocationFailure {
    pub(crate) fn new(diagnostic: MexDiagnostic, causes: Vec<MexGpuError>) -> Self {
        Self { diagnostic, causes }
    }
    /// A native gateway can report more than one failing operation before exit.
    /// Every cause remains owned even though Error::source exposes the first.
    pub fn causes(&self) -> &[MexGpuError] {
        &self.causes
    }
}
impl fmt::Display for MexInvocationFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.diagnostic.message)
    }
}
impl Error for MexInvocationFailure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.causes
            .first()
            .map(|cause| cause as &(dyn Error + 'static))
    }
}

#[cfg(test)]
pub(crate) fn retained_upload_fixture() -> (MexGpuError, std::sync::Weak<()>) {
    #[derive(Debug)]
    struct RetainedCause {
        _retained: std::sync::Arc<()>,
    }
    impl fmt::Display for RetainedCause {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("retained upload failure")
        }
    }
    impl Error for RetainedCause {}
    let retained = std::sync::Arc::new(());
    let weak = std::sync::Arc::downgrade(&retained);
    (
        MexGpuError::Upload {
            context: "test upload",
            source: NumericUploadError::from(anyhow::Error::new(RetainedCause {
                _retained: retained,
            })),
        },
        weak,
    )
}
