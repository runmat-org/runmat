//! Original-attempt raw output, separate from child-certified console receipts.
//! The process owner drains its pipes; callers read bounded committed pages.
use std::sync::Arc;

mod budget;
mod journal;
mod drain;
mod retained;
pub(crate) use retained::RawOutputReadHandle;

pub(crate) use budget::RawOutputBudget;
pub(crate) use drain::{drain_pipes, OutputDrainControl, OutputDrainLifetime};
pub use journal::RawOutputState;
pub(crate) use journal::{RawOutputIdentity, RawOutputJournal, RawOutputReader};

pub(crate) const OUTPUT_PAGE_BYTES: usize = 8192;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RawOutputStream {
    Stdout,
    Stderr,
}

impl RawOutputStream {
    fn index(self) -> usize {
        match self {
            Self::Stdout => 0,
            Self::Stderr => 1,
        }
    }
    fn file_name(self) -> &'static str {
        match self {
            Self::Stdout => "stdout.bytes",
            Self::Stderr => "stderr.bytes",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RawOutputError {
    #[error("raw output host allocation failed: {0}")]
    HostAllocation(#[from] runmat_runtime::context::host_allocation::HostAllocationError),
    #[error("invalid raw output journal: {0}")]
    Invalid(&'static str),
    #[error(
        "raw output exceeds its original logical storage grant ({required} > {maximum} bytes)"
    )]
    StorageLimit { required: u64, maximum: u64 },
    #[error("raw output storage failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("raw output metadata publication failed: {0}")]
    Publication(#[source] Box<crate::NativeExecutionError>),
    #[error("{primary}; output cleanup also failed: {cleanup}")]
    Cleanup {
        #[source]
        primary: Arc<RawOutputError>,
        cleanup: Box<RawOutputError>,
    },
    #[error("{primary}; another application stream also failed: {other}")]
    Streams {
        #[source]
        primary: Arc<RawOutputError>,
        other: Arc<RawOutputError>,
    },
}

type OutputResult<T> = Result<T, RawOutputError>;

impl RawOutputError {
    pub(crate) fn primary_allocation_failure(
        &self,
    ) -> Option<runmat_runtime::context::host_allocation::HostAllocationError> {
        match self {
            Self::HostAllocation(cause) => Some(*cause),
            Self::Cleanup { primary, .. } | Self::Streams { primary, .. } => {
                primary.primary_allocation_failure()
            }
            Self::Invalid(_) | Self::StorageLimit { .. } | Self::Io(_) | Self::Publication(_) => {
                None
            }
        }
    }
}

impl From<crate::NativeExecutionError> for RawOutputError {
    fn from(error: crate::NativeExecutionError) -> Self {
        Self::Publication(Box::new(error))
    }
}
