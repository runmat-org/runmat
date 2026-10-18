use super::RootAdmissionError;
use std::convert::Infallible;
use std::{error::Error, fmt};

/// Retains the source domain's typed traversal failure separately from GC
/// admission. A failed traversal never publishes a partial root batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RootTraceFailure<E> {
    Source(E),
    Admission(RootAdmissionError),
}

impl<E> From<RootAdmissionError> for RootTraceFailure<E> {
    fn from(error: RootAdmissionError) -> Self {
        Self::Admission(error)
    }
}

impl<E: fmt::Display> fmt::Display for RootTraceFailure<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source(error) => write!(formatter, "root source traversal failed: {error}"),
            Self::Admission(error) => fmt::Display::fmt(error, formatter),
        }
    }
}

impl<E: Error + 'static> Error for RootTraceFailure<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Source(error) => Some(error),
            Self::Admission(error) => Some(error),
        }
    }
}

pub(super) fn infallible_source(error: RootTraceFailure<Infallible>) -> RootAdmissionError {
    match error {
        RootTraceFailure::Source(impossible) => match impossible {},
        RootTraceFailure::Admission(error) => error,
    }
}
