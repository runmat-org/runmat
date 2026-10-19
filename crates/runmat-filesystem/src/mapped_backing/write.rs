//! Write effects are not rolled back by a later cancellation or revocation.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FsMappedWriteDisposition {
    /// Validation failed before any bytes were accepted.
    BeforeEffect,
    /// Exactly this many bytes were accepted before a later failure.
    Committed { accepted_bytes: usize },
    /// The provider cannot prove whether any requested bytes committed.
    Indeterminate { attempted_bytes: usize },
}

#[derive(Debug)]
/// A write disposition with its original cause. Providers use the default
/// I/O cause; invocation interruption can retain a platform-local cause.
pub struct FsMappedWriteFailure<E = io::Error> {
    disposition: FsMappedWriteDisposition,
    source: E,
}

impl<E> FsMappedWriteFailure<E> {
    pub fn before_effect(source: E) -> Self {
        Self {
            disposition: FsMappedWriteDisposition::BeforeEffect,
            source,
        }
    }

    pub fn committed(accepted_bytes: usize, source: E) -> Self {
        Self {
            disposition: FsMappedWriteDisposition::Committed { accepted_bytes },
            source,
        }
    }

    pub fn indeterminate(attempted_bytes: usize, source: E) -> Self {
        Self {
            disposition: FsMappedWriteDisposition::Indeterminate { attempted_bytes },
            source,
        }
    }

    pub fn disposition(&self) -> FsMappedWriteDisposition {
        self.disposition
    }

    pub(crate) fn validate_for_request(self, requested_bytes: usize) -> Self {
        match self.disposition {
            FsMappedWriteDisposition::Committed { accepted_bytes }
                if accepted_bytes > requested_bytes =>
            {
                Self::indeterminate(requested_bytes, self.source)
            }
            FsMappedWriteDisposition::Indeterminate { attempted_bytes }
                if attempted_bytes > requested_bytes =>
            {
                Self::indeterminate(requested_bytes, self.source)
            }
            FsMappedWriteDisposition::BeforeEffect
            | FsMappedWriteDisposition::Committed { .. }
            | FsMappedWriteDisposition::Indeterminate { .. } => self,
        }
    }
}

impl FsMappedWriteFailure {
    pub fn io_error(&self) -> &io::Error {
        &self.source
    }
}

impl<E: Display> Display for FsMappedWriteFailure<E> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "mapped write {:?}: {}",
            self.disposition, self.source
        )
    }
}

impl<E: Error + 'static> Error for FsMappedWriteFailure<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FsMappedWriteOutcome {
    pub accepted_bytes: usize,
    pub requested_bytes: usize,
}

impl FsMappedWriteOutcome {
    pub fn is_complete(self) -> bool {
        self.accepted_bytes == self.requested_bytes
    }
}
