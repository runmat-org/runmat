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
pub struct FsMappedWriteFailure {
    disposition: FsMappedWriteDisposition,
    source: io::Error,
}

impl FsMappedWriteFailure {
    pub fn before_effect(source: io::Error) -> Self {
        Self {
            disposition: FsMappedWriteDisposition::BeforeEffect,
            source,
        }
    }

    pub fn committed(accepted_bytes: usize, source: io::Error) -> Self {
        Self {
            disposition: FsMappedWriteDisposition::Committed { accepted_bytes },
            source,
        }
    }

    pub fn indeterminate(attempted_bytes: usize, source: io::Error) -> Self {
        Self {
            disposition: FsMappedWriteDisposition::Indeterminate { attempted_bytes },
            source,
        }
    }

    pub fn disposition(&self) -> FsMappedWriteDisposition {
        self.disposition
    }

    pub fn io_error(&self) -> &io::Error {
        &self.source
    }
}

impl Display for FsMappedWriteFailure {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "mapped write {:?}: {}",
            self.disposition, self.source
        )
    }
}

impl Error for FsMappedWriteFailure {
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
