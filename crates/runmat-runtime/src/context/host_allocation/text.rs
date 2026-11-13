//! Exact, fallible UTF-8 backing for retained host text.
//!
//! Text owners retain their exact backing claim through console transport.

use runmat_value::{
    exact_capacity_vec, ExactHostAllocationClaim, ExactHostAllocationPermit,
    ExactHostNumericAllocationError,
};
use std::alloc::Layout;
use std::fmt;

#[derive(Debug)]
pub enum HostTextError {
    Allocation(ExactHostNumericAllocationError),
    CapacityExceeded,
    FormattingFailed,
    InvalidUtf8(std::str::Utf8Error),
}

impl fmt::Display for HostTextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allocation(error) => write!(formatter, "text allocation failed: {error:?}"),
            Self::CapacityExceeded => formatter.write_str("text exceeded its admitted capacity"),
            Self::FormattingFailed => formatter.write_str("text formatting failed"),
            Self::InvalidUtf8(error) => write!(formatter, "invalid UTF-8 text: {error}"),
        }
    }
}

impl std::error::Error for HostTextError {}

/// A string and its retained backing claim are one move-only owner. Sharing
/// belongs to the enclosing entry owner, not a second text allocation.
#[derive(Debug)]
pub(crate) struct HostText {
    text: String,
    // The physical string is destroyed before its claim is retired.
    _claim: Option<ExactHostAllocationClaim>,
}

impl HostText {
    pub(crate) fn as_str(&self) -> &str {
        &self.text
    }

    /// Rehome an existing input backing without copying its bytes. Charge its
    /// full capacity before ownership transfer or any bounded suffix write.
    pub(crate) fn adopt_owned(
        mut text: String,
        suffix: &str,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, HostTextError> {
        let required = text
            .len()
            .checked_add(suffix.len())
            .ok_or(HostTextError::CapacityExceeded)?;
        if required > text.capacity() {
            return Err(HostTextError::CapacityExceeded);
        }
        let physical = HostTextAllocation::requested_bytes(text.capacity())?;
        let claim = if physical == 0 {
            None
        } else {
            Some(
                permit
                    .reserve_exact_claim(physical)
                    .map_err(HostTextError::Allocation)?,
            )
        };
        // Capacity was checked before admission; this cannot reallocate.
        text.push_str(suffix);
        Ok(Self {
            text,
            _claim: claim,
        })
    }
}

/// A bounded writer created only after its exact backing is admitted. It
/// cannot grow, and any failed write prevents publishing a partial message.
pub(crate) struct HostTextAllocation {
    bytes: Vec<u8>,
    failure: Option<HostTextError>,
    claim: Option<ExactHostAllocationClaim>,
}

impl HostTextAllocation {
    /// Physical byte backing only. The operation's plan must additionally
    /// include the canonical claim-control charge for `claim_count` owners.
    pub(crate) fn requested_bytes(capacity: usize) -> Result<usize, HostTextError> {
        Layout::array::<u8>(capacity)
            .map(|layout| layout.size())
            .map_err(|_| HostTextError::Allocation(ExactHostNumericAllocationError::LayoutOverflow))
    }

    pub(crate) fn claim_count(capacity: usize) -> usize {
        usize::from(capacity != 0)
    }

    pub(crate) fn try_new(
        capacity: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, HostTextError> {
        let physical = Self::requested_bytes(capacity)?;
        let claim = if physical == 0 {
            None
        } else {
            Some(
                permit
                    .reserve_exact_claim(physical)
                    .map_err(HostTextError::Allocation)?,
            )
        };
        let bytes = exact_capacity_vec(capacity).map_err(HostTextError::Allocation)?;
        Ok(Self {
            bytes,
            failure: None,
            claim,
        })
    }

    pub(crate) fn finish(self) -> Result<HostText, HostTextError> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        // String adopts the existing exact-capacity Vec without another
        // payload allocation. Only UTF-8 str segments can enter the writer.
        let text = match String::from_utf8(self.bytes) {
            Ok(text) => text,
            Err(error) => return Err(HostTextError::InvalidUtf8(error.utf8_error())),
        };
        Ok(HostText {
            text,
            _claim: self.claim,
        })
    }
    pub(crate) fn fail_formatting(&mut self) {
        if self.failure.is_none() {
            self.failure = Some(HostTextError::FormattingFailed);
        }
    }
}

impl fmt::Write for HostTextAllocation {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        if self.failure.is_some() {
            return Err(fmt::Error);
        }
        if text.len() > self.bytes.capacity() - self.bytes.len() {
            self.failure = Some(HostTextError::CapacityExceeded);
            return Err(fmt::Error);
        }
        self.bytes.extend_from_slice(text.as_bytes());
        Ok(())
    }

    fn write_fmt(&mut self, arguments: fmt::Arguments<'_>) -> fmt::Result {
        if self.failure.is_some() {
            return Err(fmt::Error);
        }
        let result = fmt::write(self, arguments);
        if result.is_err() {
            self.fail_formatting();
        }
        result
    }
}
