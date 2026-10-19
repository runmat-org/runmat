//! Retained console text shared by entries and host transports.
//!
//! Managed clones share one admitted UTF-8 backing. Ordinary text keeps its
//! existing String ownership without introducing a reference-count allocation.

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostText, HostTextAllocation, HostTextError,
};
use runmat_execution::resource::ExactShared;
use runmat_value::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostNumericAllocationError,
};
use std::fmt;
use std::ops::Deref;

pub struct ConsoleText {
    owner: TextOwner,
}

enum TextOwner {
    Unbounded(String),
    Managed(ExactShared<ManagedText>),
}

struct ManagedText {
    text: HostText,
    _header_claim: ExactHostAllocationClaim,
}

pub(crate) enum PreparedConsoleText {
    Empty {
        failure: Option<HostTextError>,
    },
    Managed {
        writer: HostTextAllocation,
        header_claim: ExactHostAllocationClaim,
    },
}

impl ConsoleText {
    pub fn retained_bytes(capacity: usize) -> Result<u64, HostAllocationError> {
        if capacity == 0 {
            return Ok(0);
        }
        let text = u64::try_from(capacity).map_err(|_| HostAllocationError::Overflow)?;
        let header = u64::try_from(ExactShared::<ManagedText>::requested_bytes())
            .map_err(|_| HostAllocationError::Overflow)?;
        HostAllocationPlan::retained_exact_output_bytes(
            text.checked_add(header)
                .ok_or(HostAllocationError::Overflow)?,
            HostTextAllocation::claim_count(capacity) + 1,
        )
    }

    /// Initialize directly in pre-admitted text storage. Formatting cannot
    /// grow the allocation, and an error never publishes a partial message.
    pub fn try_initialize(
        capacity: usize,
        permit: &dyn ExactHostAllocationPermit,
        write: impl FnOnce(&mut dyn fmt::Write) -> fmt::Result,
    ) -> Result<Self, HostTextError> {
        let mut prepared = Self::prepare(capacity, permit)?;
        if write(&mut prepared).is_err() {
            prepared.fail_formatting();
        }
        prepared.finish()
    }

    /// Charge an existing unclaimed input's entire backing, not just its
    /// visible text. Already-admitted text needs no second allocation claim.
    pub(crate) fn adoption_retained_bytes(&self) -> Result<u64, HostAllocationError> {
        match &self.owner {
            TextOwner::Unbounded(text) => Self::retained_bytes(text.capacity()),
            TextOwner::Managed(_) => Ok(0),
        }
    }

    pub(crate) fn try_adopt(
        self,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, HostTextError> {
        match self.owner {
            TextOwner::Unbounded(text) => Self::try_adopt_owned(text, permit),
            owner @ TextOwner::Managed(_) => Ok(Self { owner }),
        }
    }

    pub(crate) fn try_adopt_owned(
        text: String,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, HostTextError> {
        Self::adopt_owned_with_suffix(text, "", permit)
    }

    /// The caller uses the exact writer instead when the input has no spare
    /// byte. This path never grows an existing input allocation.
    pub(crate) fn try_adopt_owned_line(
        text: String,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, HostTextError> {
        let suffix = if text.ends_with('\n') { "" } else { "\n" };
        Self::adopt_owned_with_suffix(text, suffix, permit)
    }

    fn adopt_owned_with_suffix(
        text: String,
        suffix: &str,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, HostTextError> {
        if text.capacity() == 0 && suffix.is_empty() {
            return Ok(Self::from(text));
        }
        let header_claim = permit
            .reserve_exact_claim(ExactShared::<ManagedText>::requested_bytes())
            .map_err(HostTextError::Allocation)?;
        let text = HostText::adopt_owned(text, suffix, permit)?;
        Self::share_managed(text, header_claim)
    }

    fn share_managed(
        text: HostText,
        header_claim: ExactHostAllocationClaim,
    ) -> Result<Self, HostTextError> {
        let owner = ExactShared::try_new(ManagedText {
            text,
            _header_claim: header_claim,
        })
        .map_err(|_| {
            HostTextError::Allocation(ExactHostNumericAllocationError::AllocationFailed)
        })?;
        Ok(Self {
            owner: TextOwner::Managed(owner),
        })
    }

    /// The caller admits both the UTF-8 backing and its known-layout shared
    /// owner before constructing this writer, then fences entry publication.
    pub(crate) fn prepare(
        capacity: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<PreparedConsoleText, HostTextError> {
        if capacity == 0 {
            return Ok(PreparedConsoleText::Empty { failure: None });
        }
        let header_claim = permit
            .reserve_exact_claim(ExactShared::<ManagedText>::requested_bytes())
            .map_err(HostTextError::Allocation)?;
        let writer = HostTextAllocation::try_new(capacity, permit)?;
        Ok(PreparedConsoleText::Managed {
            writer,
            header_claim,
        })
    }

    pub fn as_str(&self) -> &str {
        match &self.owner {
            TextOwner::Unbounded(text) => text,
            TextOwner::Managed(owner) => owner.get().text.as_str(),
        }
    }

    /// Conversion to a bare String cannot discard a live managed claim.
    pub fn try_into_unbounded(self) -> Result<String, Self> {
        match self.owner {
            TextOwner::Unbounded(text) => Ok(text),
            owner @ TextOwner::Managed(_) => Err(Self { owner }),
        }
    }

    pub fn has_managed_backing(&self) -> bool {
        matches!(&self.owner, TextOwner::Managed(_))
    }
}

impl PreparedConsoleText {
    pub(crate) fn finish(self) -> Result<ConsoleText, HostTextError> {
        match self {
            Self::Empty {
                failure: Some(error),
            } => Err(error),
            Self::Empty { failure: None } => Ok(ConsoleText::from(String::new())),
            Self::Managed {
                writer,
                header_claim,
            } => ConsoleText::share_managed(writer.finish()?, header_claim),
        }
    }
    fn fail_formatting(&mut self) {
        match self {
            Self::Empty { failure } if failure.is_none() => {
                *failure = Some(HostTextError::FormattingFailed);
            }
            Self::Empty { .. } => {}
            Self::Managed { writer, .. } => writer.fail_formatting(),
        }
    }
}

impl fmt::Write for PreparedConsoleText {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        match self {
            Self::Empty { failure } => {
                if failure.is_some() {
                    return Err(fmt::Error);
                }
                if !text.is_empty() {
                    *failure = Some(HostTextError::CapacityExceeded);
                    Err(fmt::Error)
                } else {
                    Ok(())
                }
            }
            Self::Managed { writer, .. } => fmt::Write::write_str(writer, text),
        }
    }

    fn write_fmt(&mut self, arguments: fmt::Arguments<'_>) -> fmt::Result {
        match self {
            Self::Managed { writer, .. } => fmt::Write::write_fmt(writer, arguments),
            Self::Empty { failure: Some(_) } => Err(fmt::Error),
            Self::Empty { failure: None } => {
                let result = fmt::write(self, arguments);
                if result.is_err() {
                    self.fail_formatting();
                }
                result
            }
        }
    }
}

impl Clone for ConsoleText {
    fn clone(&self) -> Self {
        Self {
            owner: match &self.owner {
                TextOwner::Unbounded(text) => TextOwner::Unbounded(text.clone()),
                TextOwner::Managed(owner) => TextOwner::Managed(owner.clone()),
            },
        }
    }
}

impl From<String> for ConsoleText {
    fn from(text: String) -> Self {
        Self {
            owner: TextOwner::Unbounded(text),
        }
    }
}

impl From<&str> for ConsoleText {
    fn from(text: &str) -> Self {
        Self::from(text.to_owned())
    }
}

impl PartialEq for ConsoleText {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for ConsoleText {}

impl PartialEq<&str> for ConsoleText {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<String> for ConsoleText {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<ConsoleText> for String {
    fn eq(&self, other: &ConsoleText) -> bool {
        self.as_str() == other.as_str()
    }
}

impl PartialEq<ConsoleText> for &str {
    fn eq(&self, other: &ConsoleText) -> bool {
        *self == other.as_str()
    }
}

impl serde::Serialize for ConsoleText {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl Deref for ConsoleText {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for ConsoleText {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Debug for ConsoleText {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), output)
    }
}

impl fmt::Display for ConsoleText {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), output)
    }
}
