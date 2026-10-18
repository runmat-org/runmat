//! Shared ownership of one transient output list's physical backing.
//!
//! Children retain their own allocation and GC-root policies. Admitting this
//! collection does not certify a clone or materialization of its children.

use std::alloc::Layout;
use std::fmt;
use std::ops::Deref;

use runmat_execution::resource::ExactShared;

use crate::{
    exact_capacity_vec, ExactHostAllocationClaim, ExactHostAllocationPermit,
    ExactHostNumericAllocationError, Value,
};

#[derive(Debug)]
struct OutputListPayload {
    values: Vec<Value>,
    // The vector and shared header must retire before their claim refunds.
    claim: Option<ExactHostAllocationClaim>,
}

/// An immutable output collection. Cloning shares its physical allocation;
/// consuming a shared collection never implicitly clones language values.
#[derive(Clone, Debug, Default)]
pub struct OutputListStorage {
    payload: Option<ExactShared<OutputListPayload>>,
}

#[derive(Debug)]
pub enum OutputListInitializationError<E> {
    Allocation(ExactHostNumericAllocationError),
    Element(E),
}

impl<E: fmt::Display> fmt::Display for OutputListInitializationError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allocation(error) => write!(formatter, "output-list allocation failed: {error}"),
            Self::Element(error) => error.fmt(formatter),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for OutputListInitializationError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Allocation(error) => Some(error),
            Self::Element(error) => Some(error),
        }
    }
}

impl OutputListStorage {
    pub const fn empty() -> Self {
        Self { payload: None }
    }

    /// Adopt an ordinary owned vector under an explicitly unbounded policy.
    /// Existing child allocations move into the owner without cloning.
    pub fn unbounded(values: Vec<Value>) -> Self {
        if values.is_empty() {
            return Self::empty();
        }
        Self {
            payload: Some(ExactShared::new(OutputListPayload {
                values,
                claim: None,
            })),
        }
    }

    /// Requested bytes for the exact-capacity vector and its actual shared
    /// control allocation. Child allocations require independent authority.
    pub fn requested_bytes(count: usize) -> Result<usize, ExactHostNumericAllocationError> {
        if count == 0 {
            return Ok(0);
        }
        Layout::array::<Value>(count)
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?
            .size()
            .checked_add(ExactShared::<OutputListPayload>::requested_bytes())
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }

    /// Reserve both collection allocations before constructing any element.
    /// Each element moves into its final slot; the producer is responsible
    /// for independent child allocation/alias and root admission.
    pub fn try_initialize<E>(
        count: usize,
        permit: &dyn ExactHostAllocationPermit,
        mut initialize: impl FnMut(usize) -> Result<Value, E>,
    ) -> Result<Self, OutputListInitializationError<E>> {
        use OutputListInitializationError::{Allocation, Element};

        if count == 0 {
            return Ok(Self::empty());
        }
        let claim = permit
            .reserve_exact_claim(Self::requested_bytes(count).map_err(Allocation)?)
            .map_err(Allocation)?;
        let mut values = exact_capacity_vec(count).map_err(Allocation)?;
        for index in 0..count {
            values.push(initialize(index).map_err(Element)?);
        }
        let payload = ExactShared::try_new(OutputListPayload {
            values,
            claim: Some(claim),
        })
        .map_err(|_| Allocation(ExactHostNumericAllocationError::AllocationFailed))?;
        Ok(Self {
            payload: Some(payload),
        })
    }

    pub fn as_slice(&self) -> &[Value] {
        self.payload
            .as_ref()
            .map_or(&[], |payload| payload.get().values.as_slice())
    }

    pub fn has_retained_claim(&self) -> bool {
        self.payload
            .as_ref()
            .is_some_and(|payload| payload.get().claim.is_some())
    }

    /// Extract unique storage with its claim intact. A shared collection is
    /// returned unchanged rather than allocating an unadmitted copy.
    pub fn try_into_owned(self) -> Result<OwnedOutputListValues, Self> {
        let Some(payload) = self.payload else {
            return Ok(OwnedOutputListValues {
                values: Vec::new(),
                claim: None,
            });
        };
        match payload.try_unwrap() {
            Ok(OutputListPayload { values, claim }) => Ok(OwnedOutputListValues { values, claim }),
            Err(payload) => Err(Self {
                payload: Some(payload),
            }),
        }
    }
}

impl Deref for OutputListStorage {
    type Target = [Value];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl PartialEq for OutputListStorage {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<'a> IntoIterator for &'a OutputListStorage {
    type Item = &'a Value;
    type IntoIter = std::slice::Iter<'a, Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

/// A unique output vector whose physical claim follows its backing through
/// consumption. No bare Vec escape can discard that claim prematurely.
#[derive(Debug)]
pub struct OwnedOutputListValues {
    values: Vec<Value>,
    claim: Option<ExactHostAllocationClaim>,
}

impl OwnedOutputListValues {
    pub fn as_slice(&self) -> &[Value] {
        &self.values
    }
}

impl IntoIterator for OwnedOutputListValues {
    type Item = Value;
    type IntoIter = OutputListIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        OutputListIntoIter {
            values: self.values.into_iter(),
            _claim: self.claim,
        }
    }
}

/// The original vector backing is freed before its retained claim refunds,
/// including early iterator drop and partial consumption.
#[derive(Debug)]
pub struct OutputListIntoIter {
    values: std::vec::IntoIter<Value>,
    _claim: Option<ExactHostAllocationClaim>,
}

impl Iterator for OutputListIntoIter {
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        self.values.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.values.size_hint()
    }
}

impl DoubleEndedIterator for OutputListIntoIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.values.next_back()
    }
}

impl ExactSizeIterator for OutputListIntoIter {}
impl std::iter::FusedIterator for OutputListIntoIter {}

#[cfg(test)]
mod tests;
