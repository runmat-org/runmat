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

/// The sole output-list backing. Ordinary unbounded construction adopts the
/// producer's Vec without allocating another header; exactly admitted
/// storage shares its immutable physical owner in O(1).
#[derive(Debug)]
enum OutputListStorageInner {
    Empty,
    OwnedUnbounded(Vec<Value>),
    SharedExact {
        payload: ExactShared<OutputListPayload>,
        len: usize,
    },
}

#[derive(Debug)]
pub struct OutputListStorage(OutputListStorageInner);

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
        Self(OutputListStorageInner::Empty)
    }

    /// Adopt an ordinary owned vector under an explicitly unbounded policy.
    /// Existing child allocations move into the owner without cloning.
    pub fn unbounded(values: Vec<Value>) -> Self {
        if values.is_empty() {
            return Self::empty();
        }
        Self(OutputListStorageInner::OwnedUnbounded(values))
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
        let values = exact_capacity_vec(count).map_err(Allocation)?;
        let mut payload = ExactShared::try_new(OutputListPayload {
            values,
            claim: Some(claim),
        })
        .map_err(|_| Allocation(ExactHostNumericAllocationError::AllocationFailed))?;
        // Allocate the header as well as the element backing before asking
        // the producer to move any rooted source value into this owner.
        let values = &mut payload.get_mut_unique().values;
        for index in 0..count {
            values.push(initialize(index).map_err(Element)?);
        }
        Ok(Self(OutputListStorageInner::SharedExact {
            payload,
            len: count,
        }))
    }

    pub fn as_slice(&self) -> &[Value] {
        match &self.0 {
            OutputListStorageInner::Empty => &[],
            OutputListStorageInner::OwnedUnbounded(values) => values,
            OutputListStorageInner::SharedExact { payload, len } => &payload.get().values[..*len],
        }
    }

    pub fn has_retained_claim(&self) -> bool {
        matches!(&self.0, OutputListStorageInner::SharedExact { payload, .. } if payload.get().claim.is_some())
    }

    /// Mutate the existing physical backing only when no shared alias can
    /// observe it. This preserves an exact vector/header claim across a
    /// same-cardinality transformation without creating another owner.
    pub fn try_as_mut_slice_unique(&mut self) -> Option<&mut [Value]> {
        match &mut self.0 {
            OutputListStorageInner::Empty => Some(&mut []),
            OutputListStorageInner::OwnedUnbounded(values) => Some(values),
            OutputListStorageInner::SharedExact { payload, len } => {
                if !payload.is_unique() {
                    return None;
                }
                Some(&mut payload.get_mut_unique().values[..*len])
            }
        }
    }

    #[cfg(test)]
    fn physical_capacity(&self) -> usize {
        match &self.0 {
            OutputListStorageInner::Empty => 0,
            OutputListStorageInner::OwnedUnbounded(values) => values.capacity(),
            OutputListStorageInner::SharedExact { payload, .. } => payload.get().values.capacity(),
        }
    }

    /// Extract unique storage with its claim intact. A shared collection is
    /// returned unchanged rather than allocating an unadmitted copy.
    pub fn try_into_owned(self) -> Result<OwnedOutputListValues, Self> {
        match self.0 {
            OutputListStorageInner::Empty => Ok(OwnedOutputListValues {
                values: Vec::new(),
                claim: None,
            }),
            OutputListStorageInner::OwnedUnbounded(values) => Ok(OwnedOutputListValues {
                values,
                claim: None,
            }),
            OutputListStorageInner::SharedExact { payload, len } => match payload.try_unwrap() {
                Ok(OutputListPayload { mut values, claim }) => {
                    values.truncate(len);
                    Ok(OwnedOutputListValues { values, claim })
                }
                Err(payload) => Err(Self(OutputListStorageInner::SharedExact { payload, len })),
            },
        }
    }

    pub fn truncate(&mut self, len: usize) {
        match &mut self.0 {
            OutputListStorageInner::Empty => {}
            OutputListStorageInner::OwnedUnbounded(values) => values.truncate(len),
            OutputListStorageInner::SharedExact {
                len: visible_len, ..
            } => *visible_len = (*visible_len).min(len),
        }
    }

    pub fn try_into_first(self) -> Result<Option<Value>, Self> {
        self.try_into_owned()
            .map(|values| values.into_iter().next())
    }
}

impl Clone for OutputListStorage {
    fn clone(&self) -> Self {
        match &self.0 {
            OutputListStorageInner::Empty => Self::empty(),
            OutputListStorageInner::OwnedUnbounded(values) => Self::unbounded(values.clone()),
            OutputListStorageInner::SharedExact { payload, len } => {
                Self(OutputListStorageInner::SharedExact {
                    payload: payload.clone(),
                    len: *len,
                })
            }
        }
    }
}

impl Default for OutputListStorage {
    fn default() -> Self {
        Self::empty()
    }
}

impl From<Vec<Value>> for OutputListStorage {
    fn from(values: Vec<Value>) -> Self {
        Self::unbounded(values)
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
    pub fn as_mut_slice(&mut self) -> &mut [Value] {
        &mut self.values
    }

    /// Legacy mutation is available only to ordinary unbounded storage. A
    /// claimed backing cannot shed its owner or fund a replacement Vec.
    pub fn try_into_unbounded_vec(self) -> Result<Vec<Value>, Self> {
        if self.claim.is_some() {
            return Err(self);
        }
        Ok(self.values)
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
