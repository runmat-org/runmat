//! Immutable child views retaining their original collection backing.

use std::fmt;
use std::ptr::NonNull;

use super::OutputListStorage;
use crate::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetainedArrayValueError {
    UnsharedBacking,
    IndexOutOfBounds,
}

impl fmt::Display for RetainedArrayValueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::UnsharedBacking => "retained child views require shared collection backing",
            Self::IndexOutOfBounds => "retained child index is out of bounds",
        })
    }
}

impl std::error::Error for RetainedArrayValueError {}

/// An immutable view into a Value tree owned by shared collection backing.
/// Cloning or projecting the view retains the same physical owner and claim;
/// neither operation copies children or allocates another shared header.
///
/// This is physical ownership, not a GC root. Runtime carriers must keep the
/// selected Value rooted for as long as they use this view, just as they do
/// for an ordinary Value borrowed across execution or collection.
#[derive(Clone)]
pub struct RetainedArrayValue {
    owner: OutputListStorage,
    value: NonNull<Value>,
}

impl fmt::Debug for RetainedArrayValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RetainedArrayValue")
            .finish_non_exhaustive()
    }
}

impl OutputListStorage {
    /// Retain one child without materializing the collection. Ordinary owned
    /// vectors must first be consumed uniquely or explicitly published into
    /// shared backing by their selected allocation policy.
    pub fn retain_value_at(
        &self,
        index: usize,
    ) -> Result<RetainedArrayValue, RetainedArrayValueError> {
        if !self.is_shallow_clone() {
            return Err(RetainedArrayValueError::UnsharedBacking);
        }
        let value = self
            .as_slice()
            .get(index)
            .ok_or(RetainedArrayValueError::IndexOutOfBounds)?;
        Ok(RetainedArrayValue {
            owner: self.clone(),
            value: NonNull::from(value),
        })
    }
}

impl RetainedArrayValue {
    pub fn value(&self) -> &Value {
        // SAFETY: the initial pointer is checked against shared immutable
        // backing before cloning that owner. Further pointers come only from
        // lifetime-bound projections of its transitively owned Value tree.
        // Holding `owner` prevents unique mutable access to that backing.
        unsafe { self.value.as_ref() }
    }

    /// Select an immutable descendant while retaining the original owner.
    /// The higher-ranked selector cannot return a shorter-lived captured
    /// reference in place of a child borrowed from the supplied Value.
    pub fn project(
        &self,
        select: impl for<'a> FnOnce(&'a Value) -> Option<&'a Value>,
    ) -> Option<Self> {
        let child = select(self.value())?;
        Some(Self {
            owner: self.owner.clone(),
            value: NonNull::from(child),
        })
    }
}
