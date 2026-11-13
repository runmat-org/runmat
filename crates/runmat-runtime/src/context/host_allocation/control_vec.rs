//! Fixed-capacity control backing whose claim follows its physical allocation.
use std::ops::{Deref, DerefMut};

use runmat_value::{exact_capacity_vec, ExactHostAllocationClaim};

use super::{HostAllocationError, HostControlAllocationOwner};

pub struct HostControlVec<T> {
    // Field order matters: free backing before refunding its claim.
    values: Vec<T>,
    claim: Option<ExactHostAllocationClaim>,
    owner: Option<HostControlAllocationOwner>,
}

impl<T> HostControlVec<T> {
    pub fn with_capacity(
        capacity: usize,
        owner: Option<&HostControlAllocationOwner>,
    ) -> Result<Self, HostAllocationError> {
        let layout =
            std::alloc::Layout::array::<T>(capacity).map_err(|_| HostAllocationError::Overflow)?;
        let claim = if layout.size() == 0 {
            None
        } else {
            owner.map(|owner| owner.claim(layout.size())).transpose()?
        };
        let values = exact_capacity_vec(capacity).map_err(HostAllocationError::Numeric)?;
        Ok(Self {
            values,
            claim,
            owner: owner.cloned(),
        })
    }

    /// Never grows implicitly. A failed insertion returns the original value.
    pub fn try_push(&mut self, value: T) -> Result<(), T> {
        if self.values.len() == self.values.capacity() {
            return Err(value);
        }
        self.values.push(value);
        Ok(())
    }

    /// Admit replacement backing before growth. Old and new allocations are
    /// both live while elements move; the old claim refunds only after free.
    pub fn reserve_capacity(&mut self, capacity: usize) -> Result<(), HostAllocationError> {
        if capacity <= self.values.capacity() {
            return Ok(());
        }
        let mut replacement = Self::with_capacity(capacity, self.owner.as_ref())?;
        for value in self.values.drain(..) {
            replacement.values.push(value);
        }
        let retired = std::mem::replace(self, replacement);
        drop(retired);
        Ok(())
    }

    pub fn capacity(&self) -> usize {
        self.values.capacity()
    }

    pub fn allocation_owner(&self) -> Option<&HostControlAllocationOwner> {
        self.owner.as_ref()
    }
}

impl<T> Deref for HostControlVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.values
    }
}

impl<T> DerefMut for HostControlVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.values
    }
}

pub struct HostControlIntoIter<T> {
    values: std::vec::IntoIter<T>,
    _claim: Option<ExactHostAllocationClaim>,
}

impl<T> Iterator for HostControlIntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.values.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.values.size_hint()
    }
}

impl<T> ExactSizeIterator for HostControlIntoIter<T> {}

impl<T> IntoIterator for HostControlVec<T> {
    type Item = T;
    type IntoIter = HostControlIntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        HostControlIntoIter {
            values: self.values.into_iter(),
            _claim: self.claim,
        }
    }
}
