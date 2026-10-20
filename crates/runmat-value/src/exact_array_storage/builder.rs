//! Staged construction of the canonical exact shared-array owner.

use runmat_execution::resource::ExactShared;

use crate::{exact_capacity_vec, ExactHostAllocationPermit, ExactHostNumericAllocationError};

use super::{claim_cell, ExactArrayPayload, ExactArrayStorage, ExactArrayStorageInner};

/// A unique, fully reserved final backing whose initialized prefix is safe to
/// inspect while a producer cooperatively constructs its remaining elements.
/// Dropping an unfinished builder retires initialized elements and backing
/// before refunding its retained claim.
#[derive(Debug)]
pub struct ExactArrayStorageBuilder<T> {
    payload: Option<ExactShared<ExactArrayPayload<T>>>,
    expected: usize,
}

impl<T> ExactArrayStorageBuilder<T> {
    pub(super) fn try_new(
        count: usize,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        if count == 0 {
            return Ok(Self {
                payload: None,
                expected: 0,
            });
        }
        let requested = ExactArrayStorage::<T>::requested_bytes(count)?;
        let claim = permit.reserve_exact_claim(requested)?;
        let values = exact_capacity_vec(count)?;
        let payload = ExactShared::try_new(ExactArrayPayload {
            values,
            claim: claim_cell(Some(claim)),
        })
        .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)?;
        Ok(Self {
            payload: Some(payload),
            expected: count,
        })
    }

    pub fn expected_len(&self) -> usize {
        self.expected
    }

    pub fn initialized(&self) -> &[T] {
        self.payload
            .as_ref()
            .map_or(&[], |payload| &payload.get().values)
    }

    /// Edit only initialized elements in this unique, unpublished backing.
    pub fn initialized_mut(&mut self) -> &mut [T] {
        self.payload
            .as_mut()
            .map_or(&mut [], |payload| &mut payload.get_mut_unique().values)
    }

    /// Remove an initialized element without reallocating the final backing.
    pub fn remove_initialized(&mut self, index: usize) -> Option<T> {
        let payload = self.payload.as_mut()?;
        let values = &mut payload.get_mut_unique().values;
        (index < values.len()).then(|| values.remove(index))
    }

    /// Publish the current prefix, retaining the original capacity claim.
    pub fn finish_initialized(mut self) -> ExactArrayStorage<T> {
        let len = self.initialized().len();
        match self.payload.take() {
            Some(payload) => {
                ExactArrayStorage(ExactArrayStorageInner::SharedExact { payload, len })
            }
            None => ExactArrayStorage(ExactArrayStorageInner::EmptyExact),
        }
    }

    /// Move one already-owned element into its final slot. An overfull push
    /// returns that same element without modifying the admitted owner.
    pub fn try_push(&mut self, value: T) -> Result<(), T> {
        let Some(payload) = self.payload.as_mut() else {
            return Err(value);
        };
        let values = &mut payload.get_mut_unique().values;
        if values.len() == self.expected {
            return Err(value);
        }
        values.push(value);
        Ok(())
    }

    /// Publish only a complete exact collection. On early finish, ownership
    /// and the initialized prefix remain with the returned builder.
    pub fn try_finish(mut self) -> Result<ExactArrayStorage<T>, Self> {
        if self.initialized().len() != self.expected {
            return Err(self);
        }
        let Some(payload) = self.payload.take() else {
            return Ok(ExactArrayStorage(ExactArrayStorageInner::EmptyExact));
        };
        Ok(ExactArrayStorage(ExactArrayStorageInner::SharedExact {
            payload,
            len: self.expected,
        }))
    }
}
