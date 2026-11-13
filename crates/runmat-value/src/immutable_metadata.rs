//! Original immutable metadata backing. Claims cover the known-layout shared
//! header and the original allocations inventoried by its domain owner.

use runmat_execution::resource::ExactShared;
use std::sync::OnceLock;

use crate::{ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostNumericAllocationError};

#[derive(Debug)]
struct Payload<T> {
    value: T,
    // Payload allocations retire before their one physical claim refunds.
    claim: OnceLock<ExactHostAllocationClaim>,
}

#[derive(Debug)]
pub(crate) struct ImmutableMetadata<T>(ExactShared<Payload<T>>);

impl<T> Clone for ImmutableMetadata<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> ImmutableMetadata<T> {
    pub(crate) const fn requested_header_bytes() -> usize {
        ExactShared::<Payload<T>>::requested_bytes()
    }

    /// Admit the one known-layout header before allocating it. Child metadata
    /// keeps its own original owners/claims; this certifies no child backing.
    pub(crate) fn try_new_exact(
        value: T,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let claim = permit.reserve_exact_claim(Self::requested_header_bytes())?;
        let claims = OnceLock::new();
        let _ = claims.set(claim);
        ExactShared::try_new(Payload {
            value,
            claim: claims,
        })
        .map(Self)
        .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)
    }

    pub(crate) fn new(value: T) -> Self {
        Self(ExactShared::new(Payload {
            value,
            claim: OnceLock::new(),
        }))
    }
    pub(crate) fn get(&self) -> &T {
        &self.0.get().value
    }
    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        self.0.ptr_eq(&other.0)
    }
    pub(crate) fn has_claim(&self) -> bool {
        self.0.get().claim.get().is_some()
    }
    pub(crate) fn unclaimed_bytes(
        &self,
        payload_bytes: impl FnOnce(&T) -> Result<usize, ExactHostNumericAllocationError>,
    ) -> Result<Option<usize>, ExactHostNumericAllocationError> {
        if self.has_claim() {
            return Ok(None);
        }
        payload_bytes(self.get())?
            .checked_add(ExactShared::<Payload<T>>::requested_bytes())
            .map(Some)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)
    }
    pub(crate) fn promote(
        &self,
        permit: &dyn ExactHostAllocationPermit,
        payload_bytes: impl FnOnce(&T) -> Result<usize, ExactHostNumericAllocationError>,
    ) -> Result<(), ExactHostNumericAllocationError> {
        let Some(bytes) = self.unclaimed_bytes(payload_bytes)? else {
            return Ok(());
        };
        let claim = match permit.reserve_exact_claim(bytes) {
            Ok(claim) => claim,
            Err(_) if self.has_claim() => return Ok(()),
            Err(error) => return Err(error),
        };
        // Concurrent aliases may attach first; the losing claim refunds.
        let _ = self.0.get().claim.set(claim);
        Ok(())
    }
    pub(crate) fn into_inner(self) -> T
    where
        T: Clone,
    {
        // Moving admitted backing out would refund its claim too early. An
        // explicit detach clones the payload instead. Canonical child owners
        // retain their own original physical claims through that O(1) clone.
        if self.has_claim() {
            return self.get().clone();
        }
        match self.0.try_unwrap() {
            Ok(payload) => payload.value,
            Err(shared) => shared.get().value.clone(),
        }
    }
}
