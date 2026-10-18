//! One immutable physical Value generation, independent of GC root policy.
//!
//! A GC activation traces the contained Value as an ordinary child. An
//! external workspace or read carrier pairs this owner with its own root
//! lease; storing a root lease in this payload would make self-referential
//! closure activations uncollectible.

use runmat_execution::resource::ExactShared;

use crate::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostNumericAllocationError, Value,
};

#[derive(Debug)]
struct SharedValuePayload {
    value: Value,
    // The Value and shared header retire before this physical claim refunds.
    _claim: Option<ExactHostAllocationClaim>,
}

#[derive(Clone, Debug)]
pub struct SharedValueGeneration(ExactShared<SharedValuePayload>);

impl SharedValueGeneration {
    pub const fn requested_bytes() -> usize {
        ExactShared::<SharedValuePayload>::requested_bytes()
    }

    pub fn prepare(
        value: Value,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let claim = permit.reserve_exact_claim(Self::requested_bytes())?;
        ExactShared::try_new(SharedValuePayload {
            value,
            _claim: Some(claim),
        })
        .map(Self)
        .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)
    }

    pub fn unbounded(value: Value) -> Self {
        Self(ExactShared::new(SharedValuePayload {
            value,
            _claim: None,
        }))
    }

    pub fn value(&self) -> &Value {
        &self.0.get().value
    }

    pub fn has_retained_claim(&self) -> bool {
        self.0.get()._claim.is_some()
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.0.ptr_eq(&other.0)
    }
}
