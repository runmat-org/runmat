//! Mutable poll-local projections with an exactly claimed managed owner.

use super::host_allocation::HostOutputPermit;
use super::host_effect::ManagedHostEffectStageReservation;
use runmat_execution::resource::ExactShared;
use runmat_value::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostNumericAllocationError,
};
use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PollProjectionQuote {
    pub(crate) scratch_bytes: u64,
    pub(crate) retained_bytes: u64,
    pub(crate) retained_claims: usize,
}

impl PollProjectionQuote {
    pub(crate) fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            scratch_bytes: self.scratch_bytes.checked_add(other.scratch_bytes)?,
            retained_bytes: self.retained_bytes.checked_add(other.retained_bytes)?,
            retained_claims: self.retained_claims.checked_add(other.retained_claims)?,
        })
    }
}

pub(crate) struct AdmittedProjection<T> {
    value: RefCell<T>,
    _claim: Option<ExactHostAllocationClaim>,
    // Variable-size backing owned by `T` is quoted as scratch by the domain
    // preparer and remains charged for the complete projection lifetime.
    _scratch: Arc<ManagedHostEffectStageReservation>,
}

pub(crate) enum PollProjection<T> {
    Legacy(Rc<RefCell<T>>),
    Admitted(ExactShared<AdmittedProjection<T>>),
}

impl<T> Clone for PollProjection<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Legacy(owner) => Self::Legacy(Rc::clone(owner)),
            Self::Admitted(owner) => Self::Admitted(owner.clone()),
        }
    }
}

impl<T> PollProjection<T> {
    pub(crate) fn requested_bytes() -> usize {
        ExactShared::<AdmittedProjection<T>>::requested_bytes()
    }

    pub(crate) fn quote(
        scratch_bytes: u64,
    ) -> Result<PollProjectionQuote, std::num::TryFromIntError> {
        Ok(PollProjectionQuote {
            scratch_bytes,
            retained_bytes: u64::try_from(Self::requested_bytes())?,
            retained_claims: 1,
        })
    }

    pub(crate) fn legacy(value: T) -> Self {
        Self::Legacy(Rc::new(RefCell::new(value)))
    }

    pub(crate) fn prepare(
        value: T,
        permit: &HostOutputPermit,
        scratch: Arc<ManagedHostEffectStageReservation>,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let claim = permit.reserve_exact_claim(Self::requested_bytes())?;
        ExactShared::try_new(AdmittedProjection {
            value: RefCell::new(value),
            _claim: Some(claim),
            _scratch: scratch,
        })
        .map(Self::Admitted)
        .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)
    }

    pub(crate) fn borrow(&self) -> Ref<'_, T> {
        match self {
            Self::Legacy(owner) => owner.borrow(),
            Self::Admitted(owner) => owner.get().value.borrow(),
        }
    }

    pub(crate) fn borrow_mut(&self) -> RefMut<'_, T> {
        match self {
            Self::Legacy(owner) => owner.borrow_mut(),
            Self::Admitted(owner) => owner.get().value.borrow_mut(),
        }
    }
}
