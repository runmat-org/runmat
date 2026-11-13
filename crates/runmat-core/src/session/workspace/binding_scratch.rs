//! Exact temporary vector backing and its physical allocation claim.

use runmat_value::ExactHostAllocationClaim;
use std::ops::{Deref, DerefMut};

pub(super) struct ClaimedVec<T> {
    // Fields drop in declaration order: backing before its allocation claim.
    values: Vec<T>,
    _claim: Option<ExactHostAllocationClaim>,
}

impl<T> ClaimedVec<T> {
    pub(super) fn new(values: Vec<T>, claim: Option<ExactHostAllocationClaim>) -> Self {
        Self {
            values,
            _claim: claim,
        }
    }
}

impl<T> Deref for ClaimedVec<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl<T> DerefMut for ClaimedVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}
