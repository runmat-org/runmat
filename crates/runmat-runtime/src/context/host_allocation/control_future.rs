//! Exact allocation ownership for a process supervisor without a RuntimeContext.
use allocator_api2::boxed::Box as FallibleBox;
use runmat_value::{ExactHostAllocationClaim, ExactHostNumericAllocationError};
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use super::{HostAllocationError, HostControlAllocationOwner};

pub struct HostControlFuture<F> {
    // Free the pinned body before refunding its original claim.
    future: Pin<FallibleBox<F>>,
    _claim: Option<ExactHostAllocationClaim>,
}

impl<F: Future> HostControlFuture<F> {
    pub fn new(
        owner: Option<&HostControlAllocationOwner>,
        make_future: impl FnOnce() -> F,
    ) -> Result<Self, HostAllocationError> {
        let bytes = std::mem::size_of::<F>();
        let claim = if bytes == 0 {
            None
        } else {
            owner.map(|owner| owner.claim(bytes)).transpose()?
        };
        let future = FallibleBox::try_new(make_future()).map_err(|_| {
            HostAllocationError::Numeric(ExactHostNumericAllocationError::AllocationFailed)
        })?;
        Ok(Self {
            future: FallibleBox::into_pin(future),
            _claim: claim,
        })
    }
}

impl<F: Future> Future for HostControlFuture<F> {
    type Output = F::Output;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        self.get_mut().future.as_mut().poll(context)
    }
}
