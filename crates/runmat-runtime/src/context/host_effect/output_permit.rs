//! Receipt-scoped access to exact output allocation authority. A caller may
//! retain this view, but only the live complete call owns admission.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Weak,
};

use runmat_value::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostAllocationReservation,
    ExactHostNumericAllocationError,
};

pub(super) struct ReceiptOutputPermit {
    authority: Weak<dyn ExactHostAllocationPermit>,
    active_epoch: Arc<AtomicBool>,
    cancellation: runmat_execution::cancellation::CancellationToken,
}

impl ReceiptOutputPermit {
    pub(super) fn new(
        authority: Weak<dyn ExactHostAllocationPermit>,
        active_epoch: Arc<AtomicBool>,
        cancellation: runmat_execution::cancellation::CancellationToken,
    ) -> Self {
        Self {
            authority,
            active_epoch,
            cancellation,
        }
    }
}

impl std::fmt::Debug for ReceiptOutputPermit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReceiptOutputPermit")
            .finish_non_exhaustive()
    }
}

impl ExactHostAllocationPermit for ReceiptOutputPermit {
    fn reserve_exact(
        &self,
        bytes: usize,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError> {
        if !self.active_epoch.load(Ordering::Acquire) || self.cancellation.is_cancelled() {
            return Err(ExactHostNumericAllocationError::ExceedsPermit);
        }
        let authority = self
            .authority
            .upgrade()
            .ok_or(ExactHostNumericAllocationError::ExceedsPermit)?;
        let claim = authority.reserve_exact(bytes)?;
        if !self.active_epoch.load(Ordering::Acquire) || self.cancellation.is_cancelled() {
            return Err(ExactHostNumericAllocationError::ExceedsPermit);
        }
        Ok(claim)
    }

    fn reserve_exact_claim(
        &self,
        bytes: usize,
    ) -> Result<ExactHostAllocationClaim, ExactHostNumericAllocationError> {
        if !self.active_epoch.load(Ordering::Acquire) || self.cancellation.is_cancelled() {
            return Err(ExactHostNumericAllocationError::ExceedsPermit);
        }
        let authority = self
            .authority
            .upgrade()
            .ok_or(ExactHostNumericAllocationError::ExceedsPermit)?;
        let claim = authority.reserve_exact_claim(bytes)?;
        if !self.active_epoch.load(Ordering::Acquire) || self.cancellation.is_cancelled() {
            return Err(ExactHostNumericAllocationError::ExceedsPermit);
        }
        Ok(claim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Claim;

    impl ExactHostAllocationReservation for Claim {}

    #[derive(Debug)]
    struct DirectPermit;

    impl ExactHostAllocationPermit for DirectPermit {
        fn reserve_exact(
            &self,
            _: usize,
        ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError>
        {
            Ok(Arc::new(Claim))
        }
    }

    #[test]
    fn direct_authority_view_cannot_allocate_after_terminal_epoch() {
        let authority: Arc<dyn ExactHostAllocationPermit> = Arc::new(DirectPermit);
        let epoch = Arc::new(AtomicBool::new(true));
        let view = ReceiptOutputPermit::new(
            Arc::downgrade(&authority),
            epoch.clone(),
            runmat_execution::cancellation::CancellationSource::new().token(),
        );
        let retained = view.reserve_exact(4).expect("live direct claim");
        epoch.store(false, Ordering::Release);
        assert!(matches!(
            view.reserve_exact(1),
            Err(ExactHostNumericAllocationError::ExceedsPermit)
        ));
        drop(authority);
        assert!(matches!(
            view.reserve_exact(1),
            Err(ExactHostNumericAllocationError::ExceedsPermit)
        ));
        drop(retained);
    }
}
