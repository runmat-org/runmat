//! One known-layout exact-output claim shared by dynamic and pooled managed
//! headroom. The refund policy remains with the headroom owner; this module
//! owns only the physical claim-control allocation and its release order.

use runmat_value::{
    ExactHostAllocationClaim, ExactHostAllocationReservation, ExactHostNumericAllocationError,
};
use std::alloc::{alloc, Layout};
use std::fmt;
use std::sync::Arc;

pub(super) trait OwnedOutputClaimRefund: fmt::Debug + Send + Sync {
    fn refund_exact_output_claim(&self, charged: u64);
}

pub(in crate::context::host_effect::managed_ledger) struct OwnedOutputClaim {
    owner: Arc<dyn OwnedOutputClaimRefund>,
    charged: u64,
}

impl OwnedOutputClaim {
    pub(super) const fn metadata_bytes() -> usize {
        std::mem::size_of::<Self>()
    }

    pub(super) fn allocate(
        owner: Arc<dyn OwnedOutputClaimRefund>,
        charged: u64,
    ) -> Result<ExactHostAllocationClaim, ExactHostNumericAllocationError> {
        let allocate = |layout| {
            // SAFETY: the exact layout is released by the owned claim.
            unsafe { alloc(layout) }
        };
        // SAFETY: the global allocator returns either null or a unique block
        // with exactly the requested layout and alignment.
        let claim = unsafe { Self::try_box_with(owner, charged, allocate)? };
        Ok(ExactHostAllocationClaim::owned(claim))
    }

    /// # Safety
    /// A non-null allocator result must be a fresh, uniquely owned global
    /// allocation with exactly the supplied layout and alignment.
    pub(in crate::context::host_effect::managed_ledger) unsafe fn try_box_with(
        owner: Arc<dyn OwnedOutputClaimRefund>,
        charged: u64,
        allocate: impl FnOnce(Layout) -> *mut u8,
    ) -> Result<Box<Self>, ExactHostNumericAllocationError> {
        let claim = Self { owner, charged };
        let layout = Layout::new::<Self>();
        let pointer = allocate(layout).cast::<Self>();
        if pointer.is_null() {
            // The caller has already committed this charge. Dropping the
            // stack claim runs the same domain refund as a retired header;
            // callers must not issue a second rollback.
            drop(claim);
            return Err(ExactHostNumericAllocationError::AllocationFailed);
        }
        // SAFETY: the fresh exact-layout allocation is initialized once and
        // transferred immediately to its sole Box owner.
        unsafe {
            pointer.write(claim);
            Ok(Box::from_raw(pointer))
        }
    }

    /// Release the claim header before refunding the bytes which funded it.
    ///
    /// # Safety
    /// `deallocate` must release the supplied exact global-allocator block
    /// once and must not unwind after ownership leaves its Box.
    pub(in crate::context::host_effect::managed_ledger) unsafe fn retire_owned_with(
        claim: Box<Self>,
        deallocate: impl FnOnce(std::ptr::NonNull<u8>, Layout),
    ) {
        let pointer = Box::into_raw(claim);
        // SAFETY: consuming the sole Box makes the owner field uniquely owned.
        let owner = unsafe { std::ptr::read(std::ptr::addr_of!((*pointer).owner)) };
        // SAFETY: the allocation remains live until `deallocate` below.
        let charged = unsafe { (*pointer).charged };
        deallocate(
            unsafe { std::ptr::NonNull::new_unchecked(pointer.cast::<u8>()) },
            Layout::new::<Self>(),
        );
        owner.refund_exact_output_claim(charged);
    }
}

impl fmt::Debug for OwnedOutputClaim {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OwnedOutputClaim")
            .field("charged", &self.charged)
            .finish_non_exhaustive()
    }
}

impl ExactHostAllocationReservation for OwnedOutputClaim {
    fn retire_owned(self: Box<Self>) {
        // SAFETY: the unique Box came from the global allocator with the
        // exact OwnedOutputClaim layout and is retired exactly once.
        unsafe {
            Self::retire_owned_with(self, |pointer, layout| {
                std::alloc::dealloc(pointer.as_ptr(), layout)
            })
        }
    }
}

impl Drop for OwnedOutputClaim {
    fn drop(&mut self) {
        self.owner.refund_exact_output_claim(self.charged);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[derive(Debug)]
    struct RefundProbe(AtomicU64);

    impl OwnedOutputClaimRefund for RefundProbe {
        fn refund_exact_output_claim(&self, charged: u64) {
            self.0.fetch_sub(charged, Ordering::SeqCst);
        }
    }

    #[test]
    fn header_is_deallocated_before_refund() {
        let owner = Arc::new(RefundProbe(AtomicU64::new(17)));
        // SAFETY: the global allocator returns the exact requested block.
        let claim =
            unsafe { OwnedOutputClaim::try_box_with(owner.clone(), 17, |layout| alloc(layout)) }
                .expect("owned output claim");
        // SAFETY: the test Box owns one exact allocation; the callback
        // releases it once with its original layout.
        unsafe {
            OwnedOutputClaim::retire_owned_with(claim, |pointer, layout| {
                assert_eq!(owner.0.load(Ordering::SeqCst), 17);
                std::alloc::dealloc(pointer.as_ptr(), layout);
            })
        };
        assert_eq!(owner.0.load(Ordering::SeqCst), 0);
    }
}
