//! Zero-copy transfer of a claimed native sparse array into a Value buffer.
use super::{allocation_failed, ClaimedVec, SparseWorkError, Storage};
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPlan, HostAllocationReceipt},
    RuntimeContext,
};
use runmat_execution::resource::RetainedHostAllocationClaim;
use runmat_value::{AdoptedHostAllocation, HostAllocationProvenance, HostAllocationRelease};
use std::{
    alloc::{alloc, dealloc, Layout},
    ptr::NonNull,
    sync::Arc,
};

struct Release<T: Copy> {
    values: ClaimedVec<T>,
    claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
}
impl<T: Copy> std::fmt::Debug for Release<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClaimedSparseFactorRelease")
            .finish_non_exhaustive()
    }
}
impl<T: Copy + Send + Sync + 'static> HostAllocationRelease for Release<T> {
    unsafe fn release(&self, _: NonNull<u8>) {
        unreachable!("factor transfers have a unique owned release authority")
    }
    unsafe fn release_owned(self: Box<Self>, _: NonNull<u8>) {
        let pointer = Box::into_raw(self);
        // SAFETY: this consumes the sole release authority. Move each field
        // exactly once; the Box header retires before either claim refunds.
        let values = unsafe { std::ptr::read(std::ptr::addr_of!((*pointer).values)) };
        let claim = unsafe { std::ptr::read(std::ptr::addr_of!((*pointer).claim)) };
        unsafe { dealloc(pointer.cast::<u8>(), Layout::new::<Self>()) };
        // ClaimedVec's original allocator deallocates its actual block before
        // refunding that block's original allocation claim. It is not a Vec
        // reconstructed from a foreign pointer or a substituted allocator.
        drop(values);
        drop(claim);
    }
}

impl<T: Copy + Send + Sync + 'static> ClaimedVec<T> {
    pub(in crate::builtins::math::linalg::ops::sparse_factor) fn into_adopted(
        self,
        shared_header_bytes: usize,
        runtime: &RuntimeContext,
    ) -> Result<AdoptedHostAllocation, SparseWorkError> {
        let len = self.len();
        let capacity = self.values.as_ref().map_or(0, Storage::capacity);
        let layout = Layout::array::<T>(capacity).map_err(|_| HostAllocationError::Overflow)?;
        let byte_length = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(HostAllocationError::Overflow)?;
        let metadata = Layout::new::<Release<T>>()
            .size()
            .checked_add(shared_header_bytes)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(HostAllocationError::Overflow)?;
        let receipt = HostAllocationReceipt::admit(
            Some(runtime),
            HostAllocationPlan::new(0, HostAllocationPlan::retained_metadata_bytes(metadata)?)?,
        )?;
        let claim = receipt
            .as_ref()
            .map(|r| r.claim_retained_metadata(runtime, metadata))
            .transpose()?;
        if let Some(r) = receipt.as_ref() {
            r.checkpoint(runtime)?;
        }
        let data = NonNull::new(self.as_slice().as_ptr().cast_mut().cast::<u8>())
            .expect("native sparse slice has a nonnull aligned pointer, including empty storage");
        // SAFETY: exact-layout fallible allocation, charged before attempt.
        let owner = unsafe { alloc(Layout::new::<Release<T>>()) }.cast::<Release<T>>();
        if owner.is_null() {
            return Err(allocation_failed());
        }
        // No fallible path or await exists between Box creation and transfer
        // to AdoptedHostAllocation's owned-release path.
        let release = unsafe {
            owner.write(Release {
                values: self,
                claim,
            });
            Box::from_raw(owner)
        };
        let allocation = unsafe {
            AdoptedHostAllocation::new_owned(
                data,
                HostAllocationProvenance {
                    allocator: "runmat-sparse-claimed".into(),
                    byte_length,
                    capacity_bytes: layout.size(),
                    alignment: layout.align(),
                },
                release,
            )
        };
        if let Some(r) = receipt.as_ref() {
            r.checkpoint(runtime)?;
        }
        Ok(allocation)
    }
}
