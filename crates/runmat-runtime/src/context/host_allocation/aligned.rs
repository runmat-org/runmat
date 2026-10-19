//! Exactly claimed aligned bytes for caller-owned numerical workspaces.
//!
//! Faer's stack may contain uninitialized bytes. Its backing is therefore
//! allocated with the requested `Layout` and never exposed as initialized
//! integer storage or another typed slice.

use std::alloc::{alloc, dealloc, Layout};
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::Arc;

use runmat_value::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostNumericAllocationError,
};

use super::receipt::AllocationFence;
use super::temporary::{scratch_claim_bytes, ScratchPermit};
use super::{HostAllocationError, HostAllocationReceipt};
use crate::context::host_effect::ManagedHostEffectStageReservation;
use crate::context::RuntimeContext;

pub(crate) struct AlignedHostWorkspace {
    pointer: Option<NonNull<u8>>,
    layout: Layout,
    // Drop frees the exact physical block before retiring this claim.
    _claim: Option<ExactHostAllocationClaim>,
}

impl AlignedHostWorkspace {
    pub(crate) fn quoted_bytes(layout: Layout) -> Result<u64, HostAllocationError> {
        if layout.size() == 0 {
            return Ok(0);
        }
        let total = layout
            .size()
            .checked_add(scratch_claim_bytes())
            .ok_or(HostAllocationError::Overflow)?;
        u64::try_from(total).map_err(|_| HostAllocationError::Overflow)
    }

    pub(crate) fn new(
        layout: Layout,
        receipt: &HostAllocationReceipt,
        context: &RuntimeContext,
    ) -> Result<Self, HostAllocationError> {
        receipt.temporary_aligned(context, layout)
    }

    pub(super) fn new_claimed(
        layout: Layout,
        stage: Arc<ManagedHostEffectStageReservation>,
        fence: AllocationFence,
    ) -> Result<Self, HostAllocationError> {
        if layout.size() == 0 {
            return Ok(Self {
                pointer: None,
                layout,
                _claim: None,
            });
        }
        let permit = ScratchPermit::new(stage, fence);
        let claim = permit
            .reserve_exact_claim(layout.size())
            .map_err(HostAllocationError::Numeric)?;
        // SAFETY: `layout` is checked by the numerical kernel's StackReq.
        // The returned block is uniquely owned until this owner drops.
        let pointer = NonNull::new(unsafe { alloc(layout) }).ok_or(
            HostAllocationError::Numeric(ExactHostNumericAllocationError::AllocationFailed),
        )?;
        Ok(Self {
            pointer: Some(pointer),
            layout,
            _claim: Some(claim),
        })
    }

    pub(crate) fn as_mut_uninit(&mut self) -> &mut [MaybeUninit<u8>] {
        let Some(pointer) = self.pointer else {
            // A zero-length MemStack still receives a non-null pointer with
            // the original Layout alignment. No allocation or dereference
            // occurs for this empty view.
            let aligned = std::ptr::without_provenance_mut::<MaybeUninit<u8>>(self.layout.align());
            // SAFETY: the zero-length slice touches no memory; its pointer is
            // non-null and aligned at least as strongly as the requested Layout.
            return unsafe { std::slice::from_raw_parts_mut(aligned, 0) };
        };
        // SAFETY: this unique owner retains the exact physical block; the
        // numerical kernel receives only uninitialized byte storage.
        unsafe { std::slice::from_raw_parts_mut(pointer.as_ptr().cast(), self.layout.size()) }
    }
}

impl Drop for AlignedHostWorkspace {
    fn drop(&mut self) {
        if let Some(pointer) = self.pointer.take() {
            // SAFETY: this is the original global-allocator block and Layout.
            unsafe { dealloc(pointer.as_ptr(), self.layout) };
        }
        // Field destruction now retires `_claim`, after physical deallocation.
    }
}

// SAFETY: the pointer is uniquely owned, never exposed as a typed shared
// reference, and is freed only by this owner after any blocking worker ends.
unsafe impl Send for AlignedHostWorkspace {}
