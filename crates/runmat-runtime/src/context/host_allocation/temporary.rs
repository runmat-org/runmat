use std::alloc::{alloc, Layout};
use std::fmt;
use std::sync::Arc;

use runmat_value::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostAllocationReservation,
    ExactHostNumericAllocation, ExactHostNumericAllocationError, NumericDType,
};

use super::receipt::AllocationFence;
use super::HostAllocationError;
use crate::context::host_effect::{
    HostEffectLiveAllocationClaim, HostEffectLiveAllocationKind, HostEffectStageReservation,
    ManagedHostEffectStageReservation,
};

pub(super) fn scratch_claim_bytes() -> usize {
    std::mem::size_of::<ScratchClaim>()
        + ManagedHostEffectStageReservation::exact_temporary_claim_metadata_bytes()
}

struct ScratchClaim {
    _live: HostEffectLiveAllocationClaim,
}

impl fmt::Debug for ScratchClaim {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ScratchClaim")
            .finish_non_exhaustive()
    }
}

impl ExactHostAllocationReservation for ScratchClaim {}

struct ScratchPermit {
    stage: Arc<ManagedHostEffectStageReservation>,
    fence: AllocationFence,
}

impl fmt::Debug for ScratchPermit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ScratchPermit")
            .finish_non_exhaustive()
    }
}

impl ExactHostAllocationPermit for ScratchPermit {
    fn reserve_exact(
        &self,
        _bytes: usize,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError> {
        // This private permit is consumed by ExactHostNumericAllocation, whose
        // owned-claim path avoids a second shared allocation owner.
        Err(ExactHostNumericAllocationError::ExceedsPermit)
    }

    fn reserve_exact_claim(
        &self,
        bytes: usize,
    ) -> Result<ExactHostAllocationClaim, ExactHostNumericAllocationError> {
        self.fence
            .check()
            .map_err(|_| ExactHostNumericAllocationError::ExceedsPermit)?;
        let charged = bytes
            .checked_add(std::mem::size_of::<ScratchClaim>())
            .ok_or(ExactHostNumericAllocationError::ExceedsPermit)?;
        let live = self
            .stage
            .reserve_live_exact_claim(HostEffectLiveAllocationKind::TemporaryResult, charged)
            .map_err(|_| ExactHostNumericAllocationError::ExceedsPermit)?;
        self.fence
            .check()
            .map_err(|_| ExactHostNumericAllocationError::ExceedsPermit)?;
        let layout = Layout::new::<ScratchClaim>();
        // SAFETY: this exact-layout block is initialized once and transferred
        // immediately to Box; on failure the live claim is dropped/refunded.
        let pointer = unsafe { alloc(layout) }.cast::<ScratchClaim>();
        if pointer.is_null() {
            return Err(ExactHostNumericAllocationError::AllocationFailed);
        }
        unsafe {
            pointer.write(ScratchClaim { _live: live });
            Ok(ExactHostAllocationClaim::Owned(Box::from_raw(pointer)))
        }
    }
}

/// Aligned numeric workspace whose physical bytes remain in the stage's
/// scratch envelope until the workspace itself drops. No output conversion is
/// exposed, so retained Values must use the receipt's output permit instead.
pub struct HostNumericWorkspace {
    allocation: ExactHostNumericAllocation,
}

impl HostNumericWorkspace {
    pub(super) fn new(
        dtype: NumericDType,
        len: usize,
        stage: Arc<ManagedHostEffectStageReservation>,
        fence: AllocationFence,
    ) -> Result<Self, HostAllocationError> {
        let permit = ScratchPermit { stage, fence };
        let allocation = ExactHostNumericAllocation::new(dtype, len, &permit)?;
        Ok(Self { allocation })
    }

    pub fn as_f64_slice_mut(&mut self) -> Option<&mut [f64]> {
        self.allocation.as_f64_slice_mut()
    }

    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        self.allocation.as_f64_slice()
    }

    pub fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        self.allocation.as_f32_slice_mut()
    }
}
