use std::fmt;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};

use runmat_execution::cancellation::CancellationToken;
use runmat_value::{
    ExactHostAllocationPermit, ExactHostAllocationReservation, ExactHostNumericAllocationError,
    NumericDType,
};

use super::{HostAllocationPlan, HostNumericWorkspace};
use crate::context::host_effect::{
    HostEffectAllocationError, HostEffectOutputHeadroom, ManagedHostEffectLedger,
    ManagedHostEffectStageReservation,
};
use crate::context::RuntimeContext;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum HostAllocationError {
    Overflow,
    MissingManagedOwner,
    StaleAuthority,
    Cancelled,
    Resource(HostEffectAllocationError),
    Numeric(ExactHostNumericAllocationError),
}

impl fmt::Display for HostAllocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for HostAllocationError {}

impl From<HostEffectAllocationError> for HostAllocationError {
    fn from(error: HostEffectAllocationError) -> Self {
        Self::Resource(error)
    }
}

#[derive(Clone)]
pub(super) struct AllocationFence {
    active: Arc<AtomicBool>,
    ledger: ManagedHostEffectLedger,
    epoch: u64,
    cancellation: CancellationToken,
    generation_owner: Arc<AtomicU64>,
    generation: u64,
}

impl AllocationFence {
    pub(super) fn check(&self) -> Result<(), HostAllocationError> {
        if self.cancellation.is_cancelled() {
            return Err(HostAllocationError::Cancelled);
        }
        let snapshot = self.ledger.snapshot();
        if !self.active.load(Ordering::Acquire)
            || self.generation_owner.load(Ordering::Acquire) != self.generation
            || !snapshot.current
            || snapshot.epoch != self.epoch
        {
            return Err(HostAllocationError::StaleAuthority);
        }
        Ok(())
    }
}

/// One ordinary host call's actual managed scratch and output precharges.
/// Exact returned-value claims outlive this receipt until their storage drops.
pub struct HostAllocationReceipt {
    runtime: RuntimeContext,
    ledger: ManagedHostEffectLedger,
    stage: Arc<ManagedHostEffectStageReservation>,
    output: GuardedOutputPermit,
    fence: AllocationFence,
}

impl HostAllocationReceipt {
    /// `None` denotes the existing unbounded policy. A finite parallel lease
    /// without the same managed owner fails closed before any allocation.
    pub fn admit(
        runtime: Option<&RuntimeContext>,
        plan: HostAllocationPlan,
    ) -> Result<Option<Self>, HostAllocationError> {
        let Some(runtime) = runtime else {
            return Ok(None);
        };
        if runtime.is_cancelled() {
            return Err(HostAllocationError::Cancelled);
        }
        let Some(parallel) = runtime.service_ports().parallel() else {
            return Ok(None);
        };
        let resources = parallel.placement_resources();
        if resources.memory_available_bytes.is_none() {
            return Ok(None);
        }
        let ledger = parallel
            .managed_host_effect_ledger()
            .ok_or(HostAllocationError::MissingManagedOwner)?;
        let snapshot = ledger.snapshot();
        if !snapshot.current || snapshot.epoch != resources.epoch {
            return Err(HostAllocationError::StaleAuthority);
        }
        let stage = ledger.reserve_stage(plan.scratch_peak_bytes())?;
        let headroom = stage.reserve_output_headroom(plan.retained_output_bytes())?;
        let (generation_owner, generation) = runtime.host_effect_session_generation();
        let fence = AllocationFence {
            active: Arc::new(AtomicBool::new(true)),
            ledger: ledger.clone(),
            epoch: resources.epoch,
            cancellation: runtime.cancellation(),
            generation_owner,
            generation,
        };
        let receipt = Self {
            runtime: runtime.clone(),
            ledger,
            stage,
            output: GuardedOutputPermit {
                headroom,
                fence: fence.clone(),
            },
            fence,
        };
        receipt.checkpoint(runtime)?;
        Ok(Some(receipt))
    }

    pub fn checkpoint(&self, runtime: &RuntimeContext) -> Result<(), HostAllocationError> {
        self.fence.check()?;
        if self.runtime.local_identity() != runtime.local_identity()
            || !self
                .runtime
                .cancellation()
                .same_generation(&runtime.cancellation())
        {
            return Err(HostAllocationError::StaleAuthority);
        }
        let parallel = runtime
            .service_ports()
            .parallel()
            .ok_or(HostAllocationError::StaleAuthority)?;
        let owner = parallel
            .managed_host_effect_ledger()
            .ok_or(HostAllocationError::StaleAuthority)?;
        if !owner.same_owner(&self.ledger)
            || parallel.placement_resources().epoch != self.fence.epoch
        {
            return Err(HostAllocationError::StaleAuthority);
        }
        self.fence.check()
    }

    /// Allocates one aligned, zero-initialized workspace inside the precharged
    /// scratch peak. It cannot be published as an output by this API.
    pub fn temporary_numeric(
        &self,
        runtime: &RuntimeContext,
        dtype: NumericDType,
        len: usize,
    ) -> Result<HostNumericWorkspace, HostAllocationError> {
        self.checkpoint(runtime)?;
        let result = HostNumericWorkspace::new(dtype, len, self.stage.clone(), self.fence.clone());
        // The exact allocator has a deliberately narrow physical-error type.
        // A concurrent cancellation or authority rebind takes precedence over
        // its ExceedsPermit result; a successful allocation is dropped if the
        // second checkpoint fails.
        self.checkpoint(runtime)?;
        result
    }

    /// Construct and publish one exact retained output using only this
    /// receipt's precharged headroom. The constructed result is dropped if a
    /// concurrent cancellation or rebind wins the post-allocation checkpoint;
    /// a returned Tensor retains its exact claim after the receipt retires.
    pub fn with_output_allocation<T>(
        &self,
        runtime: &RuntimeContext,
        construct: impl FnOnce(
            &dyn ExactHostAllocationPermit,
        ) -> Result<T, ExactHostNumericAllocationError>,
    ) -> Result<T, HostAllocationError> {
        self.checkpoint(runtime)?;
        let result = construct(&self.output);
        self.checkpoint(runtime)?;
        result.map_err(HostAllocationError::Numeric)
    }
}

impl Drop for HostAllocationReceipt {
    fn drop(&mut self) {
        self.fence.active.store(false, Ordering::Release);
        self.output.headroom.retire_invocation();
    }
}

struct GuardedOutputPermit {
    headroom: Arc<dyn HostEffectOutputHeadroom>,
    fence: AllocationFence,
}

impl fmt::Debug for GuardedOutputPermit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GuardedOutputPermit")
            .finish_non_exhaustive()
    }
}

impl ExactHostAllocationPermit for GuardedOutputPermit {
    fn reserve_exact(
        &self,
        bytes: usize,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError> {
        self.fence
            .check()
            .map_err(|_| ExactHostNumericAllocationError::ExceedsPermit)?;
        let claim = self.headroom.reserve_exact(bytes)?;
        self.fence
            .check()
            .map_err(|_| ExactHostNumericAllocationError::ExceedsPermit)?;
        Ok(claim)
    }
}
