//! Sparse factor storage and cooperative work under one original issuer.
mod growth;
use crate::context::{
    host_allocation::{
        HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
    },
    RuntimeContext,
};
use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
pub(super) use growth::ClaimedVec;

#[derive(Debug)]
pub(super) enum SparseWorkError {
    Admission(HostAllocationError),
    Execution(ExecutionServiceError),
}
impl From<HostAllocationError> for SparseWorkError {
    fn from(error: HostAllocationError) -> Self {
        Self::Admission(error)
    }
}
impl From<ExecutionServiceError> for SparseWorkError {
    fn from(error: ExecutionServiceError) -> Self {
        Self::Execution(error)
    }
}

pub(super) enum Fence<'a> {
    Managed(&'a HostAllocationReceipt),
    Unbounded,
}
impl Fence<'_> {
    pub(super) fn checkpoint(&self, runtime: &RuntimeContext) -> Result<(), HostAllocationError> {
        self.checkpoint_optional(Some(runtime))
    }
    pub(super) fn checkpoint_optional(
        &self,
        runtime: Option<&RuntimeContext>,
    ) -> Result<(), HostAllocationError> {
        match self {
            Self::Managed(receipt) => {
                receipt.checkpoint(runtime.ok_or(HostAllocationError::StaleAuthority)?)
            }
            Self::Unbounded => match HostAllocationReceipt::policy(runtime)? {
                HostAllocationPolicy::Unbounded => Ok(()),
                HostAllocationPolicy::Managed => Err(HostAllocationError::StaleAuthority),
            },
        }
    }
}

pub(super) struct Work<'a> {
    pub(super) runtime: Option<&'a RuntimeContext>,
    pub(super) fence: Fence<'a>,
    pub(super) cooperation: Option<CpuCooperationBudget>,
    pending: u64,
    quantum: u64,
    remaining: u64,
    #[cfg(test)]
    pub(super) total: u64,
}
impl<'a> Work<'a> {
    /// Explicitly absent context has no issuer/port to invent. It is only an
    /// unbounded synchronous sparse numerical authority; Some preserves the
    /// original policy, token, cooperation budget and managed receipts.
    pub(super) fn for_optional_policy(
        runtime: Option<&'a RuntimeContext>,
        fence: Option<&'a HostAllocationReceipt>,
    ) -> Result<Self, SparseWorkError> {
        if let Some(runtime) = runtime {
            return Self::for_policy(runtime, fence);
        }
        let fence = fence.map_or(Fence::Unbounded, Fence::Managed);
        fence.checkpoint_optional(None)?;
        Ok(Self {
            runtime: None,
            fence,
            cooperation: None,
            pending: 0,
            quantum: 256,
            remaining: 256,
            #[cfg(test)]
            total: 0,
        })
    }
    pub(super) fn required_runtime(&self) -> Result<&'a RuntimeContext, HostAllocationError> {
        self.runtime.ok_or(HostAllocationError::StaleAuthority)
    }
    pub(super) fn required_budget(
        &mut self,
    ) -> Result<&mut CpuCooperationBudget, HostAllocationError> {
        self.cooperation
            .as_mut()
            .ok_or(HostAllocationError::StaleAuthority)
    }
    #[cfg(test)]
    pub(super) fn new(runtime: &'a RuntimeContext, fence: &'a HostAllocationReceipt) -> Self {
        Self::with_fence(runtime, Fence::Managed(fence))
    }
    pub(super) fn for_policy(
        runtime: &'a RuntimeContext,
        fence: Option<&'a HostAllocationReceipt>,
    ) -> Result<Self, SparseWorkError> {
        let result = Self::with_fence(runtime, fence.map_or(Fence::Unbounded, Fence::Managed));
        result.fence.checkpoint(runtime)?;
        Ok(result)
    }
    fn with_fence(runtime: &'a RuntimeContext, fence: Fence<'a>) -> Self {
        let quantum = match runtime.execution().cpu_cooperation_policy() {
            CpuCooperationPolicy::NeverYield => 256,
            CpuCooperationPolicy::WorkQuantum(quantum) => quantum.get(),
        };
        Self {
            runtime: Some(runtime),
            fence,
            cooperation: Some(runtime.cpu_cooperation()),
            pending: 0,
            quantum,
            remaining: quantum,
            #[cfg(test)]
            total: 0,
        }
    }
    pub(super) async fn tick(&mut self) -> Result<(), SparseWorkError> {
        self.charge(1).await
    }
    pub(super) fn capacity(&self) -> usize {
        (self.remaining.min(256) - self.pending) as usize
    }
    pub(super) async fn charge(&mut self, count: usize) -> Result<(), SparseWorkError> {
        assert!(
            count <= self.capacity(),
            "bound sparse work before processing it"
        );
        self.pending += count as u64;
        #[cfg(test)]
        {
            self.total += count as u64;
        }
        if self.pending == self.remaining.min(256) {
            self.flush().await?;
        }
        Ok(())
    }
    pub(super) async fn flush(&mut self) -> Result<(), SparseWorkError> {
        self.fence.checkpoint_optional(self.runtime)?;
        if let Some(cooperation) = self.cooperation.as_mut() {
            cooperation.checkpoint(self.pending).await?;
        }
        self.remaining -= self.pending;
        if self.remaining == 0 {
            self.remaining = self.quantum;
        }
        self.pending = 0;
        self.fence.checkpoint_optional(self.runtime)?;
        Ok(())
    }
    #[cfg(test)]
    pub(super) fn with_test_budget(
        runtime: &'a RuntimeContext,
        fence: &'a HostAllocationReceipt,
        budget: CpuCooperationBudget,
        quantum: std::num::NonZeroU64,
    ) -> Self {
        let mut result = Self::new(runtime, fence);
        result.cooperation = Some(budget);
        result.quantum = quantum.get();
        result.remaining = quantum.get();
        result
    }
}

pub(super) fn admit(
    runtime: &RuntimeContext,
    scratch: u64,
    retained: u64,
) -> Result<HostAllocationReceipt, HostAllocationError> {
    HostAllocationReceipt::admit(Some(runtime), HostAllocationPlan::new(scratch, retained)?)?
        .ok_or(HostAllocationError::MissingManagedOwner)
}
