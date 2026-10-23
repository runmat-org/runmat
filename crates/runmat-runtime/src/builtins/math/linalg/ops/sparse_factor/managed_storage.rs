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
        match self {
            Self::Managed(receipt) => receipt.checkpoint(runtime),
            Self::Unbounded => match HostAllocationReceipt::policy(Some(runtime))? {
                HostAllocationPolicy::Unbounded => Ok(()),
                HostAllocationPolicy::Managed => Err(HostAllocationError::StaleAuthority),
            },
        }
    }
}

pub(super) struct Work<'a> {
    pub(super) runtime: &'a RuntimeContext,
    pub(super) fence: Fence<'a>,
    pub(super) cooperation: CpuCooperationBudget,
    pending: u64,
    quantum: u64,
    remaining: u64,
    #[cfg(test)]
    pub(super) total: u64,
}
impl<'a> Work<'a> {
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
            runtime,
            fence,
            cooperation: runtime.cpu_cooperation(),
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
        self.fence.checkpoint(self.runtime)?;
        self.cooperation.checkpoint(self.pending).await?;
        self.remaining -= self.pending;
        if self.remaining == 0 {
            self.remaining = self.quantum;
        }
        self.pending = 0;
        self.fence.checkpoint(self.runtime)?;
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
        result.cooperation = budget;
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
