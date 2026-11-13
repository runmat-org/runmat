//! Bounded work with the captured numerical operation's allocation issuer.
use super::CooperativeOperatorError;
use crate::context::{
    host_allocation::{
        HostAllocationCheckpoint, HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
    },
    RuntimeContext,
};
use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};

pub(in crate::builtins::math::linalg) struct OperatorWork<'a> {
    runtime: &'a RuntimeContext,
    owner: Option<HostAllocationCheckpoint>,
    budget: CpuCooperationBudget,
    quantum: u64,
    remaining: u64,
    pending: u64,
}
impl<'a> OperatorWork<'a> {
    pub(in crate::builtins::math::linalg) fn new(
        runtime: &'a RuntimeContext,
    ) -> Result<Self, CooperativeOperatorError> {
        let quantum = match runtime.execution().cpu_cooperation_policy() {
            CpuCooperationPolicy::NeverYield => 256,
            CpuCooperationPolicy::WorkQuantum(quantum) => quantum.get(),
        };
        Self::build(runtime, runtime.cpu_cooperation(), quantum)
    }
    fn build(
        runtime: &'a RuntimeContext,
        budget: CpuCooperationBudget,
        quantum: u64,
    ) -> Result<Self, CooperativeOperatorError> {
        Ok(Self {
            runtime,
            owner: HostAllocationReceipt::capture_owner(Some(runtime))?,
            budget,
            quantum,
            remaining: quantum,
            pending: 0,
        })
    }
    #[cfg(test)]
    pub(in crate::builtins::math::linalg) fn with_test_budget(
        runtime: &'a RuntimeContext,
        budget: CpuCooperationBudget,
        quantum: std::num::NonZeroU64,
    ) -> Result<Self, CooperativeOperatorError> {
        Self::build(runtime, budget, quantum.get())
    }
    fn check(&self) -> Result<(), CooperativeOperatorError> {
        check_owner(self.runtime, self.owner.as_ref())
    }
    /// Never process past the next real execution quantum or authority fence.
    pub(in crate::builtins::math::linalg) fn capacity(&self) -> usize {
        (self.remaining.min(256) - self.pending) as usize
    }
    pub(in crate::builtins::math::linalg) async fn charge(
        &mut self,
        count: usize,
    ) -> Result<(), CooperativeOperatorError> {
        assert!(
            count <= self.capacity(),
            "caller must bound work before performing it"
        );
        self.pending += count as u64;
        if self.pending == self.remaining.min(256) {
            self.flush().await?;
        }
        Ok(())
    }
    pub(in crate::builtins::math::linalg) async fn tick(
        &mut self,
    ) -> Result<(), CooperativeOperatorError> {
        self.charge(1).await
    }
    pub(in crate::builtins::math::linalg) async fn flush(
        &mut self,
    ) -> Result<(), CooperativeOperatorError> {
        self.check()?;
        self.budget
            .checkpoint(self.pending)
            .await
            .map_err(CooperativeOperatorError::Execution)?;
        self.check()?;
        self.remaining -= self.pending;
        if self.remaining == 0 {
            self.remaining = self.quantum;
        }
        self.pending = 0;
        Ok(())
    }
}

/// The same captured issuer/generation check for owned and borrowed work.
pub(super) fn check_owner(
    runtime: &RuntimeContext,
    owner: Option<&HostAllocationCheckpoint>,
) -> Result<(), CooperativeOperatorError> {
    if let Some(owner) = owner {
        owner.check(runtime)?;
    } else if HostAllocationReceipt::policy(Some(runtime))? != HostAllocationPolicy::Unbounded {
        return Err(HostAllocationError::StaleAuthority.into());
    }
    Ok(())
}
