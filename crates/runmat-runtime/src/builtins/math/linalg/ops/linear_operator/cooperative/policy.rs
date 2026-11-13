//! Cooperative products use either the existing operator work owner or the
//! caller's original budget. No allocation or new authority joins the latter.
use super::{CooperativeOperatorError, OperatorWork};
use crate::context::{
    host_allocation::{HostAllocationCheckpoint, HostAllocationError, HostAllocationReceipt},
    RuntimeContext,
};
use crate::execution::CpuCooperationBudget;
use std::future::{ready, Future};

pub(in crate::builtins::math::linalg) trait ProductWork {
    fn capacity(&self) -> usize;
    fn charge(
        &mut self,
        completed: usize,
    ) -> impl Future<Output = Result<(), CooperativeOperatorError>>;
    fn tick(&mut self) -> impl Future<Output = Result<(), CooperativeOperatorError>>;
    fn flush(&mut self) -> impl Future<Output = Result<(), CooperativeOperatorError>>;
}

impl ProductWork for OperatorWork<'_> {
    fn capacity(&self) -> usize {
        OperatorWork::capacity(self)
    }
    fn charge(
        &mut self,
        completed: usize,
    ) -> impl Future<Output = Result<(), CooperativeOperatorError>> {
        OperatorWork::charge(self, completed)
    }
    fn tick(&mut self) -> impl Future<Output = Result<(), CooperativeOperatorError>> {
        OperatorWork::tick(self)
    }
    fn flush(&mut self) -> impl Future<Output = Result<(), CooperativeOperatorError>> {
        OperatorWork::flush(self)
    }
}

pub(in crate::builtins::math::linalg) struct BorrowedWork<'a> {
    runtime: &'a RuntimeContext,
    budget: &'a mut CpuCooperationBudget,
    owner: Option<HostAllocationCheckpoint>,
    pending: u64,
}
impl<'a> BorrowedWork<'a> {
    pub(in crate::builtins::math::linalg) fn new(
        runtime: &'a RuntimeContext,
        budget: &'a mut CpuCooperationBudget,
    ) -> Result<Self, CooperativeOperatorError> {
        Self::check_token(runtime, budget)?;
        let work = Self {
            runtime,
            budget,
            owner: HostAllocationReceipt::capture_owner(Some(runtime))?,
            pending: 0,
        };
        work.check()?;
        Ok(work)
    }
    fn check_token(
        runtime: &RuntimeContext,
        budget: &CpuCooperationBudget,
    ) -> Result<(), CooperativeOperatorError> {
        let token = budget.cancellation_token();
        if !token.same_generation(&runtime.cancellation()) {
            return Err(HostAllocationError::StaleAuthority.into());
        }
        if token.is_cancelled() {
            return Err(CooperativeOperatorError::Execution(
                crate::execution::ExecutionServiceError::Cancelled,
            ));
        }
        Ok(())
    }
    fn check(&self) -> Result<(), CooperativeOperatorError> {
        Self::check_token(self.runtime, self.budget)?;
        super::work::check_owner(self.runtime, self.owner.as_ref())
    }
    fn limit(&self) -> u64 {
        self.budget
            .remaining_until_turn()
            .map_or(256, |remaining| remaining.get().min(256))
    }
    fn capacity(&self) -> usize {
        (self.limit() - self.pending) as usize
    }
    async fn charge(&mut self, completed: usize) -> Result<(), CooperativeOperatorError> {
        assert!(
            completed <= self.capacity(),
            "caller must bound work before performing it"
        );
        self.pending += completed as u64;
        if self.pending == self.limit() {
            self.flush().await?;
        }
        Ok(())
    }
    async fn flush(&mut self) -> Result<(), CooperativeOperatorError> {
        self.check()?;
        self.budget
            .checkpoint(self.pending)
            .await
            .map_err(CooperativeOperatorError::Execution)?;
        self.check()?;
        self.pending = 0;
        Ok(())
    }
}
impl ProductWork for BorrowedWork<'_> {
    fn capacity(&self) -> usize {
        BorrowedWork::capacity(self)
    }
    fn charge(
        &mut self,
        completed: usize,
    ) -> impl Future<Output = Result<(), CooperativeOperatorError>> {
        BorrowedWork::charge(self, completed)
    }
    fn tick(&mut self) -> impl Future<Output = Result<(), CooperativeOperatorError>> {
        BorrowedWork::charge(self, 1)
    }
    fn flush(&mut self) -> impl Future<Output = Result<(), CooperativeOperatorError>> {
        BorrowedWork::flush(self)
    }
}

pub(in crate::builtins::math::linalg) struct UnmanagedWork;
impl ProductWork for UnmanagedWork {
    fn capacity(&self) -> usize {
        256
    }
    fn charge(
        &mut self,
        _completed: usize,
    ) -> impl Future<Output = Result<(), CooperativeOperatorError>> {
        ready(Ok(()))
    }
    fn tick(&mut self) -> impl Future<Output = Result<(), CooperativeOperatorError>> {
        ready(Ok(()))
    }
    fn flush(&mut self) -> impl Future<Output = Result<(), CooperativeOperatorError>> {
        ready(Ok(()))
    }
}
