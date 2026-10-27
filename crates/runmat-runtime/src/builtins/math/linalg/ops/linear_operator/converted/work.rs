//! Cooperative products use either the existing operator work owner or the
//! caller's original budget. No allocation or new authority joins the latter.
use super::*;
use std::future::{ready, Future};

pub(super) trait ProductWork {
    fn tick(&mut self) -> impl Future<Output = Result<(), CooperativeOperatorError>>;
    fn flush(&mut self) -> impl Future<Output = Result<(), CooperativeOperatorError>>;
}

impl ProductWork for OperatorWork<'_> {
    fn tick(&mut self) -> impl Future<Output = Result<(), CooperativeOperatorError>> {
        OperatorWork::tick(self)
    }
    fn flush(&mut self) -> impl Future<Output = Result<(), CooperativeOperatorError>> {
        OperatorWork::flush(self)
    }
}

pub(super) struct BorrowedWork<'a> {
    pub runtime: &'a RuntimeContext,
    pub budget: &'a mut CpuCooperationBudget,
}
impl BorrowedWork<'_> {
    async fn checkpoint(&mut self, completed: u64) -> Result<(), CooperativeOperatorError> {
        self.budget
            .checkpoint(completed)
            .await
            .map_err(CooperativeOperatorError::Execution)?;
        HostAllocationReceipt::policy(Some(self.runtime))?;
        Ok(())
    }
}
impl ProductWork for BorrowedWork<'_> {
    fn tick(&mut self) -> impl Future<Output = Result<(), CooperativeOperatorError>> {
        self.checkpoint(1)
    }
    fn flush(&mut self) -> impl Future<Output = Result<(), CooperativeOperatorError>> {
        self.checkpoint(0)
    }
}

pub(super) struct UnmanagedWork;
impl ProductWork for UnmanagedWork {
    fn tick(&mut self) -> impl Future<Output = Result<(), CooperativeOperatorError>> {
        ready(Ok(()))
    }
    fn flush(&mut self) -> impl Future<Output = Result<(), CooperativeOperatorError>> {
        ready(Ok(()))
    }
}
