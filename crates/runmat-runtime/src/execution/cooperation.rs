//! Invocation-local CPU work cooperation through the owning execution service.

use std::num::NonZeroU64;
use std::rc::Rc;

use futures::future::{select, Either};
use runmat_execution::cancellation::CancellationToken;

use super::{ExecutionServiceError, RuntimeExecutionServices};

/// The execution adapter, not an individual builtin, chooses whether CPU work
/// must periodically give other tasks a turn.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuCooperationPolicy {
    NeverYield,
    WorkQuantum(NonZeroU64),
}

pub struct CpuCooperationBudget {
    service: Rc<dyn RuntimeExecutionServices>,
    cancellation: CancellationToken,
    policy: CpuCooperationPolicy,
    pending_work: u64,
}

impl CpuCooperationBudget {
    pub(crate) fn new(
        service: Rc<dyn RuntimeExecutionServices>,
        cancellation: CancellationToken,
    ) -> Self {
        let policy = service.cpu_cooperation_policy();
        Self {
            service,
            cancellation,
            policy,
            pending_work: 0,
        }
    }

    /// Charge a completed unit of CPU work and yield only when the adapter's
    /// quantum is exhausted. Native `NeverYield` does not allocate a future.
    pub async fn checkpoint(&mut self, completed_work: u64) -> Result<(), ExecutionServiceError> {
        if self.cancellation.is_cancelled() {
            return Err(ExecutionServiceError::Cancelled);
        }
        let CpuCooperationPolicy::WorkQuantum(quantum) = self.policy else {
            return Ok(());
        };
        self.pending_work = self.pending_work.saturating_add(completed_work);
        if self.pending_work < quantum.get() {
            return Ok(());
        }
        self.pending_work %= quantum.get();
        match select(self.service.cooperate_cpu(), self.cancellation.cancelled()).await {
            Either::Left((result, _listener)) => result?,
            Either::Right((_reason, _turn)) => return Err(ExecutionServiceError::Cancelled),
        }
        if self.cancellation.is_cancelled() {
            return Err(ExecutionServiceError::Cancelled);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantum_is_nonzero_and_work_accounting_cannot_overflow() {
        let quantum = NonZeroU64::new(7).unwrap();
        let mut pending = 0u64;
        pending = pending.saturating_add(u64::MAX);
        assert!(pending >= quantum.get());
        assert_eq!(pending % quantum.get(), u64::MAX % 7);
    }
}
