//! Invocation-owned CPU work cooperation for scalar callback solvers.

use runmat_builtins::{BuiltinCatalogIdentity, OptimSolver};

use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::BuiltinResult;

pub(crate) struct ScalarCooperation {
    budget: Option<CpuCooperationBudget>,
    runtime: Option<RuntimeContext>,
    builtin: BuiltinCatalogIdentity,
    allocation_epoch: Option<AllocationEpoch>,
}

enum AllocationEpoch {
    Unbounded,
    Managed(crate::context::host_allocation::HostAllocationCheckpoint),
}

impl ScalarCooperation {
    pub(crate) fn new(solver: OptimSolver, runtime: Option<&RuntimeContext>) -> Self {
        Self::for_builtin(
            BuiltinCatalogIdentity {
                name: solver.name(),
            },
            runtime,
        )
    }

    pub(crate) fn for_builtin(
        builtin: BuiltinCatalogIdentity,
        runtime: Option<&RuntimeContext>,
    ) -> Self {
        let (budget, runtime) = runtime
            .map(|runtime| {
                let budget = runtime.cpu_cooperation();
                // Both views retain the same invocation even if the caller's
                // shared cancellation slot is rebound while a solver awaits.
                let runtime = runtime
                    .clone()
                    .with_cancellation_token(budget.cancellation_token());
                (budget, runtime)
            })
            .unzip();
        Self {
            budget,
            runtime,
            builtin,
            allocation_epoch: None,
        }
    }

    pub(crate) async fn completed(&mut self, units: u64) -> BuiltinResult<()> {
        self.check_allocation_epoch()?;
        if let Some(budget) = self.budget.as_mut() {
            budget
                .checkpoint(units)
                .await
                .map_err(|source| self.execution_error(source))?;
        }
        self.check_allocation_epoch()
    }

    pub(crate) async fn integer_boundary<'a>(
        &mut self,
        value: &'a runmat_value::Value,
    ) -> BuiltinResult<crate::builtins::common::validation::PreparedIntegerBoundary<'a>> {
        use crate::builtins::common::validation::{
            prepare_direct_integer_f64_boundary, IntegerBoundaryFailure,
        };
        self.check_allocation_epoch()?;
        let prepared = prepare_direct_integer_f64_boundary(value, self.budget.as_mut())
            .await
            .map_err(|error| match error {
                IntegerBoundaryFailure::Execution(source) => self.execution_error(source),
                other => other.into_runtime_error(),
            })?;
        self.check_allocation_epoch()?;
        Ok(prepared)
    }

    /// Opt in to a read-only original allocation epoch fence. This reserves no
    /// bytes: each concrete allocator still needs its own canonical receipt.
    pub(crate) fn pin_allocation_epoch(&mut self) -> BuiltinResult<()> {
        use crate::context::host_allocation::HostAllocationReceipt;
        if self.allocation_epoch.is_none() {
            self.allocation_epoch =
                Some(match HostAllocationReceipt::capture_owner(self.runtime()) {
                    Ok(Some(owner)) => AllocationEpoch::Managed(owner),
                    Ok(None) => AllocationEpoch::Unbounded,
                    Err(source) => return Err(self.allocation_error(source)),
                });
        }
        self.check_allocation_epoch()
    }

    /// Call at synchronous allocation/publication boundaries too. Yield
    /// checking alone cannot cover an operation that has not yielded yet.
    pub(crate) fn check_allocation_epoch(&self) -> BuiltinResult<()> {
        use crate::context::host_allocation::{
            HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
        };
        let result = match &self.allocation_epoch {
            None => Ok(()),
            Some(AllocationEpoch::Managed(owner)) => match self.runtime() {
                Some(runtime) => owner.check(runtime).and_then(|()| {
                    match HostAllocationReceipt::policy(Some(runtime))? {
                        HostAllocationPolicy::Managed => Ok(()),
                        HostAllocationPolicy::Unbounded => Err(HostAllocationError::StaleAuthority),
                    }
                }),
                None => Err(HostAllocationError::StaleAuthority),
            },
            Some(AllocationEpoch::Unbounded) => match HostAllocationReceipt::policy(self.runtime())
            {
                Ok(HostAllocationPolicy::Unbounded) => Ok(()),
                Ok(HostAllocationPolicy::Managed) => Err(HostAllocationError::StaleAuthority),
                Err(source) => Err(source),
            },
        };
        result.map_err(|source| self.allocation_error(source))
    }
    fn allocation_error(
        &self,
        source: crate::context::host_allocation::HostAllocationError,
    ) -> crate::RuntimeError {
        let failure = crate::context::host_allocation::host_allocation_runtime_error(
            self.runtime(),
            "RunMat:Optim:AllocationEpoch",
            "original allocation epoch",
            source,
        );
        crate::builtins::common::map_control_flow_with_builtin(failure, self.builtin.name)
    }

    pub(crate) fn runtime(&self) -> Option<&RuntimeContext> {
        self.runtime.as_ref()
    }

    /// Borrow the original scheduling budget for one numerical owner. This
    /// neither mints a second budget nor substitutes an ambient context.
    pub(crate) fn runtime_budget(
        &mut self,
    ) -> (Option<&RuntimeContext>, Option<&mut CpuCooperationBudget>) {
        (self.runtime.as_ref(), self.budget.as_mut())
    }

    fn execution_error(&self, source: ExecutionServiceError) -> crate::RuntimeError {
        let failure = source.into_runtime_error_in_context(self.runtime.as_ref());
        crate::builtins::common::map_control_flow_with_builtin(failure, self.builtin.name)
    }

    #[cfg(test)]
    pub(crate) fn with_budget(solver: OptimSolver, budget: CpuCooperationBudget) -> Self {
        Self::with_builtin_budget(
            BuiltinCatalogIdentity {
                name: solver.name(),
            },
            budget,
        )
    }

    #[cfg(test)]
    pub(crate) fn with_builtin_budget(
        builtin: BuiltinCatalogIdentity,
        budget: CpuCooperationBudget,
    ) -> Self {
        Self {
            budget: Some(budget),
            runtime: None,
            builtin,
            allocation_epoch: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    use std::rc::Rc;

    #[test]
    fn solver_context_and_budget_retain_the_same_original_cancellation_generation() {
        let original = CancellationSource::new();
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(crate::execution::RuntimeExecutionService::new()),
            original.token(),
        );
        let mut cooperation = ScalarCooperation::new(OptimSolver::Patternsearch, Some(&runtime));
        let successor = CancellationSource::new();
        let _replacement = runtime.scoped_cancellation_token(successor.token());
        assert!(!runtime.cancellation().same_generation(&original.token()));
        let (selected, budget) = cooperation.runtime_budget();
        assert!(selected
            .unwrap()
            .cancellation()
            .same_generation(&original.token()));
        assert!(budget
            .unwrap()
            .cancellation_token()
            .same_generation(&original.token()));
        original.cancel(CancellationReason::User);
        assert!(cooperation.runtime().unwrap().is_cancelled());
        assert!(!runtime.is_cancelled());
        let error = futures::executor::block_on(cooperation.completed(0)).unwrap_err();
        assert!(crate::runtime_error::error_has_control_cancellation_origin(
            &error
        ));
    }
}
