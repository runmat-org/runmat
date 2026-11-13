//! Runtime authority adapter for the shared, allocation-free PCG owner.
use super::super::iterative::PreconditionerStatus;
use super::kernel::{Native, PcgOutcome, PcgScalar, PcgSolveInput};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::execution::CpuCooperationBudget;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;
pub(super) use runmat_sparse_solvers::krylov::pcg::PcgOperation;
use runmat_sparse_solvers::krylov::{Control, KrylovFailure};

pub(super) trait AsyncPcgOperation<T: PcgScalar> {
    async fn apply(
        &mut self,
        operation: PcgOperation,
        input: &[T],
        output: &mut [T],
        runtime: Option<&RuntimeContext>,
    ) -> Result<PreconditionerStatus, RuntimeError>;
    fn checkpoint(&mut self) -> Result<(), RuntimeError> {
        Ok(())
    }
}
fn admission(runtime: Option<&RuntimeContext>, error: HostAllocationError) -> RuntimeError {
    super::execution::map_operator_in_context(error.into(), runtime)
}
fn terminal(mut error: RuntimeError) -> RuntimeError {
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}
struct Driver<'a, T, O, F> {
    operation: O,
    context: Option<&'a RuntimeContext>,
    fence: F,
    cooperation: Option<CpuCooperationBudget>,
    scalar: std::marker::PhantomData<T>,
}
impl<T: PcgScalar, O: AsyncPcgOperation<T>, F: FnMut() -> Result<(), RuntimeError>> Control
    for Driver<'_, T, O, F>
{
    type Error = RuntimeError;
    fn checkpoint(&mut self) -> Result<(), RuntimeError> {
        (self.fence)()?;
        self.operation.checkpoint()
    }
    async fn cooperate(&mut self, completed: u64) -> Result<(), RuntimeError> {
        self.checkpoint()?;
        if let Some(budget) = self.cooperation.as_mut() {
            budget
                .checkpoint(completed)
                .await
                .map_err(|error| error.into_runtime_error_in_context(self.context))?;
        }
        self.checkpoint()
    }
}
impl<T: PcgScalar, O: AsyncPcgOperation<T>, F: FnMut() -> Result<(), RuntimeError>>
    runmat_sparse_solvers::krylov::pcg::PcgOperations<T> for Driver<'_, T, O, F>
{
    async fn measured<R>(
        &mut self,
        work: runmat_sparse_solvers::krylov::usage::VectorWork,
        run: impl FnOnce(&mut Self) -> (R, runmat_sparse_solvers::krylov::usage::BatchCompletion),
    ) -> Result<R, RuntimeError> {
        use crate::analysis::fea_run_admission::{reserve_work, FeaRunDenied};
        use runmat_simulation_usage::{LedgerError, UsageCategory, UsageContext};
        self.checkpoint()?;
        let Some(usage) = UsageContext::capture() else {
            return Ok(run(self).0);
        };
        let maximum = work.operations().map_err(|cause| {
            FeaRunDenied::from(LedgerError::from(cause)).into_runtime_error("pcg")
        })?;
        let mut reservation = reserve_work(&usage, UsageCategory::Solve, maximum, self.context)
            .await
            .map_err(|denied| denied.into_runtime_error("pcg"))?;
        self.checkpoint()?;
        let (result, completed) = run(self);
        use runmat_sparse_solvers::krylov::usage::BatchCompletion;
        let completed = match completed {
            BatchCompletion::Skipped => None,
            BatchCompletion::Complete => Some(maximum),
            BatchCompletion::Prefix(count) => Some(count),
        };
        if let Some(completed) = completed {
            reservation
                .complete(completed)
                .map_err(|cause| FeaRunDenied::from(cause).into_runtime_error("pcg"))?;
        }
        self.checkpoint()?;
        Ok(result)
    }
    async fn apply(
        &mut self,
        operation: PcgOperation,
        input: &[T],
        output: &mut [T],
    ) -> Result<PreconditionerStatus, RuntimeError> {
        self.operation.checkpoint()?;
        let result = self
            .operation
            .apply(operation, input, output, self.context)
            .await?;
        self.operation.checkpoint()?;
        Ok(result)
    }
}
/// Consumes the selected operation owner; failed effects cannot be replayed.
pub(super) async fn drive<T: PcgScalar, O: AsyncPcgOperation<T>>(
    input: PcgSolveInput<'_, T>,
    operation: O,
    runtime: Option<&RuntimeContext>,
    receipt: Option<&HostAllocationReceipt>,
    preconditioners: (bool, bool),
) -> Result<PcgOutcome, RuntimeError> {
    let initial_policy =
        HostAllocationReceipt::policy(runtime).map_err(|error| admission(runtime, error))?;
    if receipt.is_none() && initial_policy == HostAllocationPolicy::Managed {
        return Err(admission(runtime, HostAllocationError::MissingManagedOwner));
    }
    let cancellation = runtime.map(RuntimeContext::cancellation);
    let fence = || {
        if let (Some(runtime), Some(cancellation)) = (runtime, cancellation.as_ref()) {
            if !cancellation.same_generation(&runtime.cancellation()) {
                return Err(admission(
                    Some(runtime),
                    HostAllocationError::StaleAuthority,
                ));
            }
        }
        match (receipt, runtime) {
            (Some(receipt), Some(runtime)) => receipt
                .checkpoint(runtime)
                .map_err(|error| admission(Some(runtime), error)),
            (Some(_), None) => Err(admission(None, HostAllocationError::MissingManagedOwner)),
            (None, _) => {
                let policy = HostAllocationReceipt::policy(runtime)
                    .map_err(|error| admission(runtime, error))?;
                if policy != initial_policy {
                    return Err(admission(runtime, HostAllocationError::StaleAuthority));
                }
                Ok(())
            }
        }
    };
    let mut driver = Driver {
        operation,
        context: runtime,
        fence,
        cooperation: runtime.map(RuntimeContext::cpu_cooperation),
        scalar: std::marker::PhantomData,
    };
    runmat_sparse_solvers::krylov::pcg::solve::<T, Native<T>, _>(
        input.rhs,
        input.initial,
        runmat_sparse_solvers::krylov::pcg::PcgParameters {
            tolerance: input.tolerance,
            maximum_iterations: input.maximum_iterations,
            preconditioners,
        },
        input.work,
        input.operation_scratch,
        &mut driver,
    )
    .await
    .map_err(|error| {
        terminal(match error {
            KrylovFailure::Operation(original) => original,
            KrylovFailure::Invalid(message) => semantic_error("PcgRecurrence", message),
        })
    })
}
