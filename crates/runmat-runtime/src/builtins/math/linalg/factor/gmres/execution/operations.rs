//! Selected coefficient and preconditioner owners for one GMRES solve.

use runmat_value::Value;

use super::super::super::iterative::callback;
use super::super::super::iterative::controls::is_callable;
use super::super::super::iterative::value::IterativeValue;
use super::super::super::iterative::PreconditionerStatus;
use super::super::recurrence::GmresOperations;
use super::{admission, factor_error, invalid, operator_error, CALLBACK_ERRORS};
use crate::builtins::math::linalg::ops::callable_linear_operator::RepeatedCallableVector;
use crate::builtins::math::linalg::ops::iterative_preconditioner::{
    MatrixFactorValue, PreparedMatrixFactor,
};
use crate::builtins::math::linalg::ops::linear_operator::TypedLinearOperator;
use crate::context::host_allocation::HostAllocationReceipt;
use crate::context::RuntimeContext;
use crate::RuntimeError;

pub(super) enum Coefficient<'a, T> {
    Matrix(TypedLinearOperator<'a, T>),
    Callable(&'a RepeatedCallableVector),
}

pub(super) enum Preconditioner<'a, T: MatrixFactorValue> {
    Matrix(PreparedMatrixFactor<'a, T>),
    Callable(&'a RepeatedCallableVector),
    NumericalFailure,
}

impl<T: IterativeValue + MatrixFactorValue> Preconditioner<'_, T> {
    async fn apply(
        &mut self,
        input: &[T],
        output: &mut [T],
        context: Option<&RuntimeContext>,
    ) -> Result<PreconditionerStatus, RuntimeError> {
        match self {
            Self::Matrix(factor) => match factor.solve_into(input, output, context).await {
                Ok(()) => Ok(PreconditionerStatus::Applied),
                Err(error) if error.numerical_failure() => {
                    Ok(PreconditionerStatus::NumericalFailure)
                }
                Err(error) => Err(factor_error(context, error)),
            },
            Self::NumericalFailure => Ok(PreconditionerStatus::NumericalFailure),
            Self::Callable(selected) => {
                callback::apply(
                    selected,
                    context.ok_or_else(|| invalid("callable requires execution context"))?,
                    input,
                    output,
                    CALLBACK_ERRORS,
                )
                .await?;
                let mut budget = context.map(RuntimeContext::cpu_cooperation);
                for chunk in output.chunks(1024) {
                    if let Some(budget) = budget.as_mut() {
                        budget
                            .checkpoint(chunk.len() as u64)
                            .await
                            .map_err(|error| error.into_runtime_error_in_context(context))?;
                    }
                    HostAllocationReceipt::policy(context)
                        .map_err(|error| admission(context, error))?;
                    if !chunk.iter().all(|&value| T::finite(T::to_factor(value))) {
                        return Ok(PreconditionerStatus::NumericalFailure);
                    }
                }
                Ok(PreconditionerStatus::Applied)
            }
        }
    }
}

pub(super) fn select_callable<T: IterativeValue>(
    source: &mut Option<Value>,
    order: usize,
    context: Option<&RuntimeContext>,
) -> Result<Option<RepeatedCallableVector>, RuntimeError> {
    if source.as_ref().is_some_and(is_callable) {
        let context = context.ok_or_else(|| invalid("callable requires execution context"))?;
        let target = source
            .take()
            .ok_or_else(|| invalid("missing preconditioner"))?;
        return Ok(Some(RepeatedCallableVector::select(
            context,
            target,
            callback::contract::<T>(order, CALLBACK_ERRORS)?,
        )?));
    }
    Ok(None)
}

pub(super) async fn prepare_preconditioner<'a, T: IterativeValue + MatrixFactorValue + 'a>(
    source: Option<&'a Value>,
    callable: Option<&'a RepeatedCallableVector>,
    order: usize,
    context: Option<&'a RuntimeContext>,
) -> Result<Option<Preconditioner<'a, T>>, RuntimeError> {
    if let Some(callable) = callable {
        return Ok(Some(Preconditioner::Callable(callable)));
    }
    match source {
        Some(value) => match PreparedMatrixFactor::<T>::prepare(value, order, context).await {
            Ok(factor) => Ok(Some(Preconditioner::Matrix(factor))),
            Err(error) if error.numerical_failure() => Ok(Some(Preconditioner::NumericalFailure)),
            Err(error) => Err(factor_error(context, error)),
        },
        None => Ok(None),
    }
}

pub(super) struct SolverOperations<'a, T: MatrixFactorValue> {
    pub(super) coefficient: Coefficient<'a, T>,
    pub(super) first: Option<Preconditioner<'a, T>>,
    pub(super) second: Option<Preconditioner<'a, T>>,
    pub(super) context: Option<&'a RuntimeContext>,
    pub(super) receipt: Option<&'a HostAllocationReceipt>,
    pub(super) cooperation: Option<crate::execution::CpuCooperationBudget>,
}

impl<T: IterativeValue + MatrixFactorValue> GmresOperations<T> for SolverOperations<'_, T> {
    type Error = RuntimeError;

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
            FeaRunDenied::from(LedgerError::from(cause)).into_runtime_error("gmres")
        })?;
        let mut reservation = reserve_work(&usage, UsageCategory::Solve, maximum, self.context)
            .await
            .map_err(|denied| denied.into_runtime_error("gmres"))?;
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
                .map_err(|cause| FeaRunDenied::from(cause).into_runtime_error("gmres"))?;
        }
        self.checkpoint()?;
        Ok(result)
    }

    async fn coefficient(&mut self, input: &[T], output: &mut [T]) -> Result<(), Self::Error> {
        match &self.coefficient {
            Coefficient::Matrix(operator) => {
                if let (Some(context), Some(budget)) = (self.context, self.cooperation.as_mut()) {
                    T::cooperative_matvec(operator, input, output, context, budget).await.map_err(|error| match error {
                        crate::builtins::math::linalg::ops::linear_operator::CooperativeOperatorError::Operator(error) => operator_error(self.context, error),
                        crate::builtins::math::linalg::ops::linear_operator::CooperativeOperatorError::Execution(error) => error.into_runtime_error_in_context(self.context),
                    })
                } else {
                    T::matvec(operator, input, output, self.context)
                        .map_err(|error| operator_error(self.context, error))
                }
            }
            Coefficient::Callable(selected) => {
                callback::apply(
                    selected,
                    self.context
                        .ok_or_else(|| invalid("callable requires execution context"))?,
                    input,
                    output,
                    CALLBACK_ERRORS,
                )
                .await
            }
        }
    }

    async fn precondition(
        &mut self,
        input: &[T],
        output: &mut [T],
        intermediate: &mut [T],
    ) -> Result<PreconditionerStatus, Self::Error> {
        self.checkpoint()?;
        match (&mut self.first, &mut self.second) {
            (Some(first), Some(second)) => {
                let status = first.apply(input, intermediate, self.context).await?;
                if status == PreconditionerStatus::NumericalFailure {
                    return Ok(status);
                }
                let status = second.apply(intermediate, output, self.context).await?;
                if status == PreconditionerStatus::NumericalFailure {
                    return Ok(status);
                }
            }
            (Some(first), None) => {
                let status = first.apply(input, output, self.context).await?;
                if status == PreconditionerStatus::NumericalFailure {
                    return Ok(status);
                }
            }
            (None, Some(second)) => {
                let status = second.apply(input, output, self.context).await?;
                if status == PreconditionerStatus::NumericalFailure {
                    return Ok(status);
                }
            }
            (None, None) => {
                for (input, output) in input.chunks(1024).zip(output.chunks_mut(1024)) {
                    self.cooperate(input.len() as u64).await?;
                    output.copy_from_slice(input);
                }
            }
        }
        self.checkpoint()?;
        Ok(PreconditionerStatus::Applied)
    }

    fn checkpoint(&mut self) -> Result<(), Self::Error> {
        match (self.receipt, self.context) {
            (Some(receipt), Some(context)) => receipt
                .checkpoint(context)
                .map_err(|error| admission(Some(context), error)),
            (None, Some(context)) => {
                super::super::super::iterative::copy::unbounded_checkpoint(Some(context), "gmres")
            }
            (None, None) => Ok(()),
            (Some(_), None) => Err(invalid("managed receipt has no execution context")),
        }
    }

    async fn cooperate(&mut self, completed: u64) -> Result<(), Self::Error> {
        self.checkpoint()?;
        if let Some(budget) = self.cooperation.as_mut() {
            budget
                .checkpoint(completed)
                .await
                .map_err(|e| e.into_runtime_error_in_context(self.context))?;
        }
        self.checkpoint()
    }
}
