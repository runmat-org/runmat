//! Concrete coefficient and preconditioner routing for the shared PCG state.
//!
//! Dense/CSC products and validated triangular solves borrow their canonical
//! matrix Values. Callable routes are asynchronous and must supply their own
//! repeated-invocation owner; this layer never blocks on a language callback
//! or treats a temporary numeric slice as an escaping Value.

use runmat_value::ComplexElement;

use super::async_driver::{self, AsyncPcgOperation, PcgOperation};
use super::kernel::{PcgOutcome, PcgScalar, PcgSolveInput};
use super::recurrence_state::RecurrenceState;
use crate::builtins::math::linalg::ops::linear_operator::TypedLinearOperator;
use crate::builtins::math::linalg::ops::sparse_operator::SparseOperatorError;
use crate::builtins::math::linalg::ops::triangular_operator::{
    TriangularOperator, TriangularOperatorError,
};
use crate::context::host_allocation::{HostAllocationPolicy, HostAllocationReceipt};
use crate::context::RuntimeContext;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

fn matrix_error(error: SparseOperatorError, runtime: &RuntimeContext) -> RuntimeError {
    match error {
        SparseOperatorError::Admission(error) => crate::builtin::builtin_control_error(
            Some(&runtime.cancellation_context()),
            "pcg",
            error,
        ),
        other => semantic_error(
            "PcgLinearOperator",
            format!("pcg coefficient failed: {other:?}"),
        ),
    }
}

pub(super) fn triangular_error(
    error: TriangularOperatorError,
    runtime: &RuntimeContext,
) -> RuntimeError {
    match error {
        TriangularOperatorError::Admission(error)
        | TriangularOperatorError::Operator(SparseOperatorError::Admission(error)) => {
            crate::builtin::builtin_control_error(
                Some(&runtime.cancellation_context()),
                "pcg",
                error,
            )
        }
        other => crate::runtime_error::semantic_error_with_source(
            "PcgPreconditioner",
            format!("pcg preconditioner failed: {other}"),
            other,
        ),
    }
}

/// Only the four native PCG classes participate. This adapter delegates to
/// the existing product/triangular traits at concrete types; it defines no
/// second scalar arithmetic or conversion policy.
pub(super) trait NativePcgVector: PcgScalar {
    fn matrix_into(
        operator: &TypedLinearOperator<'_, Self>,
        input: &[Self],
        output: &mut [Self],
        runtime: &RuntimeContext,
    ) -> Result<(), SparseOperatorError>;

    fn triangular_into(
        operator: &TriangularOperator<'_, Self>,
        input: &[Self],
        output: &mut [Self],
        runtime: &RuntimeContext,
    ) -> Result<(), TriangularOperatorError>;
}

macro_rules! native_vector {
    ($scalar:ty) => {
        impl NativePcgVector for $scalar {
            fn matrix_into(
                operator: &TypedLinearOperator<'_, Self>,
                input: &[Self],
                output: &mut [Self],
                runtime: &RuntimeContext,
            ) -> Result<(), SparseOperatorError> {
                operator.apply_into(input, output, runtime)
            }

            fn triangular_into(
                operator: &TriangularOperator<'_, Self>,
                input: &[Self],
                output: &mut [Self],
                runtime: &RuntimeContext,
            ) -> Result<(), TriangularOperatorError> {
                operator.apply_into(input, output, runtime)
            }
        }
    };
}

native_vector!(f32);
native_vector!(f64);
native_vector!(ComplexElement<f32>);
native_vector!(ComplexElement<f64>);

/// A callable must retain an authentic target and independently own/root each
/// input and output across `await`. A borrowed recurrence slice is only a
/// source for an exactly admitted operation input, never its owner.
pub(super) trait RepeatedCallableVector<T: NativePcgVector> {
    async fn apply_into(
        &mut self,
        input: &[T],
        output: &mut [T],
        runtime: &RuntimeContext,
    ) -> Result<(), RuntimeError>;
}

pub(super) enum Coefficient<'a, T: NativePcgVector, C> {
    Matrix(TypedLinearOperator<'a, T>),
    Callable(C),
}

impl<'a, T: NativePcgVector, C> Coefficient<'a, T, C> {
    pub(super) fn matrix(matrix: TypedLinearOperator<'a, T>) -> Self {
        Self::Matrix(matrix)
    }

    pub(super) fn callable(callable: C) -> Self {
        Self::Callable(callable)
    }
}

pub(super) enum Preconditioner<'a, T: NativePcgVector, C> {
    Triangular(TriangularOperator<'a, T>),
    Callable(C),
}

impl<'a, T: NativePcgVector, C> Preconditioner<'a, T, C> {
    pub(super) fn triangular(triangular: TriangularOperator<'a, T>) -> Self {
        Self::Triangular(triangular)
    }

    pub(super) fn callable(callable: C) -> Self {
        Self::Callable(callable)
    }
}

pub(super) struct Operations<'a, T: NativePcgVector, C> {
    coefficient: Coefficient<'a, T, C>,
    first: Option<Preconditioner<'a, T, C>>,
    second: Option<Preconditioner<'a, T, C>>,
}

impl<'a, T: NativePcgVector, C> Operations<'a, T, C> {
    pub(super) fn new(
        coefficient: Coefficient<'a, T, C>,
        first: Option<Preconditioner<'a, T, C>>,
        second: Option<Preconditioner<'a, T, C>>,
    ) -> Self {
        Self {
            coefficient,
            first,
            second,
        }
    }

    pub(super) async fn solve(
        self,
        input: PcgSolveInput<'_, T>,
        runtime: &RuntimeContext,
        receipt: Option<&HostAllocationReceipt>,
    ) -> Result<PcgOutcome, RuntimeError>
    where
        C: RepeatedCallableVector<T>,
    {
        let policy = HostAllocationReceipt::policy(Some(runtime)).map_err(|error| {
            crate::builtin::builtin_control_error(
                Some(&runtime.cancellation_context()),
                "pcg",
                error,
            )
        })?;
        if receipt.is_none() && policy == HostAllocationPolicy::Managed {
            return Err(crate::builtin::builtin_control_error(
                Some(&runtime.cancellation_context()),
                "pcg",
                crate::context::host_allocation::HostAllocationError::MissingManagedOwner,
            ));
        }
        let first = self.first.is_some();
        let second = self.second.is_some();
        let mut checkpoint = || {
            if let Some(receipt) = receipt {
                receipt.checkpoint(runtime).map_err(|error| {
                    crate::builtin::builtin_control_error(
                        Some(&runtime.cancellation_context()),
                        "pcg",
                        error,
                    )
                })
            } else {
                HostAllocationReceipt::policy(Some(runtime))
                    .map(|_| ())
                    .map_err(|error| {
                        crate::builtin::builtin_control_error(
                            Some(&runtime.cancellation_context()),
                            "pcg",
                            error,
                        )
                    })
            }
        };
        let recurrence = RecurrenceState::new(
            input.rhs,
            input.initial,
            input.tolerance,
            input.maximum_iterations,
            input.work,
            (first, second),
            &mut checkpoint,
        )?;
        async_driver::drive(recurrence, self, input.operation_scratch, runtime, receipt).await
    }
}

impl<T: NativePcgVector, C: RepeatedCallableVector<T>> AsyncPcgOperation<T>
    for Operations<'_, T, C>
{
    async fn apply(
        &mut self,
        operation: PcgOperation,
        input: &[T],
        output: &mut [T],
        runtime: &RuntimeContext,
    ) -> Result<(), RuntimeError> {
        match operation {
            PcgOperation::Coefficient => match &mut self.coefficient {
                Coefficient::Matrix(matrix) => T::matrix_into(matrix, input, output, runtime)
                    .map_err(|error| matrix_error(error, runtime)),
                Coefficient::Callable(callable) => {
                    callable.apply_into(input, output, runtime).await
                }
            },
            PcgOperation::FirstPreconditioner => {
                apply_preconditioner(self.first.as_mut(), input, output, runtime).await
            }
            PcgOperation::SecondPreconditioner => {
                apply_preconditioner(self.second.as_mut(), input, output, runtime).await
            }
        }
    }
}

async fn apply_preconditioner<T: NativePcgVector, C: RepeatedCallableVector<T>>(
    factor: Option<&mut Preconditioner<'_, T, C>>,
    input: &[T],
    output: &mut [T],
    runtime: &RuntimeContext,
) -> Result<(), RuntimeError> {
    let Some(factor) = factor else {
        return Err(semantic_error(
            "PcgPreconditioner",
            "PCG requested an absent preconditioner",
        ));
    };
    match factor {
        Preconditioner::Triangular(triangle) => {
            T::triangular_into(triangle, input, output, runtime)
                .map_err(|error| triangular_error(error, runtime))
        }
        Preconditioner::Callable(callable) => callable.apply_into(input, output, runtime).await,
    }
}

/// Deliberate absence of a language callable in a native-only solve. This is
/// not an identity preconditioner: the recurrence receives `None` for M1/M2.
#[cfg(test)]
pub(super) struct NoCallable;

#[cfg(test)]
impl<T: NativePcgVector> RepeatedCallableVector<T> for NoCallable {
    async fn apply_into(
        &mut self,
        _input: &[T],
        _output: &mut [T],
        _runtime: &RuntimeContext,
    ) -> Result<(), RuntimeError> {
        Err(semantic_error(
            "PcgCallable",
            "no callable route was selected for this PCG solve",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::math::linalg::ops::linear_operator::LinearOperator;
    use crate::builtins::math::linalg::ops::triangular_operator::{
        Diagonal, Triangle, TriangularLinearOperator,
    };
    use crate::execution::RuntimeExecutionService;
    use runmat_value::{Tensor, Value};
    use std::rc::Rc;

    #[test]
    fn two_explicit_triangular_factors_feed_one_async_recurrence() {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let coefficient = Value::Tensor(Tensor::new(vec![4.0], vec![1, 1]).unwrap());
        let factor = Value::Tensor(Tensor::new(vec![2.0], vec![1, 1]).unwrap());
        let LinearOperator::RealF64(coefficient) =
            LinearOperator::borrow_with_context(&coefficient, &runtime).unwrap()
        else {
            panic!("double coefficient");
        };
        let TriangularLinearOperator::RealF64(factor) =
            TriangularLinearOperator::borrow_with_context(
                &factor,
                Triangle::Lower,
                Diagonal::Explicit,
                &runtime,
            )
            .unwrap()
        else {
            panic!("double triangular factor");
        };
        let mut x = [0.0];
        let mut best = [0.0];
        let mut residual = [0.0];
        let mut direction = [0.0];
        let mut product = [0.0];
        let mut history = [0.0; 2];
        let mut operation = [0.0];
        let routes: Operations<'_, f64, NoCallable> = Operations::new(
            Coefficient::matrix(coefficient),
            Some(Preconditioner::triangular(factor)),
            Some(Preconditioner::triangular(factor)),
        );
        let outcome = futures::executor::block_on(routes.solve(
            PcgSolveInput {
                rhs: &[8.0],
                initial: None,
                tolerance: 1e-12,
                maximum_iterations: 1,
                work: super::super::kernel::PcgWork {
                    x: &mut x,
                    best: &mut best,
                    residual: &mut residual,
                    direction: &mut direction,
                    intermediate: Some(&mut product),
                    residual_norms: &mut history,
                },
                operation_scratch: &mut operation,
            },
            &runtime,
            None,
        ))
        .unwrap();
        assert_eq!(outcome.flag, 0);
        assert_eq!(x, [2.0]);
    }
}
