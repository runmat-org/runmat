//! Concrete coefficient and preconditioner routing for the shared PCG state.
//!
//! Dense/CSC products and prepared factors borrow their canonical matrix
//! Values. Callable routes are asynchronous and must supply their own
//! repeated-invocation owner; this layer never blocks on a language callback
//! or treats a temporary numeric slice as an escaping Value.

use runmat_value::ComplexElement;

use super::super::iterative::PreconditionerStatus;
use super::async_driver::{self, AsyncPcgOperation, PcgOperation};
use super::kernel::{PcgOutcome, PcgScalar, PcgSolveInput};
use crate::builtins::math::linalg::ops::iterative_preconditioner::{
    MatrixFactorError, MatrixFactorValue, PreparedMatrixFactor,
};
use crate::builtins::math::linalg::ops::linear_operator::TypedLinearOperator;
use crate::builtins::math::linalg::ops::sparse_operator::SparseOperatorError;
use crate::context::host_allocation::HostAllocationReceipt;
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

pub(super) fn preconditioner_error(
    error: MatrixFactorError,
    runtime: &RuntimeContext,
) -> RuntimeError {
    match error {
        MatrixFactorError::ManagedSparse(crate::builtins::math::linalg::ops::sparse_factor::ManagedSparseLuError::Execution(error)) => error.into_runtime_error_in_context(Some(runtime)),
        MatrixFactorError::Execution(error)
        | MatrixFactorError::Triangular(crate::builtins::math::linalg::ops::triangular_operator::TriangularOperatorError::Execution(error))
        | MatrixFactorError::Dense(crate::builtins::math::linalg::ops::dense_factor::DenseFactorError::Execution(error)) => error.into_runtime_error_in_context(Some(runtime)),
        MatrixFactorError::Admission(error)
        | MatrixFactorError::ManagedSparse(crate::builtins::math::linalg::ops::sparse_factor::ManagedSparseLuError::Admission(error))
        | MatrixFactorError::Dense(crate::builtins::math::linalg::ops::dense_factor::DenseFactorError::Admission(error))
        | MatrixFactorError::Operator(SparseOperatorError::Admission(error))
        | MatrixFactorError::Triangular(crate::builtins::math::linalg::ops::triangular_operator::TriangularOperatorError::Admission(error))
        | MatrixFactorError::Triangular(crate::builtins::math::linalg::ops::triangular_operator::TriangularOperatorError::Operator(SparseOperatorError::Admission(error))) => {
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
/// the existing product trait at concrete types; it defines no
/// second scalar arithmetic or conversion policy.
pub(super) trait NativePcgVector: PcgScalar + MatrixFactorValue {
    async fn matrix_into(
        operator: &TypedLinearOperator<'_, Self>,
        input: &[Self],
        output: &mut [Self],
        runtime: &RuntimeContext,
    ) -> Result<(), RuntimeError>;
}

macro_rules! native_vector {
    ($scalar:ty) => {
        impl NativePcgVector for $scalar {
            async fn matrix_into(
                operator: &TypedLinearOperator<'_, Self>,
                input: &[Self],
                output: &mut [Self],
                runtime: &RuntimeContext,
            ) -> Result<(), RuntimeError> {
                let mut budget = runtime.cpu_cooperation();
                operator.apply_cooperatively(input, output, runtime, &mut budget).await.map_err(|error| match error {
                    crate::builtins::math::linalg::ops::linear_operator::CooperativeOperatorError::Operator(error) => matrix_error(error, runtime),
                    crate::builtins::math::linalg::ops::linear_operator::CooperativeOperatorError::Execution(error) => error.into_runtime_error_in_context(Some(runtime)),
                })
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
    Matrix(PreparedMatrixFactor<'a, T>),
    Callable(C),
    NumericalFailure,
}

impl<'a, T: NativePcgVector, C> Preconditioner<'a, T, C> {
    pub(super) fn matrix(matrix: PreparedMatrixFactor<'a, T>) -> Self {
        Self::Matrix(matrix)
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
        let first = self.first.is_some();
        let second = self.second.is_some();
        async_driver::drive(input, self, Some(runtime), receipt, (first, second)).await
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
        runtime: Option<&RuntimeContext>,
    ) -> Result<PreconditionerStatus, RuntimeError> {
        let runtime = runtime.ok_or_else(|| {
            semantic_error(
                "PcgContext",
                "selected PCG operation has no invocation context",
            )
        })?;
        match operation {
            PcgOperation::Coefficient => match &mut self.coefficient {
                Coefficient::Matrix(matrix) => T::matrix_into(matrix, input, output, runtime)
                    .await
                    .map(|()| PreconditionerStatus::Applied),
                Coefficient::Callable(callable) => callable
                    .apply_into(input, output, runtime)
                    .await
                    .map(|()| PreconditionerStatus::Applied),
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
) -> Result<PreconditionerStatus, RuntimeError> {
    let Some(factor) = factor else {
        return Err(semantic_error(
            "PcgPreconditioner",
            "PCG requested an absent preconditioner",
        ));
    };
    match factor {
        Preconditioner::Matrix(matrix) => match matrix
            .solve_into(input, output, Some(runtime))
            .await
        {
            Ok(()) => Ok(PreconditionerStatus::Applied),
            Err(error) if error.numerical_failure() => Ok(PreconditionerStatus::NumericalFailure),
            Err(error) => Err(preconditioner_error(error, runtime)),
        },
        Preconditioner::NumericalFailure => Ok(PreconditionerStatus::NumericalFailure),
        Preconditioner::Callable(callable) => {
            callable.apply_into(input, output, runtime).await?;
            let mut budget = runtime.cpu_cooperation();
            for chunk in output.chunks(1024) {
                budget
                    .checkpoint(chunk.len() as u64)
                    .await
                    .map_err(|error| error.into_runtime_error_in_context(Some(runtime)))?;
                HostAllocationReceipt::policy(Some(runtime)).map_err(|error| {
                    crate::builtin::builtin_control_error(
                        Some(&runtime.cancellation_context()),
                        "pcg",
                        error,
                    )
                })?;
                if chunk.iter().any(|value| {
                    let (real, imaginary) = value.components();
                    !real.is_finite() || !imaginary.is_finite()
                }) {
                    return Ok(PreconditionerStatus::NumericalFailure);
                }
            }
            Ok(PreconditionerStatus::Applied)
        }
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
    use crate::execution::RuntimeExecutionService;
    use runmat_value::{SparseTensor, Tensor, Value};
    use std::rc::Rc;

    #[test]
    fn owned_sparse_preconditioner_reuse_and_typed_failure_boundary() {
        use crate::builtins::math::linalg::ops::sparse_factor::ManagedSparseLuError;
        use crate::context::host_allocation::HostAllocationError;
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let coefficient = Value::SparseTensor(
            SparseTensor::new(2, 2, vec![0, 1, 3], vec![1, 0, 1], vec![1., 2., 3.]).unwrap(),
        );
        let mut factor = futures::executor::block_on(PreparedMatrixFactor::<f64>::prepare(
            &coefficient,
            2,
            Some(&runtime),
        ))
        .unwrap();
        for (input, expected) in [([10., 19.], [4., 5.]), ([-2., -1.], [2., -1.])] {
            let mut output = [0.; 2];
            futures::executor::block_on(factor.solve_into(&input, &mut output, Some(&runtime)))
                .unwrap();
            for (actual, expected) in output.into_iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-12);
            }
        }
        let error = futures::executor::block_on(factor.solve_into(&[1., 1.], &mut [0.; 2], None))
            .unwrap_err();
        assert!(matches!(
            error,
            MatrixFactorError::Admission(HostAllocationError::MissingManagedOwner)
        ));
        assert!(
            !error.numerical_failure(),
            "missing authority must not become flag 2"
        );
        let singular = Value::SparseTensor(
            SparseTensor::new(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1], vec![1.; 4]).unwrap(),
        );
        let error = match futures::executor::block_on(PreparedMatrixFactor::<f64>::prepare(
            &singular,
            2,
            Some(&runtime),
        )) {
            Err(error) => error,
            Ok(_) => panic!("singular factor cannot publish"),
        };
        assert!(matches!(
            error,
            MatrixFactorError::ManagedSparse(ManagedSparseLuError::Singular { .. })
        ));
        assert!(
            error.numerical_failure(),
            "typed numerical failure retains flag 2 disposition"
        );
        for error in [
            MatrixFactorError::ManagedSparse(ManagedSparseLuError::Admission(
                HostAllocationError::StaleAuthority,
            )),
            MatrixFactorError::ManagedSparse(ManagedSparseLuError::Execution(
                crate::execution::ExecutionServiceError::Cancelled,
            )),
        ] {
            assert!(
                !error.numerical_failure(),
                "operational failures remain terminal"
            );
        }
    }

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
        let first = futures::executor::block_on(PreparedMatrixFactor::<f64>::prepare(
            &factor,
            1,
            Some(&runtime),
        ))
        .unwrap();
        let second = futures::executor::block_on(PreparedMatrixFactor::<f64>::prepare(
            &factor,
            1,
            Some(&runtime),
        ))
        .unwrap();
        let mut x = [0.0];
        let mut best = [0.0];
        let mut residual = [0.0];
        let mut direction = [0.0];
        let mut product = [0.0];
        let mut history = [0.0; 2];
        let mut operation = [0.0];
        let routes: Operations<'_, f64, NoCallable> = Operations::new(
            Coefficient::matrix(coefficient),
            Some(Preconditioner::matrix(first)),
            Some(Preconditioner::matrix(second)),
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
                    coefficients: None,
                },
                operation_scratch: &mut operation,
            },
            &runtime,
            None,
        ))
        .unwrap();
        assert_eq!(outcome.flag, 0);
        assert_eq!(x, [2.0]);

        // A nontriangular CSC factor must use one prepared general solve,
        // retaining its actual sparse fill across both applications.
        let coefficient = Value::Tensor(Tensor::new(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2]).unwrap());
        let factor = Value::SparseTensor(
            SparseTensor::new(
                2,
                2,
                vec![0, 2, 4],
                vec![0, 1, 0, 1],
                vec![2.0, 1.0, 1.0, 2.0],
            )
            .unwrap(),
        );
        let LinearOperator::RealF64(coefficient) =
            LinearOperator::borrow_with_context(&coefficient, &runtime).unwrap()
        else {
            panic!("double coefficient");
        };
        let first = futures::executor::block_on(PreparedMatrixFactor::<f64>::prepare(
            &factor,
            2,
            Some(&runtime),
        ))
        .unwrap();
        let second = futures::executor::block_on(PreparedMatrixFactor::<f64>::prepare(
            &factor,
            2,
            Some(&runtime),
        ))
        .unwrap();
        let mut x = [0.0; 2];
        let mut best = [0.0; 2];
        let mut residual = [0.0; 2];
        let mut direction = [0.0; 2];
        let mut intermediate = [0.0; 2];
        let mut history = [0.0; 2];
        let mut operation = [0.0; 2];
        let routes: Operations<'_, f64, NoCallable> = Operations::new(
            Coefficient::matrix(coefficient),
            Some(Preconditioner::matrix(first)),
            Some(Preconditioner::matrix(second)),
        );
        let outcome = futures::executor::block_on(routes.solve(
            PcgSolveInput {
                rhs: &[3.0, 3.0],
                initial: None,
                tolerance: 1e-12,
                maximum_iterations: 1,
                work: super::super::kernel::PcgWork {
                    x: &mut x,
                    best: &mut best,
                    residual: &mut residual,
                    direction: &mut direction,
                    intermediate: Some(&mut intermediate),
                    residual_norms: &mut history,
                    coefficients: None,
                },
                operation_scratch: &mut operation,
            },
            &runtime,
            None,
        ))
        .unwrap();
        assert_eq!(outcome.flag, 0);
        for value in x {
            assert!((value - 1.0).abs() < 1e-12);
        }
    }
}
