use super::async_operations::{
    Coefficient, NativePcgVector, Operations, Preconditioner,
    RepeatedCallableVector as CallableOperation,
};
use super::kernel::{self, PcgOutcome, PcgSolveInput, PcgWork};
use super::request::{invalid, unsupported, PcgRequest};
use crate::builtins::math::linalg::ops::callable_linear_operator::{
    RepeatedCallableVector, VectorClass, VectorContract,
};
use crate::builtins::math::linalg::ops::linear_operator::{LinearOperator, TypedLinearOperator};
use crate::builtins::math::linalg::ops::sparse_operator::SparseOperatorError;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::host_effect::HostEffectAllocationError;
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_value::{ComplexElement, NumericDType, Value};
use std::marker::PhantomData;

mod scalar;
use scalar::{vector_shape, PcgValue};
pub(super) fn map_operator(error: SparseOperatorError) -> RuntimeError {
    map_operator_in_context(error, None)
}

pub(super) fn map_operator_in_context(
    error: SparseOperatorError,
    context: Option<&RuntimeContext>,
) -> RuntimeError {
    match error {
        SparseOperatorError::Admission(
            error @ (HostAllocationError::Cancelled
            | HostAllocationError::StaleAuthority
            | HostAllocationError::MissingManagedOwner
            | HostAllocationError::Resource(
                HostEffectAllocationError::Cancelled
                | HostEffectAllocationError::Revoked
                | HostEffectAllocationError::InvalidInvocation,
            )),
        ) => {
            let cancellation = context.map(RuntimeContext::cancellation_context);
            crate::builtin::builtin_control_error(cancellation.as_ref(), "pcg", error)
        }
        SparseOperatorError::Admission(error) => {
            build_runtime_error(format!("pcg: host admission failed: {error}"))
                .with_builtin("pcg")
                .with_identifier("RunMat:pcg:HostAllocation")
                .with_source(error)
                .build()
        }
        SparseOperatorError::UnsupportedClass => {
            unsupported("A requires native single or double dense/CSC storage")
        }
        SparseOperatorError::DimensionMismatch => invalid("vector dimensions do not match A"),
        SparseOperatorError::InvalidTopology(_) => invalid("invalid CSC coefficient topology"),
        SparseOperatorError::Overflow => invalid("matrix or iteration dimensions overflow"),
    }
}

fn validate_vector<'a, T: PcgValue>(
    value: &'a Value,
    n: usize,
    role: &str,
) -> BuiltinResult<super::super::iterative::value::NativeVector<'a, T>> {
    if let Some(value) = T::scalar(value) {
        return if n == 1 {
            Ok(super::super::iterative::value::NativeVector::Scalar(value))
        } else {
            Err(invalid(role))
        };
    }
    let shape = match value {
        Value::Tensor(tensor) => tensor.shape(),
        Value::ComplexTensor(tensor) => tensor.shape(),
        _ => return Err(unsupported("b and x0 must match A's floating class")),
    };
    if !vector_shape(shape, n) {
        return Err(invalid(role));
    }
    T::input(value).ok_or_else(|| unsupported("b and x0 must match A's floating class"))
}

mod output;
use output::{publish_admitted, publish_unbounded, quote, PcgQuote};
mod async_route;
#[cfg(test)]
pub(super) fn quoted_f64_plan_for_test(
    n: usize,
    maximum_iterations: usize,
    requested: Option<usize>,
) -> BuiltinResult<HostAllocationPlan> {
    Ok(quote::<f64>(n, maximum_iterations, 5, requested, None)?.plan)
}
async fn run_typed<T: PcgValue>(
    operator: &TypedLinearOperator<'_, T>,
    request: &PcgRequest,
    context: Option<&RuntimeContext>,
    requested: Option<usize>,
) -> BuiltinResult<Value> {
    let n = operator.rows();
    let rhs_input = validate_vector::<T>(
        &request.rhs,
        n,
        "b must be a column vector with size(A,1) rows",
    )?;
    let initial_input = request
        .initial
        .as_ref()
        .map(|value| validate_vector::<T>(value, n, "x0 must match b's column shape"))
        .transpose()?;
    let rhs = rhs_input.as_slice();
    let initial = initial_input.as_ref().map(|input| input.as_slice());
    let PcgQuote {
        main_count,
        residual_count,
        plan,
    } = quote::<T>(n, request.maximum_iterations, 5, requested, context)?;
    let admission = |error| map_operator_in_context(SparseOperatorError::Admission(error), context);
    let receipt = HostAllocationReceipt::admit(context, plan).map_err(admission)?;
    if let (Some(receipt), Some(context)) = (&receipt, context) {
        let mut main = receipt
            .temporary_numeric(context, T::DTYPE, main_count)
            .map_err(admission)?;
        let mut residuals = receipt
            .temporary_numeric(context, NumericDType::F64, residual_count)
            .map_err(admission)?;
        let work = T::scratch(&mut main).ok_or_else(|| invalid("scratch class mismatch"))?;
        let (x, rest) = work.split_at_mut(n);
        let (best, rest) = rest.split_at_mut(n);
        let (r, rest) = rest.split_at_mut(n);
        let (p, operation_scratch) = rest.split_at_mut(n);
        let norms = residuals
            .as_f64_slice_mut()
            .ok_or_else(|| invalid("residual scratch class mismatch"))?;
        let outcome = kernel::solve(
            PcgSolveInput {
                rhs,
                initial,
                tolerance: request.tolerance,
                maximum_iterations: request.maximum_iterations,
                work: PcgWork {
                    x,
                    best,
                    residual: r,
                    direction: p,
                    intermediate: None,
                    residual_norms: norms,
                    coefficients: None,
                },
                operation_scratch,
            },
            Some(context),
            |input, output| T::matvec(operator, input, output, Some(context)),
            || {
                HostAllocationReceipt::policy(Some(context))
                    .map(|_| ())
                    .map_err(SparseOperatorError::from)
            },
            Some(receipt),
        )
        .await?;
        publish_admitted::<T>(receipt, context, x, norms, outcome, requested).await
    } else {
        let mut main = vec![T::zero(); 5 * n];
        let mut norms = vec![0.0; residual_count];
        let (x, rest) = main.split_at_mut(n);
        let (best, rest) = rest.split_at_mut(n);
        let (r, rest) = rest.split_at_mut(n);
        let (p, operation_scratch) = rest.split_at_mut(n);
        let outcome = kernel::solve(
            PcgSolveInput {
                rhs,
                initial,
                tolerance: request.tolerance,
                maximum_iterations: request.maximum_iterations,
                work: PcgWork {
                    x,
                    best,
                    residual: r,
                    direction: p,
                    intermediate: None,
                    residual_norms: &mut norms,
                    coefficients: None,
                },
                operation_scratch,
            },
            context,
            |input, output| T::matvec(operator, input, output, context),
            || match context {
                Some(context) => HostAllocationReceipt::policy(Some(context))
                    .map(|_| ())
                    .map_err(SparseOperatorError::from),
                None => Ok(()),
            },
            None,
        )
        .await?;
        publish_unbounded::<T>(context, main, n, &norms, outcome, requested).await
    }
}

pub(super) async fn execute(
    request: PcgRequest,
    context: Option<&RuntimeContext>,
    requested: Option<usize>,
) -> BuiltinResult<Value> {
    if requested.is_some_and(|count| count > 5) {
        return Err(invalid("pcg has at most five outputs"));
    }
    HostAllocationReceipt::policy(context)
        .map_err(|error| map_operator_in_context(SparseOperatorError::Admission(error), context))?;
    let requires_async = context.is_some()
        || super::request::is_callable(&request.coefficient)
        || request.first_preconditioner.is_some()
        || request.second_preconditioner.is_some();
    if requires_async {
        let context = context.ok_or_else(|| {
            unsupported("callable and preconditioned pcg requires an active execution context")
        })?;
        let PcgRequest {
            coefficient,
            rhs,
            tolerance,
            maximum_iterations,
            mut first_preconditioner,
            mut second_preconditioner,
            initial,
        } = request;
        let mut pending = async_route::AsyncRequest {
            rhs: &rhs,
            initial: initial.as_ref(),
            tolerance,
            maximum_iterations,
            first: &mut first_preconditioner,
            second: &mut second_preconditioner,
        };
        macro_rules! solve_async {
            ($scalar:ty, $source:expr) => {
                async_route::run_async_typed::<$scalar>($source, &mut pending, context, requested)
                    .await
            };
        }
        if super::request::is_callable(&coefficient) {
            return match &rhs {
                Value::Num(_) => {
                    solve_async!(f64, async_route::CoefficientInput::Callable(coefficient))
                }
                Value::Complex(_, _) => solve_async!(
                    ComplexElement<f64>,
                    async_route::CoefficientInput::Callable(coefficient)
                ),
                Value::Tensor(tensor) if tensor.as_f32_slice().is_some() => {
                    solve_async!(f32, async_route::CoefficientInput::Callable(coefficient))
                }
                Value::Tensor(tensor) if tensor.as_f64_slice().is_some() => {
                    solve_async!(f64, async_route::CoefficientInput::Callable(coefficient))
                }
                Value::ComplexTensor(tensor) if tensor.as_f32_slice().is_some() => {
                    solve_async!(
                        ComplexElement<f32>,
                        async_route::CoefficientInput::Callable(coefficient)
                    )
                }
                Value::ComplexTensor(tensor) if tensor.as_f64_slice().is_some() => {
                    solve_async!(
                        ComplexElement<f64>,
                        async_route::CoefficientInput::Callable(coefficient)
                    )
                }
                _ => Err(unsupported(
                    "callable A requires a native floating b vector",
                )),
            };
        }
        let operator = LinearOperator::borrow_with_context(&coefficient, context)
            .map_err(|error| map_operator_in_context(error, Some(context)))?;
        return match operator {
            LinearOperator::RealF32(operator) => {
                solve_async!(f32, async_route::CoefficientInput::Matrix(operator))
            }
            LinearOperator::RealF64(operator) => {
                solve_async!(f64, async_route::CoefficientInput::Matrix(operator))
            }
            LinearOperator::ComplexF32(operator) => {
                solve_async!(
                    ComplexElement<f32>,
                    async_route::CoefficientInput::Matrix(operator)
                )
            }
            LinearOperator::ComplexF64(operator) => {
                solve_async!(
                    ComplexElement<f64>,
                    async_route::CoefficientInput::Matrix(operator)
                )
            }
        };
    }
    let operator = match context {
        Some(context) => LinearOperator::borrow_with_context(&request.coefficient, context),
        None => LinearOperator::borrow_unbounded(&request.coefficient),
    }
    .map_err(map_operator)?;
    match operator {
        LinearOperator::RealF32(operator) => {
            run_typed(&operator, &request, context, requested).await
        }
        LinearOperator::RealF64(operator) => {
            run_typed(&operator, &request, context, requested).await
        }
        LinearOperator::ComplexF32(operator) => {
            run_typed(&operator, &request, context, requested).await
        }
        LinearOperator::ComplexF64(operator) => {
            run_typed(&operator, &request, context, requested).await
        }
    }
}
