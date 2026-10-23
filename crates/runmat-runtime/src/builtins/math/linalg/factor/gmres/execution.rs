//! Native borrowed dense/CSC execution of the shared GMRES recurrence. This
//! is a private dependency of the complete callable/preconditioned route;
//! it is not a separate public overload or a dense sparse fallback.

use num_complex::Complex64;
use runmat_value::{ComplexElement, Value};

use super::super::iterative::callback::{self, CallbackErrorPolicy};
use super::super::iterative::controls::is_callable;
use super::super::iterative::value::{vector_shape, IterativeValue};
use super::output::{publish_admitted, publish_unbounded, quote};
use super::recurrence;
use super::request::{invalid, GmresRequest};
use super::scalar::GmresScalar;
use super::workspace::GmresGeometry;
use crate::builtins::math::linalg::ops::callable_linear_operator::RepeatedCallableVector;
use crate::builtins::math::linalg::ops::iterative_preconditioner::MatrixFactorError;
use crate::builtins::math::linalg::ops::sparse_operator::SparseOperatorError;
use crate::context::host_allocation::{HostAllocationError, HostAllocationReceipt};
use crate::context::RuntimeContext;
use crate::runtime_error::semantic_error_with_source;
use crate::{BuiltinResult, RuntimeError};

const CALLBACK_ERRORS: CallbackErrorPolicy = CallbackErrorPolicy {
    name: "gmres",
    invalid,
};

pub(super) fn admission(
    context: Option<&RuntimeContext>,
    error: HostAllocationError,
) -> RuntimeError {
    let cancellation = context.map(RuntimeContext::cancellation_context);
    match error {
        error @ (HostAllocationError::Cancelled
        | HostAllocationError::StaleAuthority
        | HostAllocationError::MissingManagedOwner
        | HostAllocationError::Resource(
            crate::context::host_effect::HostEffectAllocationError::Cancelled
            | crate::context::host_effect::HostEffectAllocationError::Revoked
            | crate::context::host_effect::HostEffectAllocationError::InvalidInvocation,
        )) => crate::builtin::builtin_control_error(cancellation.as_ref(), "gmres", error),
        error => crate::build_runtime_error(format!("gmres: host admission failed: {error}"))
            .with_builtin("gmres")
            .with_identifier("RunMat:gmres:HostAllocation")
            .with_source(error)
            .build(),
    }
}

fn operator_error(context: Option<&RuntimeContext>, error: SparseOperatorError) -> RuntimeError {
    match error {
        SparseOperatorError::Admission(error) => admission(context, error),
        other => semantic_error_with_source(
            "RunMat:gmres:Operator",
            format!("gmres coefficient application failed: {other:?}"),
            MatrixFactorError::Operator(other),
        ),
    }
}

fn factor_error(context: Option<&RuntimeContext>, error: MatrixFactorError) -> RuntimeError {
    match error {
        MatrixFactorError::ManagedSparse(crate::builtins::math::linalg::ops::sparse_factor::ManagedSparseLuError::Execution(error)) => error.into_runtime_error_in_context(context),
        MatrixFactorError::Execution(error)
        | MatrixFactorError::Triangular(crate::builtins::math::linalg::ops::triangular_operator::TriangularOperatorError::Execution(error))
        | MatrixFactorError::Dense(crate::builtins::math::linalg::ops::dense_factor::DenseFactorError::Execution(error)) => error.into_runtime_error_in_context(context),
        MatrixFactorError::Admission(error)
        | MatrixFactorError::ManagedSparse(crate::builtins::math::linalg::ops::sparse_factor::ManagedSparseLuError::Admission(error))
        | MatrixFactorError::Dense(crate::builtins::math::linalg::ops::dense_factor::DenseFactorError::Admission(error))
        | MatrixFactorError::Operator(SparseOperatorError::Admission(error))
        | MatrixFactorError::Triangular(crate::builtins::math::linalg::ops::triangular_operator::TriangularOperatorError::Admission(error))
        | MatrixFactorError::Triangular(crate::builtins::math::linalg::ops::triangular_operator::TriangularOperatorError::Operator(SparseOperatorError::Admission(error))) => admission(context, error),
        other => semantic_error_with_source(
            "RunMat:gmres:Preconditioner",
            format!("gmres preconditioner failed: {other:?}"),
            other,
        ),
    }
}

mod operations;
mod scalar;
use operations::{prepare_preconditioner, select_callable, Coefficient, SolverOperations};
use scalar::GmresValue;

fn vector<'a, T: IterativeValue>(
    value: &'a runmat_value::Value,
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
        runmat_value::Value::Tensor(tensor) => tensor.shape(),
        runmat_value::Value::ComplexTensor(tensor) => tensor.shape(),
        _ => return Err(invalid(role)),
    };
    if !vector_shape(shape, n) {
        return Err(invalid(role));
    }
    T::input(value).ok_or_else(|| invalid(role))
}

fn zeroed<T: Copy>(
    count: usize,
    zero: T,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<Vec<T>> {
    super::super::iterative::copy::unbounded_checkpoint(context, "gmres")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| invalid("unable to reserve gmres scratch"))?;
    while values.len() < count {
        super::super::iterative::copy::unbounded_checkpoint(context, "gmres")?;
        values.resize(count.min(values.len().saturating_add(1024)), zero);
    }
    Ok(values)
}

pub(super) async fn execute(
    request: GmresRequest,
    context: Option<&RuntimeContext>,
    requested: Option<usize>,
) -> BuiltinResult<Value> {
    enum NativeClass {
        RealF32,
        RealF64,
        ComplexF32,
        ComplexF64,
    }
    let class = match &request.rhs {
        Value::Num(_) => NativeClass::RealF64,
        Value::Complex(_, _) => NativeClass::ComplexF64,
        Value::Tensor(tensor) if tensor.as_f32_slice().is_some() => NativeClass::RealF32,
        Value::Tensor(tensor) if tensor.as_f64_slice().is_some() => NativeClass::RealF64,
        Value::ComplexTensor(tensor) if tensor.as_f32_slice().is_some() => NativeClass::ComplexF32,
        Value::ComplexTensor(tensor) if tensor.as_f64_slice().is_some() => NativeClass::ComplexF64,
        _ => return Err(invalid("b must be a native floating vector")),
    };
    match class {
        NativeClass::RealF32 => run_owned::<f32>(request, context, requested).await,
        NativeClass::RealF64 => run_owned::<f64>(request, context, requested).await,
        NativeClass::ComplexF32 => {
            run_owned::<ComplexElement<f32>>(request, context, requested).await
        }
        NativeClass::ComplexF64 => {
            run_owned::<ComplexElement<f64>>(request, context, requested).await
        }
    }
}

async fn run_owned<T: IterativeValue + GmresScalar + GmresValue>(
    request: GmresRequest,
    context: Option<&RuntimeContext>,
    requested: Option<usize>,
) -> BuiltinResult<Value> {
    if requested.is_some_and(|count| count > 5) {
        return Err(invalid("gmres has at most five outputs"));
    }
    let GmresRequest {
        coefficient,
        rhs,
        restart,
        max_outer,
        tolerance,
        first_preconditioner,
        second_preconditioner,
        initial,
    } = request;
    let n = T::input(&rhs)
        .ok_or_else(|| invalid("b must have the selected floating class"))?
        .as_slice()
        .len();
    if n == 0 {
        return Err(invalid("A and b must have nonzero order"));
    }
    let rhs_input = vector::<T>(&rhs, n, "b must match A's n-by-1 floating class")?;
    let initial_input = initial
        .as_ref()
        .map(|value| vector::<T>(value, n, "x0 must match b's n-by-1 floating class"))
        .transpose()?;
    let rhs = rhs_input.as_slice();
    let initial = initial_input.as_ref().map(|input| input.as_slice());
    // Move selected callable targets before borrowing any matrix field. The
    // remaining Value owners outlive all borrowed matrix factors and solve
    // futures; no target/capture forest is cloned to satisfy this geometry.
    let mut coefficient = Some(coefficient);
    let callable = if coefficient.as_ref().is_some_and(is_callable) {
        let context = context.ok_or_else(|| invalid("callable requires execution context"))?;
        let target = coefficient
            .take()
            .ok_or_else(|| invalid("missing coefficient"))?;
        Some(RepeatedCallableVector::select(
            context,
            target,
            callback::contract::<T>(n, CALLBACK_ERRORS)?,
        )?)
    } else {
        None
    };
    let coefficient = match callable.as_ref() {
        Some(callable) => Coefficient::Callable(callable),
        None => {
            let source = coefficient
                .as_ref()
                .ok_or_else(|| invalid("missing coefficient"))?;
            let matrix = T::borrow_coefficient(source, context)
                .map_err(|error| factor_error(context, error))?;
            if matrix.rows() != n || matrix.cols() != n {
                return Err(invalid("A must be a nonempty square operator matching b"));
            }
            Coefficient::Matrix(matrix)
        }
    };
    let mut first_source = first_preconditioner;
    let mut second_source = second_preconditioner;
    let first_callable = select_callable::<T>(&mut first_source, n, context)?;
    let first =
        prepare_preconditioner::<T>(first_source.as_ref(), first_callable.as_ref(), n, context)
            .await?;
    let second_callable = select_callable::<T>(&mut second_source, n, context)?;
    let second =
        prepare_preconditioner::<T>(second_source.as_ref(), second_callable.as_ref(), n, context)
            .await?;
    let geometry = GmresGeometry::checked(n, restart, max_outer)
        .ok_or_else(|| invalid("gmres workspace dimensions overflow"))?;
    let plan = quote::<T>(geometry, requested).map_err(|error| admission(context, error))?;
    let receipt =
        HostAllocationReceipt::admit(context, plan).map_err(|error| admission(context, error))?;
    let mut operations = SolverOperations {
        coefficient,
        first,
        second,
        context,
        receipt: receipt.as_ref(),
        cooperation: context.map(RuntimeContext::cpu_cooperation),
    };
    // Once the admitted execution begins, any operator/preconditioner,
    // checkpoint or publication error is terminal. Preserve its original
    // diagnostic/cause; dispatcher gather-and-retry must not replay effects.
    async {
        if let (Some(receipt), Some(context)) = (receipt.as_ref(), context) {
            let native_count = geometry
                .native_elements
                .checked_mul(T::LANES)
                .ok_or_else(|| invalid("native scratch dimensions overflow"))?;
            let complex_count = geometry.hessenberg_elements + geometry.complex_small_elements;
            let real_count = geometry.real_small_elements + geometry.history_elements;
            let mut native = receipt
                .temporary_numeric(context, T::DTYPE, native_count)
                .map_err(|error| admission(Some(context), error))?;
            let mut complex = receipt
                .temporary_typed::<Complex64>(context, complex_count)
                .map_err(|error| admission(Some(context), error))?;
            let mut real = receipt
                .temporary_typed::<f64>(context, real_count)
                .map_err(|error| admission(Some(context), error))?;
            let native =
                T::scratch(&mut native).ok_or_else(|| invalid("native scratch class mismatch"))?;
            let mut work = geometry
                .project(native, complex.as_mut_slice(), real.as_mut_slice())
                .ok_or_else(|| invalid("gmres scratch projection mismatch"))?;
            let outcome = recurrence::solve(
                rhs,
                initial,
                recurrence::GmresParameters {
                    restart,
                    max_outer,
                    tolerance,
                },
                &mut work,
                &mut operations,
            )
            .await
            .map_err(recurrence::into_runtime_error)?;
            let solution = if outcome.flag == 0 {
                &*work.x
            } else {
                &*work.best
            };
            recurrence::GmresOperations::checkpoint(&mut operations)?;
            publish_admitted::<T>(receipt, context, solution, work.history, outcome, requested)
                .await
        } else {
            let complex_count = geometry.hessenberg_elements + geometry.complex_small_elements;
            let real_count = geometry.real_small_elements + geometry.history_elements;
            let mut native = zeroed(geometry.native_elements, T::zero(), context)?;
            let mut complex = zeroed(complex_count, Complex64::new(0.0, 0.0), context)?;
            let mut real = zeroed(real_count, 0.0f64, context)?;
            let mut work = geometry
                .project(&mut native, &mut complex, &mut real)
                .ok_or_else(|| invalid("gmres scratch projection mismatch"))?;
            let outcome = recurrence::solve(
                rhs,
                initial,
                recurrence::GmresParameters {
                    restart,
                    max_outer,
                    tolerance,
                },
                &mut work,
                &mut operations,
            )
            .await
            .map_err(recurrence::into_runtime_error)?;
            let solution = if outcome.flag == 0 {
                &*work.x
            } else {
                &*work.best
            };
            recurrence::GmresOperations::checkpoint(&mut operations)?;
            publish_unbounded::<T>(context, solution, work.history, outcome, requested).await
        }
    }
    .await
    .map_err(|mut error: RuntimeError| {
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })
}
