//! Controllability matrix for one native-precision state-space pair.
//!
//! The `A` product uses the existing dense linear-operator implementation;
//! this leaf owns only the source-ordered `[B, AB, ..., A^(n-1)B]` assembly.

use runmat_macros::runtime_builtin;
use runmat_value::{ComplexElement, ObjectInstance, Value};

use crate::builtins::control::tf_model::{control_error, SS_CLASS};
use crate::builtins::math::linalg::ops::linear_operator::{
    DenseLinearOperator, DenseOperator, LinearOperatorError,
};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{dispatcher, BuiltinResult, RuntimeError};

const NAME: &str = "ctrb";

mod output;

fn argument(message: impl Into<String>) -> RuntimeError {
    control_error(NAME, "RunMat:ctrb:InvalidArgument", message)
}

fn dimensions(message: impl Into<String>) -> RuntimeError {
    control_error(NAME, "RunMat:ctrb:InvalidDimensions", message)
}

fn unsupported(message: impl Into<String>) -> RuntimeError {
    control_error(NAME, "RunMat:ctrb:UnsupportedInput", message)
}

fn admission(runtime: Option<&RuntimeContext>, error: HostAllocationError) -> RuntimeError {
    let cancellation = runtime.map(RuntimeContext::cancellation_context);
    crate::builtin::builtin_control_error(cancellation.as_ref(), NAME, error)
}

fn check_runtime(runtime: Option<&RuntimeContext>) -> BuiltinResult<()> {
    HostAllocationReceipt::policy(runtime)
        .map(|_| ())
        .map_err(|error| admission(runtime, error))
}

#[runtime_builtin(
    name = "ctrb",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::control::ctrb"
)]
async fn ctrb_builtin(
    runtime: Option<RuntimeContext>,
    first: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(argument("ctrb returns one controllability matrix"));
    }
    let (a, b) = match first {
        Value::Object(object) if object.is_class(SS_CLASS) && rest.is_empty() => ss_pair(object)?,
        Value::Object(_) => return Err(argument("ctrb(sys) requires one state-space ss model")),
        a if rest.len() == 1 => (a, rest.into_iter().next().expect("one B argument")),
        _ => return Err(argument("ctrb expects (A,B) or (sys)")),
    };
    check_runtime(runtime.as_ref())?;
    // Validate both operands before a provider effect. Ordinary host calls
    // validate once below, without rescanning their complete payloads here.
    if matches!(a, Value::GpuTensor(_)) || matches!(b, Value::GpuTensor(_)) {
        preflight_matrix(&a, runtime.as_ref())?;
        preflight_matrix(&b, runtime.as_ref())?;
    }
    let a = materialize_matrix(a, runtime.as_ref()).await?;
    let b = materialize_matrix(b, runtime.as_ref()).await?;
    controllability(&a, &b, runtime.as_ref())
}

fn ss_pair(object: ObjectInstance) -> BuiltinResult<(Value, Value)> {
    // Resolve both consumed properties before any provider gather. Other ss
    // metadata is irrelevant to the A/B controllability identity.
    let b = object
        .property("B")
        .cloned()
        .ok_or_else(|| argument("ss model is missing B"))?;
    let a = object
        .property("A")
        .cloned()
        .ok_or_else(|| argument("ss model is missing A"))?;
    Ok((a, b))
}

async fn materialize_matrix(
    value: Value,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Value> {
    if matches!(value, Value::GpuTensor(_))
        && HostAllocationReceipt::policy(runtime).map_err(|error| admission(runtime, error))?
            == HostAllocationPolicy::Managed
    {
        return Err(admission(runtime, HostAllocationError::MissingManagedOwner));
    }
    let gathered = if matches!(value, Value::GpuTensor(_)) {
        dispatcher::gather_if_needed_async(&value).await?
    } else {
        value
    };
    match gathered {
        Value::Tensor(_) | Value::ComplexTensor(_) => Ok(gathered),
        Value::Num(number) if number.is_finite() => Ok(gathered),
        Value::Complex(real, imag) if real.is_finite() && imag.is_finite() => Ok(gathered),
        _ => Err(unsupported(
            "A and B require finite real or complex single/double matrices",
        )),
    }
}

fn operator_error(error: LinearOperatorError, runtime: Option<&RuntimeContext>) -> RuntimeError {
    match error {
        LinearOperatorError::Admission(source) => admission(runtime, source),
        LinearOperatorError::DimensionMismatch | LinearOperatorError::Overflow => {
            dimensions(format!("ctrb matrix shape is invalid: {error:?}"))
        }
        _ => unsupported(format!(
            "ctrb requires native real or complex single/double storage: {error:?}"
        )),
    }
}

fn preflight_matrix(value: &Value, runtime: Option<&RuntimeContext>) -> BuiltinResult<()> {
    check_runtime(runtime)?;
    let (shape, finite): (&[usize], bool) = match value {
        Value::Tensor(tensor) => (
            tensor.shape(),
            if let Some(values) = tensor.as_f32_slice() {
                finite_chunks(values, runtime, f32::is_finite)?
            } else if let Some(values) = tensor.as_f64_slice() {
                finite_chunks(values, runtime, f64::is_finite)?
            } else {
                false
            },
        ),
        Value::ComplexTensor(tensor) => (
            tensor.shape(),
            if let Some(values) = tensor.as_f32_slice() {
                finite_chunks(values, runtime, |value| {
                    value.0.is_finite() && value.1.is_finite()
                })?
            } else if let Some(values) = tensor.as_f64_slice() {
                finite_chunks(values, runtime, |value| {
                    value.0.is_finite() && value.1.is_finite()
                })?
            } else {
                false
            },
        ),
        Value::Num(value) => (&[1, 1], value.is_finite()),
        Value::Complex(real, imag) => (&[1, 1], real.is_finite() && imag.is_finite()),
        Value::GpuTensor(_) => return Ok(()),
        _ => {
            return Err(unsupported(
                "A and B require native real or complex single/double matrices",
            ))
        }
    };
    if shape.iter().skip(2).any(|extent| *extent != 1) {
        return Err(dimensions("ctrb requires a 2-D matrix"));
    }
    if !finite {
        return Err(unsupported("ctrb inputs require finite floating values"));
    }
    check_runtime(runtime)
}

fn finite_chunks<T: Copy>(
    values: &[T],
    runtime: Option<&RuntimeContext>,
    finite: impl Fn(T) -> bool,
) -> BuiltinResult<bool> {
    for chunk in values.chunks(4096) {
        check_runtime(runtime)?;
        if !chunk.iter().copied().all(&finite) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn borrow_matrix<'a>(
    value: &'a Value,
    real_scalar: &'a [f64; 1],
    complex_scalar: &'a [ComplexElement<f64>; 1],
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<DenseLinearOperator<'a>> {
    let operator = match value {
        Value::Tensor(tensor) => match runtime {
            Some(runtime) => DenseLinearOperator::borrow_real_with_context(tensor, runtime),
            None => DenseLinearOperator::borrow_real_unbounded(tensor),
        },
        Value::ComplexTensor(tensor) => match runtime {
            Some(runtime) => DenseLinearOperator::borrow_complex_with_context(tensor, runtime),
            None => DenseLinearOperator::borrow_complex_unbounded(tensor),
        },
        Value::Num(_) => {
            DenseOperator::from_column_major(real_scalar, 1, 1).map(DenseLinearOperator::RealF64)
        }
        Value::Complex(_, _) => DenseOperator::from_column_major(complex_scalar, 1, 1)
            .map(DenseLinearOperator::ComplexF64),
        _ => return Err(unsupported("expected a native floating matrix")),
    };
    operator.map_err(|error| operator_error(error, runtime))
}

fn controllability(a: &Value, b: &Value, runtime: Option<&RuntimeContext>) -> BuiltinResult<Value> {
    preflight_matrix(a, runtime)?;
    preflight_matrix(b, runtime)?;
    let real_a = [if let Value::Num(value) = a {
        *value
    } else {
        0.0
    }];
    let real_b = [if let Value::Num(value) = b {
        *value
    } else {
        0.0
    }];
    let complex_a = [if let Value::Complex(re, im) = a {
        ComplexElement(*re, *im)
    } else {
        ComplexElement(0.0, 0.0)
    }];
    let complex_b = [if let Value::Complex(re, im) = b {
        ComplexElement(*re, *im)
    } else {
        ComplexElement(0.0, 0.0)
    }];
    let a = borrow_matrix(a, &real_a, &complex_a, runtime)?;
    let b = borrow_matrix(b, &real_b, &complex_b, runtime)?;
    let n = a.rows();
    let m = b.cols();
    if n == 0 || a.cols() != n || b.rows() != n || m == 0 {
        return Err(dimensions(format!(
            "A must be nonempty n-by-n and B n-by-m with m>0; got A {}x{}, B {}x{}",
            a.rows(),
            a.cols(),
            b.rows(),
            m,
        )));
    }
    let columns = n
        .checked_mul(m)
        .ok_or_else(|| dimensions("controllability width overflow"))?;
    n.checked_mul(columns)
        .ok_or_else(|| dimensions("controllability extent overflow"))?;

    macro_rules! assemble {
        ($operator:expr, $input:expr, $storage:ident) => {{
            let source = $input.as_slice();
            let output::OutputOwner { mut value, receipt } =
                $storage::allocate(n, columns, runtime)?;
            let values = $storage::values_mut(&mut value)?;
            for (target, source) in values[..columns].chunks_mut(4096).zip(source.chunks(4096)) {
                output::checkpoint(runtime, receipt.as_ref())?;
                target.copy_from_slice(source);
            }
            for power in 1..n {
                output::checkpoint(runtime, receipt.as_ref())?;
                let split = power * columns;
                let (previous, next) = values.split_at_mut(split);
                let previous = &previous[split - columns..split];
                for column in 0..m {
                    output::checkpoint(runtime, receipt.as_ref())?;
                    let start = column * n;
                    let input = &previous[start..start + n];
                    let target = &mut next[start..start + n];
                    if let Some(runtime) = runtime {
                        $operator
                            .apply_into(input, target, runtime)
                            .map_err(|error| operator_error(error, Some(runtime)))?;
                    } else {
                        $operator
                            .apply_into_unbounded(input, target)
                            .map_err(|error| operator_error(error, None))?;
                    }
                }
                if !finite_chunks(&next[..columns], runtime, $storage::finite)? {
                    return Err(unsupported(
                        "controllability product exceeds finite numeric range",
                    ));
                }
            }
            output::checkpoint(runtime, receipt.as_ref())?;
            Ok(value)
        }};
    }

    use output::{ComplexF32, ComplexF64, RealF32, RealF64};
    match (a, b) {
        (DenseLinearOperator::RealF32(a), DenseLinearOperator::RealF32(b)) => {
            assemble!(a, b, RealF32)
        }
        (DenseLinearOperator::RealF64(a), DenseLinearOperator::RealF64(b)) => {
            assemble!(a, b, RealF64)
        }
        (DenseLinearOperator::ComplexF32(a), DenseLinearOperator::ComplexF32(b)) => {
            assemble!(a, b, ComplexF32)
        }
        (DenseLinearOperator::ComplexF64(a), DenseLinearOperator::ComplexF64(b)) => {
            assemble!(a, b, ComplexF64)
        }
        _ => Err(unsupported(
            "A and B must have the same real/complex precision",
        )),
    }
}

#[cfg(test)]
mod tests;
