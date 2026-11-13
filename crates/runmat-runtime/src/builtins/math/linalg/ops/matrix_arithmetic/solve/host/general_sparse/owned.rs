//! Square sparse owned LU: actual-policy RHS/output ownership.
//! Only a typed numerical singularity selects complete orthogonal solving.
use super::rectangular::{Element, Output};
use crate::builtins::math::linalg::ops::sparse_factor::{
    ManagedSparseLu, ManagedSparseLuError, SparseQrScalar,
};
use crate::context::{host_allocation::HostAllocationError, RuntimeContext};
use crate::{BuiltinResult, RuntimeError};
use runmat_builtins::{MLDIVIDE_ERROR_INTERNAL, MLDIVIDE_ERROR_INVALID_INPUT};
use runmat_value::{ComplexElement, ExactHostNumericAllocationError, SparseTensor, Value};

fn admission(error: HostAllocationError, context: &RuntimeContext) -> RuntimeError {
    let mut error = crate::builtin::builtin_control_error(
        Some(&context.cancellation_context()),
        "mldivide",
        error,
    );
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}
fn factor_error(error: ManagedSparseLuError, context: &RuntimeContext) -> RuntimeError {
    let mut result = match error {
        ManagedSparseLuError::Admission(error) => admission(error, context),
        ManagedSparseLuError::Execution(error) => {
            error.into_runtime_error_in_context(Some(context))
        }
        other => {
            let descriptor = match &other {
                ManagedSparseLuError::Library(_) => &MLDIVIDE_ERROR_INTERNAL,
                _ => &MLDIVIDE_ERROR_INVALID_INPUT,
            };
            let mut builder = crate::build_runtime_error(format!("mldivide: {other}"))
                .with_builtin("mldivide")
                .with_source(other);
            if let Some(identifier) = descriptor.identifier {
                builder = builder.with_identifier(identifier);
            }
            builder.build()
        }
    };
    result.gpu_gather_retry = crate::GpuGatherRetry::Never;
    result
}

async fn tensor<T: SparseQrScalar, S: Element>(
    sparse: &SparseTensor,
    input: &[S],
    columns: usize,
    context: &RuntimeContext,
    to_native: impl FnMut(S) -> T,
    mut from_native: impl FnMut(T) -> S,
) -> BuiltinResult<Value> {
    let attempt = ManagedSparseLu::<T>::factor(sparse, context).await;
    let mut factor = match attempt {
        Ok(factor) => factor,
        Err(ManagedSparseLuError::Singular { .. }) => {
            return super::rectangular::tensor::<T, S>(
                sparse,
                input,
                columns,
                context,
                to_native,
                from_native,
            )
            .await
        }
        Err(error) => return Err(factor_error(error, context)),
    };
    let mut output = Output::<S>::new(sparse.rows, columns, context, || {
        factor.check_owner(context)
    })
    .await?;
    factor
        .solve_with_output(input, columns, context, to_native, |index, value| {
            output
                .write(index, from_native(value))
                .map_err(ManagedSparseLuError::Admission)
        })
        .await
        .map_err(|e| factor_error(e, context))?;
    output.finish(context, || factor.check_owner(context))
}

/// Called only after the shared square/type/shape candidate check. Every error
/// is terminal except the explicit numerical-singularity QR transition.
pub(super) async fn solve(
    sparse: &SparseTensor,
    rhs: &Value,
    columns: usize,
    context: &RuntimeContext,
) -> BuiltinResult<Value> {
    let wrong_class = || {
        admission(
            HostAllocationError::Numeric(ExactHostNumericAllocationError::UnsupportedClass),
            context,
        )
    };
    match rhs {
        Value::Tensor(value) if sparse.as_f32_slice().is_some() => {
            tensor::<f32, f32>(
                sparse,
                value.as_f32_slice().ok_or_else(wrong_class)?,
                columns,
                context,
                std::convert::identity,
                std::convert::identity,
            )
            .await
        }
        Value::Tensor(value) => {
            tensor::<f64, f64>(
                sparse,
                value.as_f64_slice().ok_or_else(wrong_class)?,
                columns,
                context,
                std::convert::identity,
                std::convert::identity,
            )
            .await
        }
        Value::ComplexTensor(value) if sparse.as_complex_f32_slice().is_some() => {
            tensor::<faer::c32, ComplexElement<f32>>(
                sparse,
                value.as_f32_slice().ok_or_else(wrong_class)?,
                columns,
                context,
                |v| faer::c32::new(v.0, v.1),
                |v| ComplexElement(v.re, v.im),
            )
            .await
        }
        Value::ComplexTensor(value) => {
            tensor::<faer::c64, ComplexElement<f64>>(
                sparse,
                value.as_f64_slice().ok_or_else(wrong_class)?,
                columns,
                context,
                |v| faer::c64::new(v.0, v.1),
                |v| ComplexElement(v.re, v.im),
            )
            .await
        }
        Value::Num(input) => {
            let attempt = ManagedSparseLu::<f64>::factor(sparse, context).await;
            let mut factor = match attempt {
                Ok(factor) => factor,
                Err(ManagedSparseLuError::Singular { .. }) => {
                    return super::rectangular::solve(sparse, rhs, columns, context).await
                }
                Err(error) => return Err(factor_error(error, context)),
            };
            let mut output = [0.0];
            factor
                .solve_into(std::slice::from_ref(input), &mut output, 1, context)
                .await
                .map_err(|e| factor_error(e, context))?;
            Ok(Value::Num(output[0]))
        }
        Value::Complex(real, imag) => {
            let attempt = ManagedSparseLu::<faer::c64>::factor(sparse, context).await;
            let mut factor = match attempt {
                Ok(factor) => factor,
                Err(ManagedSparseLuError::Singular { .. }) => {
                    return super::rectangular::solve(sparse, rhs, columns, context).await
                }
                Err(error) => return Err(factor_error(error, context)),
            };
            let input = [faer::c64::new(*real, *imag)];
            let mut output = [faer::c64::default()];
            factor
                .solve_into(&input, &mut output, 1, context)
                .await
                .map_err(|e| factor_error(e, context))?;
            Ok(Value::Complex(output[0].re, output[0].im))
        }
        _ => Err(wrong_class()),
    }
}

#[cfg(test)]
mod tests;
