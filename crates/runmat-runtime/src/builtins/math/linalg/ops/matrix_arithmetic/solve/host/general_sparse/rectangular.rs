//! Borrowed native sparse least squares through owned rank-revealing QR/COD.
mod output;
pub(super) use output::{Element, Output};

use super::rhs_columns;
use crate::builtins::math::linalg::ops::sparse_factor::{
    SparseCompleteQr, SparseCompleteQrError, SparseQrScalar,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};
use runmat_value::{ComplexElement, SparseTensor, Value};

fn map_qr(error: SparseCompleteQrError, context: &RuntimeContext) -> RuntimeError {
    let mut result = match error {
        SparseCompleteQrError::Admission(error) => crate::builtin::builtin_control_error(
            Some(&context.cancellation_context()),
            "mldivide",
            error,
        ),
        SparseCompleteQrError::Execution(error) => {
            error.into_runtime_error_in_context(Some(context))
        }
        other => {
            let descriptor = &runmat_builtins::MLDIVIDE_ERROR_INVALID_INPUT;
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

pub(super) async fn tensor<T: SparseQrScalar, S: Element>(
    sparse: &SparseTensor,
    input: &[S],
    columns: usize,
    context: &RuntimeContext,
    to_native: impl FnMut(S) -> T,
    mut from_native: impl FnMut(T) -> S,
) -> BuiltinResult<Value> {
    let factor = SparseCompleteQr::<T>::factor(sparse, context)
        .await
        .map_err(|e| map_qr(e, context))?;
    let mut output = Output::<S>::new(sparse.cols, columns, context, || {
        factor.check_owner(context)
    })
    .await?;
    factor
        .solve_with_output(input, columns, context, to_native, |index, value| {
            output
                .write(index, from_native(value))
                .map_err(SparseCompleteQrError::Admission)
        })
        .await
        .map_err(|e| map_qr(e, context))?;
    output.finish(context, || factor.check_owner(context))
}

/// This is also the explicit numerical-singularity transition destination.
/// Its caller has already proved the native matrix/RHS class and geometry.
pub(in crate::builtins::math::linalg::ops::matrix_arithmetic::solve::host) async fn solve(
    sparse: &SparseTensor,
    rhs: &Value,
    columns: usize,
    context: &RuntimeContext,
) -> BuiltinResult<Value> {
    match rhs {
        Value::Tensor(t) if sparse.as_f32_slice().is_some() => {
            tensor::<f32, f32>(
                sparse,
                t.as_f32_slice()
                    .ok_or_else(|| map_qr(SparseCompleteQrError::UnsupportedClass, context))?,
                columns,
                context,
                std::convert::identity,
                std::convert::identity,
            )
            .await
        }
        Value::Tensor(t) => {
            tensor::<f64, f64>(
                sparse,
                t.as_f64_slice()
                    .ok_or_else(|| map_qr(SparseCompleteQrError::UnsupportedClass, context))?,
                columns,
                context,
                std::convert::identity,
                std::convert::identity,
            )
            .await
        }
        Value::ComplexTensor(t) if sparse.as_complex_f32_slice().is_some() => {
            tensor::<faer::c32, ComplexElement<f32>>(
                sparse,
                t.as_f32_slice()
                    .ok_or_else(|| map_qr(SparseCompleteQrError::UnsupportedClass, context))?,
                columns,
                context,
                |v| faer::c32::new(v.0, v.1),
                |v| ComplexElement(v.re, v.im),
            )
            .await
        }
        Value::ComplexTensor(t) => {
            tensor::<faer::c64, ComplexElement<f64>>(
                sparse,
                t.as_f64_slice()
                    .ok_or_else(|| map_qr(SparseCompleteQrError::UnsupportedClass, context))?,
                columns,
                context,
                |v| faer::c64::new(v.0, v.1),
                |v| ComplexElement(v.re, v.im),
            )
            .await
        }
        Value::Num(value) => {
            tensor::<f64, f64>(
                sparse,
                std::slice::from_ref(value),
                columns,
                context,
                std::convert::identity,
                std::convert::identity,
            )
            .await
        }
        Value::Complex(real, imag) => {
            let scalar = ComplexElement(*real, *imag);
            tensor::<faer::c64, ComplexElement<f64>>(
                sparse,
                std::slice::from_ref(&scalar),
                columns,
                context,
                |v| faer::c64::new(v.0, v.1),
                |v| ComplexElement(v.re, v.im),
            )
            .await
        }
        _ => Err(map_qr(SparseCompleteQrError::UnsupportedClass, context)),
    }
}

pub(in super::super::super) async fn try_solve(
    lhs: &Value,
    rhs: &Value,
    context: &RuntimeContext,
) -> BuiltinResult<Option<Value>> {
    let Value::SparseTensor(sparse) = lhs else {
        return Ok(None);
    };
    // Nonsingular square matrices retain triangular/LU selection. A known
    // numerical singularity calls solve directly rather than redispatching.
    if sparse.rows == sparse.cols && sparse.rows != 0 {
        return Ok(None);
    }
    let columns = match rhs {
        Value::Tensor(t) => rhs_columns(t.shape(), sparse.rows).filter(|_| {
            (sparse.as_f32_slice().is_some() && t.as_f32_slice().is_some())
                || (sparse.as_f64_slice().is_some() && t.as_f64_slice().is_some())
        }),
        Value::ComplexTensor(t) => rhs_columns(t.shape(), sparse.rows).filter(|_| {
            (sparse.as_complex_f32_slice().is_some() && t.as_f32_slice().is_some())
                || (sparse.as_complex_f64_slice().is_some() && t.as_f64_slice().is_some())
        }),
        Value::Num(_) if sparse.rows == 1 && sparse.as_f64_slice().is_some() => Some(1),
        Value::Complex(_, _) if sparse.rows == 1 && sparse.as_complex_f64_slice().is_some() => {
            Some(1)
        }
        _ => None,
    };
    let Some(columns) = columns else {
        return Ok(None);
    };
    solve(sparse, rhs, columns, context).await.map(Some)
}
