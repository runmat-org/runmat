//! Authentic gather and the existing cone solver floating-boundary policy.
use super::*;
use conic::matrix::IntegerPolicy;

#[derive(Clone, Copy)]
pub(super) enum FiniteMode {
    Finite,
    Bounds,
}

pub(super) async fn gather(value: Value, cooperation: &ScalarCooperation) -> BuiltinResult<Value> {
    crate::dispatcher::gather_if_needed_in_context_async(&value, cooperation.runtime())
        .await
        .map_err(|source| {
            if crate::runtime_error::error_has_control_cancellation_origin(&source) {
                return source;
            }
            let mut error = coneprog_error(&ERROR_INVALID_INPUT, source.message());
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error.failure_origin = source.failure_origin;
            error.source = Some(Box::new(source));
            error
        })
}

pub(super) async fn vector(
    label: &str,
    value: Value,
    mode: FiniteMode,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let value = gather(value, cooperation).await?;
    if is_empty_value(&value) {
        return Ok(Vec::new());
    }
    let data = match value {
        Value::Num(value) => vec![value],
        Value::Int(value) => vec![value.to_f64()],
        Value::Bool(value) => vec![if value { 1.0 } else { 0.0 }],
        Value::Tensor(value)
            if value.shape().len() <= 2 && (value.rows() == 1 || value.cols() == 1) =>
        {
            tensor::tensor_into_values_f64(value)
        }
        _ => return Err(invalid(label, "must be a real numeric vector")),
    };
    validate(label, &data, mode, cooperation).await?;
    Ok(data)
}
pub(super) async fn scalar(
    label: &str,
    value: Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<f64> {
    let values = vector(label, value, FiniteMode::Finite, cooperation).await?;
    if values.len() != 1 {
        return Err(invalid(label, "must be a scalar"));
    }
    Ok(values[0])
}
pub(super) async fn matrix(
    label: &str,
    value: Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<MatrixInput>> {
    conic::matrix::parse(
        gather(value, cooperation).await?,
        IntegerPolicy::FloatingBoundary,
        cooperation,
        |message| invalid(label, message),
    )
    .await
}

/// Preserve descriptor storage geometry: dense input yields dense double A;
/// sparse input yields sparse double A, never a hidden dense materialization.
pub(super) async fn descriptor_matrix(
    value: Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<(Value, usize, usize)> {
    let value = gather(value, cooperation).await?;
    if matches!(&value, Value::SparseTensor(_)) {
        let matrix = conic::matrix::parse(
            value,
            IntegerPolicy::FloatingBoundary,
            cooperation,
            |message| invalid("A", message),
        )
        .await?
        .ok_or_else(|| invalid("A", "must be nonempty"))?;
        let (rows, cols) = (matrix.m, matrix.n);
        let value =
            runmat_value::SparseTensor::new(rows, cols, matrix.colptr, matrix.rowval, matrix.nzval)
                .map_err(|message| invalid("A", &message))?;
        return Ok((Value::SparseTensor(value), rows, cols));
    }
    let (rows, cols, data) = match value {
        Value::Num(value) => (1, 1, vec![value]),
        Value::Int(value) => (1, 1, vec![value.to_f64()]),
        Value::Tensor(value) if value.shape().len() <= 2 => {
            let (rows, cols) = (value.rows(), value.cols());
            (rows, cols, tensor::tensor_into_values_f64(value))
        }
        _ => return Err(invalid("A", "must be a real numeric matrix")),
    };
    if rows == 0 || cols == 0 {
        return Err(invalid("A", "must be nonempty"));
    }
    validate("A", &data, FiniteMode::Finite, cooperation).await?;
    let value = Tensor::new(data, vec![rows, cols])
        .map(Value::Tensor)
        .map_err(|message| invalid("A", &message))?;
    Ok((value, rows, cols))
}

async fn validate(
    label: &str,
    values: &[f64],
    mode: FiniteMode,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    for chunk in values.chunks(1024) {
        for value in chunk {
            if match mode {
                FiniteMode::Finite => !value.is_finite(),
                FiniteMode::Bounds => value.is_nan(),
            } {
                return Err(invalid(
                    label,
                    match mode {
                        FiniteMode::Finite => "values must be finite",
                        FiniteMode::Bounds => "bounds cannot be NaN",
                    },
                ));
            }
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(())
}
fn invalid(label: &str, detail: &str) -> RuntimeError {
    coneprog_error(&ERROR_INVALID_INPUT, format!("{label} {detail}"))
}
