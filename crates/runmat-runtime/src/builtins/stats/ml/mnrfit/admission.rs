//! Host-visible matrix and response admission before provider materialization.
use super::{
    ensure_exact_integer_tensor, exact_integer_as_f64, gather_value, invalid, response_outcomes,
    scalar_text, FitCooperation, ResponseOutcomes,
};
use crate::BuiltinResult;
use runmat_accelerate_api::GpuTensorStorage;
use runmat_value::Value;

pub(super) fn preflight(x: &Value, y: &Value) -> BuiltinResult<usize> {
    let rows = predictor_rows(x)?;
    response_geometry(y, rows)?;
    Ok(rows)
}

pub(super) enum PreparedResponse {
    Host {
        response: ResponseOutcomes,
        rows: usize,
    },
    Resident {
        value: Value,
        rows: usize,
    },
}

pub(super) async fn prepare(
    x: &Value,
    y: Value,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<PreparedResponse> {
    let rows = preflight(x, &y)?;
    if crate::dispatcher::value_contains_gpu(&y) {
        Ok(PreparedResponse::Resident { value: y, rows })
    } else {
        Ok(PreparedResponse::Host {
            response: response_outcomes(y, rows, cooperation).await?,
            rows,
        })
    }
}

impl PreparedResponse {
    pub(super) fn rows(&self) -> usize {
        match self {
            Self::Host { rows, .. } | Self::Resident { rows, .. } => *rows,
        }
    }

    pub(super) async fn materialize(
        self,
        cooperation: &mut FitCooperation,
    ) -> BuiltinResult<ResponseOutcomes> {
        match self {
            Self::Host { response, .. } => Ok(response),
            Self::Resident { value, rows } => {
                response_outcomes(gather_value(value).await?, rows, cooperation).await
            }
        }
    }
}

fn predictor_rows(x: &Value) -> BuiltinResult<usize> {
    let shape = match x {
        Value::Tensor(tensor) => {
            ensure_exact_integer_tensor(tensor, "X")?;
            tensor.shape.as_slice()
        }
        Value::LogicalArray(array) => array.shape.as_slice(),
        Value::GpuTensor(handle) if handle.physical().storage() == GpuTensorStorage::Real => {
            handle.shape()
        }
        Value::Num(_) | Value::Bool(_) => return Ok(1),
        Value::Int(integer) => {
            exact_integer_as_f64(integer, "X")?;
            return Ok(1);
        }
        _ => return Err(invalid("mnrfit: X must be numeric")),
    };
    let rows = shape.first().copied().unwrap_or(1);
    let cols = shape.get(1).copied().unwrap_or(1);
    if rows == 0 || cols == 0 || !matrix_extent_matches(shape, rows, cols) {
        return Err(invalid("mnrfit: X must be a nonempty numeric matrix"));
    }
    Ok(rows)
}

fn response_geometry(y: &Value, rows: usize) -> BuiltinResult<()> {
    match y {
        Value::Tensor(tensor) => {
            ensure_exact_integer_tensor(tensor, "Y")?;
            numeric_response_shape(&tensor.shape, tensor.len(), rows)
        }
        Value::GpuTensor(handle) if handle.physical().storage() == GpuTensorStorage::Real => {
            let shape = handle.shape();
            let count = checked_count(shape)?;
            numeric_response_shape(shape, count, rows)
        }
        Value::LogicalArray(array) => vector_response_shape(&array.shape, array.data.len(), rows),
        Value::StringArray(array) => vector_response_shape(&array.shape, array.data.len(), rows),
        Value::Cell(array) => {
            vector_response_shape(&array.shape, array.data.len(), rows)?;
            if array.data.iter().any(|value| scalar_text(value).is_none()) {
                return Err(invalid("mnrfit: Y cell entries must be text"));
            }
            Ok(())
        }
        Value::CharArray(array) if array.rows == rows => Ok(()),
        Value::String(value) if rows == 1 && !value.is_empty() => Ok(()),
        Value::Num(_) | Value::Bool(_) if rows == 1 => Ok(()),
        Value::Object(object) if object.is_class(runmat_types::standard::CATEGORICAL) => Ok(()),
        _ => Err(invalid(
            "mnrfit: Y must be a numeric, logical, text, categorical, cellstr, or grouped counts response",
        )),
    }
}

fn numeric_response_shape(shape: &[usize], count: usize, rows: usize) -> BuiltinResult<()> {
    let matrix_rows = shape.first().copied().unwrap_or(1);
    let matrix_cols = shape.get(1).copied().unwrap_or(1);
    if matrix_rows == rows
        && matrix_cols >= 2
        && matrix_extent_matches(shape, matrix_rows, matrix_cols)
    {
        return Ok(());
    }
    vector_response_shape(shape, count, rows)
}

fn vector_response_shape(shape: &[usize], count: usize, rows: usize) -> BuiltinResult<()> {
    if count != rows || shape.iter().filter(|dimension| **dimension > 1).count() > 1 {
        return Err(invalid(
            "mnrfit: categorical Y must be a vector with one response per row of X",
        ));
    }
    Ok(())
}

fn matrix_extent_matches(shape: &[usize], rows: usize, cols: usize) -> bool {
    rows.checked_mul(cols).is_some_and(|count| {
        shape.iter().skip(2).all(|dimension| *dimension == 1)
            && checked_count(shape).is_ok_and(|actual| actual == count)
    })
}

fn checked_count(shape: &[usize]) -> BuiltinResult<usize> {
    shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| invalid("mnrfit: response shape exceeds platform limits"))
}
