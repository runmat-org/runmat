//! Host-visible matrix and response admission before provider materialization.
use super::{
    admitted_host_predictor, ensure_exact_integer_tensor, exact_integer_as_f64, gather_value,
    invalid, numeric_tensor, response_outcomes, FitCooperation, ResponseExactness,
    ResponseOutcomes,
};
use crate::BuiltinResult;
use runmat_accelerate_api::GpuTensorStorage;
use runmat_value::{Tensor, Value};

pub(super) async fn preflight(
    x: &Value,
    y: &Value,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<usize> {
    let rows = predictor_rows(x, cooperation).await?;
    response_geometry(y, rows, cooperation).await?;
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

pub(super) enum PreparedPredictor {
    Host(Value),
    Resident(Value),
}

pub(super) struct PreparedAdmission {
    predictor: PreparedPredictor,
    response: PreparedResponse,
}

impl PreparedAdmission {
    pub(super) fn rows(&self) -> usize {
        self.response.rows()
    }

    pub(super) fn into_parts(self) -> (PreparedPredictor, PreparedResponse) {
        (self.predictor, self.response)
    }
}

pub(super) async fn prepare(
    x: Value,
    y: Value,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<PreparedAdmission> {
    let rows = preflight(&x, &y, cooperation).await?;
    let predictor = if crate::dispatcher::value_contains_gpu(&x) {
        PreparedPredictor::Resident(x)
    } else {
        PreparedPredictor::Host(x)
    };
    let response = if crate::dispatcher::value_contains_gpu(&y) {
        PreparedResponse::Resident { value: y, rows }
    } else {
        PreparedResponse::Host {
            response: response_outcomes(y, rows, ResponseExactness::PrevalidatedHost, cooperation)
                .await?,
            rows,
        }
    };
    Ok(PreparedAdmission {
        predictor,
        response,
    })
}

impl PreparedPredictor {
    pub(super) async fn materialize(
        self,
        cooperation: &mut FitCooperation,
    ) -> BuiltinResult<Tensor> {
        match self {
            Self::Host(value) => admitted_host_predictor(value, cooperation).await,
            Self::Resident(value) => {
                numeric_tensor("X", gather_value(value).await?, cooperation).await
            }
        }
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
                response_outcomes(
                    gather_value(value).await?,
                    rows,
                    ResponseExactness::ValidateDownloaded,
                    cooperation,
                )
                .await
            }
        }
    }
}

async fn predictor_rows(x: &Value, cooperation: &mut FitCooperation) -> BuiltinResult<usize> {
    let shape = match x {
        Value::Tensor(tensor) => {
            ensure_exact_integer_tensor(tensor, "X", cooperation).await?;
            tensor.shape()
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

async fn response_geometry(
    y: &Value,
    rows: usize,
    cooperation: &mut FitCooperation,
) -> BuiltinResult<()> {
    match y {
        Value::Tensor(tensor) => {
            ensure_exact_integer_tensor(tensor, "Y", cooperation).await?;
            numeric_response_shape(tensor.shape(), tensor.len(), rows)
        }
        Value::GpuTensor(handle) if handle.physical().storage() == GpuTensorStorage::Real => {
            let shape = handle.shape();
            let count = checked_count(shape)?;
            numeric_response_shape(shape, count, rows)
        }
        Value::LogicalArray(array) => vector_response_shape(&array.shape, array.data.len(), rows),
        Value::StringArray(array) => vector_response_shape(&array.shape, array.data().len(), rows),
        Value::Cell(array) => {
            vector_response_shape(&array.shape, array.data().len(), rows)?;
            for (index, value) in array.data().iter().enumerate() {
                let valid_text = match value {
                    Value::String(_) => true,
                    Value::CharArray(chars) => chars.rows == 1,
                    _ => false,
                };
                if !valid_text {
                    return Err(invalid("mnrfit: Y cell entries must be text"));
                }
                if (index + 1).is_multiple_of(16_384) {
                    cooperation.completed(16_384).await?;
                }
            }
            cooperation.completed((array.data().len() % 16_384) as u64).await?;
            Ok(())
        }
        Value::CharArray(array) if array.rows == rows => Ok(()),
        Value::String(runmat_value::StringElement::Text(value)) if rows == 1 && !value.is_empty() => Ok(()),
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
