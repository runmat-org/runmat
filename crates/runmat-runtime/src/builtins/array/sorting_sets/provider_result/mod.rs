mod geometry;
#[cfg(test)]
mod tests;

use crate::{BuiltinResult, RuntimeError};
use geometry::*;
use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, HostTensorOwned, IsMemberResult, ProviderExecutionError,
    ProviderExecutionFuture, ProviderPrecision, SetdiffResult, SortResult, UnionResult,
    UniqueResult,
};
use runmat_value::{LogicalArray, NumericDType, Tensor};

type Error = fn(String) -> RuntimeError;

pub(super) async fn attempt<T>(
    pending: ProviderExecutionFuture<'_, Option<T>>,
    error: Error,
) -> BuiltinResult<Option<T>> {
    pending.await.map_err(|source| {
        crate::runtime_error::with_provider_execution_source(
            error(format!("provider execution failed: {source}")),
            source.with_unproven_effects(),
        )
    })
}

pub(super) fn plain_real(input: &GpuTensorHandle) -> bool {
    runmat_accelerate_api::handle_integer_type(input).is_none()
        && !runmat_accelerate_api::handle_is_logical(input)
        && runmat_accelerate_api::handle_storage(input) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(input).is_some()
}

pub(super) fn real_pair(a: &GpuTensorHandle, b: &GpuTensorHandle) -> bool {
    plain_real(a)
        && plain_real(b)
        && runmat_accelerate_api::handle_precision(a) == runmat_accelerate_api::handle_precision(b)
        && match (
            runmat_accelerate_api::provider_for_handle(a),
            runmat_accelerate_api::provider_for_handle(b),
        ) {
            (Some(left), Some(right)) => std::ptr::eq(left, right),
            _ => false,
        }
}

fn dtype(input: &GpuTensorHandle) -> NumericDType {
    if runmat_accelerate_api::handle_precision(input) == Some(ProviderPrecision::F32) {
        NumericDType::F32
    } else {
        NumericDType::F64
    }
}

fn values(result: HostTensorOwned, dtype: NumericDType, error: Error) -> BuiltinResult<Tensor> {
    real(&result, error)?;
    if dtype == NumericDType::F32
        && result
            .data
            .iter()
            .any(|value| !value.is_nan() && (*value as f32) as f64 != *value)
    {
        return Err(invalid(
            error,
            "provider values are not representable in the input single class",
        ));
    }
    Tensor::new_with_dtype(result.data, result.shape, dtype)
        .map_err(|detail| invalid(error, detail))
}

pub(super) enum SortGeometry {
    Dimension(usize),
    Rows,
}

pub(super) fn sort(
    mut result: SortResult,
    input: &GpuTensorHandle,
    geometry: SortGeometry,
    error: Error,
) -> BuiltinResult<(Tensor, Tensor)> {
    real(&result.values, error)?;
    real(&result.indices, error)?;
    if result.values.shape != input.shape() {
        return Err(invalid(error, "sort values changed input geometry"));
    }
    match geometry {
        SortGeometry::Dimension(axis) => {
            if result.indices.shape != input.shape() {
                return Err(invalid(error, "sort indices changed input geometry"));
            }
            permutation(&mut result.indices.data, input.shape(), axis, error)?;
        }
        SortGeometry::Rows => {
            let rows = input.shape().first().copied().unwrap_or(1);
            vector(&result.indices, Some(rows), error)?;
            permutation(&mut result.indices.data, &[rows, 1], 0, error)?;
        }
    }
    Ok((
        values(result.values, dtype(input), error)?,
        values(result.indices, NumericDType::F64, error)?,
    ))
}

pub(super) fn unique(
    result: UniqueResult,
    input: &GpuTensorHandle,
    rows: bool,
    error: Error,
) -> BuiltinResult<(Tensor, Tensor, Tensor)> {
    if runmat_accelerate_api::handle_is_logical(input)
        && result
            .values
            .data
            .iter()
            .any(|&value| value != 0.0 && value != 1.0)
    {
        return Err(invalid(
            error,
            "logical unique values must be exactly zero or one",
        ));
    }
    let input_count = records(input.shape(), rows, error)?;
    let count = set_shape(
        &result.values,
        input.shape(),
        rows,
        input_count,
        !rows && row_vector(input.shape()),
        error,
    )?;
    vector(&result.ia, Some(count), error)?;
    range(&result.ia.data, input_count, false, error)?;
    vector(&result.ic, Some(input_count), error)?;
    range(&result.ic.data, count, false, error)?;
    for (position, &original) in result.ia.data.iter().enumerate() {
        // Both columns are range-checked above, so these are exact native
        // positions. Each representative must map back to its own result.
        if result.ic.data.get(original as usize - 1) != Some(&((position + 1) as f64)) {
            return Err(invalid(
                error,
                "unique representative and inverse indices disagree",
            ));
        }
    }
    Ok((
        values(result.values, dtype(input), error)?,
        values(result.ia, NumericDType::F64, error)?,
        values(result.ic, NumericDType::F64, error)?,
    ))
}

pub(super) fn union(
    result: UnionResult,
    a: &GpuTensorHandle,
    b: &GpuTensorHandle,
    rows: bool,
    error: Error,
) -> BuiltinResult<(Tensor, Tensor, Tensor)> {
    let na = records(a.shape(), rows, error)?;
    let nb = records(b.shape(), rows, error)?;
    let maximum = na
        .checked_add(nb)
        .ok_or_else(|| invalid(error, "union cardinality overflow"))?;
    let row = !rows && super::set_policy::ElementOrientation::BothRow.row(a.shape(), b.shape());
    let count = set_shape(&result.values, a.shape(), rows, maximum, row, error)?;
    vector(&result.ia, None, error)?;
    vector(&result.ib, None, error)?;
    if result.ia.data.len().checked_add(result.ib.data.len()) != Some(count) {
        return Err(invalid(
            error,
            "union contribution index lengths differ from values",
        ));
    }
    range(&result.ia.data, na, false, error)?;
    range(&result.ib.data, nb, false, error)?;
    Ok((
        values(result.values, dtype(a), error)?,
        values(result.ia, NumericDType::F64, error)?,
        values(result.ib, NumericDType::F64, error)?,
    ))
}

pub(super) fn setdiff(
    result: SetdiffResult,
    a: &GpuTensorHandle,
    rows: bool,
    error: Error,
) -> BuiltinResult<(Tensor, Tensor)> {
    let na = records(a.shape(), rows, error)?;
    let row = !rows && super::set_policy::ElementOrientation::ARow.row(a.shape(), a.shape());
    let count = set_shape(&result.values, a.shape(), rows, na, row, error)?;
    vector(&result.ia, Some(count), error)?;
    range(&result.ia.data, na, false, error)?;
    Ok((
        values(result.values, dtype(a), error)?,
        values(result.ia, NumericDType::F64, error)?,
    ))
}

pub(super) fn membership(
    result: IsMemberResult,
    a: &GpuTensorHandle,
    b: &GpuTensorHandle,
    rows: bool,
    error: Error,
) -> BuiltinResult<(LogicalArray, Tensor)> {
    let na = records(a.shape(), rows, error)?;
    let nb = records(b.shape(), rows, error)?;
    let row_shape = [na, 1];
    let shape: &[usize] = if rows { &row_shape } else { a.shape() };
    real(&result.loc, error)?;
    if result.mask.shape != shape
        || result.loc.shape != shape
        || result.mask.data.len() != na
        || result.loc.data.len() != na
    {
        return Err(invalid(error, "membership outputs changed input geometry"));
    }
    range(&result.loc.data, nb, true, error)?;
    if result
        .mask
        .data
        .iter()
        .zip(&result.loc.data)
        .any(|(flag, location)| *flag > 1 || (*flag == 0) != (*location == 0.0))
    {
        return Err(invalid(
            error,
            "membership logical flags and locations disagree",
        ));
    }
    let mask = LogicalArray::new(result.mask.data, result.mask.shape)
        .map_err(|detail| invalid(error, detail))?;
    Ok((mask, values(result.loc, NumericDType::F64, error)?))
}

#[derive(Debug)]
struct InvalidProviderResult(String);
impl std::fmt::Display for InvalidProviderResult {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}
impl std::error::Error for InvalidProviderResult {}
fn invalid(error: Error, detail: impl Into<String>) -> RuntimeError {
    let detail = detail.into();
    crate::runtime_error::with_provider_execution_source(
        error(detail.clone()),
        ProviderExecutionError::from_source(InvalidProviderResult(detail)).after_effect(),
    )
}
