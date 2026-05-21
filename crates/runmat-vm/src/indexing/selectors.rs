use crate::indexing::plan::total_len_from_shape;
use crate::interpreter::errors::mex;
use runmat_builtins::Value;
use runmat_runtime::{
    builtins::common::shape::is_scalar_shape, dispatcher::gather_if_needed_async, RuntimeError,
};

pub type VmResult<T> = Result<T, RuntimeError>;

fn map_index_gather_error(err: impl std::fmt::Display) -> RuntimeError {
    mex(
        "AccelerationOperationFailed",
        &format!("index gather: {err}"),
    )
}

fn selector_mask_has_dim(mask: u32, dim: usize) -> bool {
    dim < u32::BITS as usize && (mask & (1u32 << dim)) != 0
}

#[derive(Clone, Debug)]
pub enum SliceSelector {
    Colon,
    Scalar(usize),
    Indices(Vec<usize>),
    LinearIndices {
        values: Vec<usize>,
        output_shape: Vec<usize>,
    },
}

fn exact_index_from_f64(value: f64) -> Option<i64> {
    if !value.is_finite() {
        return None;
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return None;
    }
    if rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
        return None;
    }
    Some(rounded as i64)
}

fn index_scalar_from_host_value(value: &Value) -> Option<i64> {
    match value {
        Value::Num(n) => exact_index_from_f64(*n),
        Value::Int(int_val) => Some(int_val.to_i64()),
        Value::Tensor(t) if t.data.len() == 1 && is_scalar_shape(&t.shape) => {
            exact_index_from_f64(t.data[0])
        }
        _ => None,
    }
}

pub async fn index_scalar_from_value(value: &Value) -> VmResult<Option<i64>> {
    if let Value::GpuTensor(handle) = value {
        let total = total_len_from_shape(&handle.shape);
        if total != 1 {
            return Ok(None);
        }
        let gathered = gather_if_needed_async(value).await?;
        return Ok(index_scalar_from_host_value(&gathered));
    }
    Ok(index_scalar_from_host_value(value))
}

pub async fn materialize_index_value(value: &Value) -> VmResult<Value> {
    if matches!(value, Value::GpuTensor(_)) {
        return gather_if_needed_async(value)
            .await
            .map_err(map_index_gather_error);
    }
    Ok(value.clone())
}

pub async fn indices_from_value_linear(value: &Value, total_len: usize) -> VmResult<Vec<usize>> {
    if let Value::Bool(b) = value {
        return Ok(if *b { vec![1] } else { Vec::new() });
    }
    if let Value::LogicalArray(la) = value {
        if la.data.len() == 1 && is_scalar_shape(&la.shape) {
            return Ok(if la.data[0] != 0 { vec![1] } else { Vec::new() });
        }
    }
    if let Some(idx_val) = index_scalar_from_value(value).await? {
        if idx_val < 1 || (idx_val as usize) > total_len {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        return Ok(vec![idx_val as usize]);
    }
    let materialized;
    let value = if matches!(value, Value::GpuTensor(_)) {
        materialized = materialize_index_value(value).await?;
        &materialized
    } else {
        value
    };
    match value {
        Value::Tensor(idx_t) => {
            let len = idx_t.shape.iter().product::<usize>();
            let mut indices = Vec::with_capacity(len);
            for &val in &idx_t.data {
                let idx = exact_index_from_f64(val).ok_or_else(|| {
                    mex(
                        "UnsupportedIndexType",
                        "Index values must be positive integers or logical values",
                    )
                })?;
                if idx < 1 || (idx as usize) > total_len {
                    return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                }
                indices.push(idx as usize);
            }
            Ok(indices)
        }
        Value::LogicalArray(la) => {
            if la.data.len() != total_len {
                return Err(mex(
                    "IndexShape",
                    "Logical mask length mismatch for linear indexing",
                ));
            }
            let mut indices = Vec::new();
            for (i, &b) in la.data.iter().enumerate() {
                if b != 0 {
                    indices.push(i + 1);
                }
            }
            Ok(indices)
        }
        _ => Err(mex(
            "UnsupportedIndexType",
            "Unsupported index type for linear indexing",
        )),
    }
}

pub async fn selector_from_value_dim(value: &Value, dim_len: usize) -> VmResult<SliceSelector> {
    if let Value::Bool(b) = value {
        if *b {
            return Ok(SliceSelector::Indices(vec![1]));
        }
        return Ok(SliceSelector::Indices(Vec::new()));
    }
    if let Value::LogicalArray(la) = value {
        if la.data.len() == 1 && is_scalar_shape(&la.shape) {
            if la.data[0] != 0 {
                return Ok(SliceSelector::Indices(vec![1]));
            }
            return Ok(SliceSelector::Indices(Vec::new()));
        }
    }
    if let Some(idx_val) = index_scalar_from_value(value).await? {
        if idx_val < 1 || (idx_val as usize) > dim_len {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        return Ok(SliceSelector::Scalar(idx_val as usize));
    }
    let materialized;
    let value = if matches!(value, Value::GpuTensor(_)) {
        materialized = materialize_index_value(value).await?;
        &materialized
    } else {
        value
    };
    match value {
        Value::Tensor(idx_t) => {
            let len = idx_t.shape.iter().product::<usize>();
            let mut indices = Vec::with_capacity(len);
            for &val in &idx_t.data {
                let idx = exact_index_from_f64(val).ok_or_else(|| {
                    mex(
                        "UnsupportedIndexType",
                        "Index values must be positive integers or logical values",
                    )
                })?;
                if idx < 1 || (idx as usize) > dim_len {
                    return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                }
                indices.push(idx as usize);
            }
            Ok(SliceSelector::Indices(indices))
        }
        Value::LogicalArray(la) => {
            if la.data.len() != dim_len {
                return Err(mex(
                    "IndexShape",
                    "Logical mask length mismatch for dimension",
                ));
            }
            let mut indices = Vec::new();
            for (i, &b) in la.data.iter().enumerate() {
                if b != 0 {
                    indices.push(i + 1);
                }
            }
            Ok(SliceSelector::Indices(indices))
        }
        _ => Err(mex(
            "UnsupportedIndexType",
            "Unsupported index type for slicing",
        )),
    }
}

pub async fn build_slice_selectors(
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    base_shape: &[usize],
) -> VmResult<Vec<SliceSelector>> {
    let mut selectors = Vec::with_capacity(dims);
    if dims == 1 {
        let total_len = total_len_from_shape(base_shape);
        if (colon_mask & 1u32) != 0 {
            selectors.push(SliceSelector::Colon);
            return Ok(selectors);
        }
        if (end_mask & 1u32) != 0 {
            selectors.push(SliceSelector::Scalar(total_len.max(1)));
            return Ok(selectors);
        }
        let value = numeric.first().ok_or_else(|| {
            mex(
                "MissingNumericIndex",
                "missing numeric index for linear slice",
            )
        })?;
        let materialized = materialize_index_value(value).await?;
        if let Value::Tensor(idx_t) = &materialized {
            let len = idx_t.shape.iter().product::<usize>();
            let mut indices = Vec::with_capacity(len);
            for &val in &idx_t.data {
                let idx = exact_index_from_f64(val).ok_or_else(|| {
                    mex(
                        "UnsupportedIndexType",
                        "Index values must be positive integers or logical values",
                    )
                })?;
                if idx < 1 || (idx as usize) > total_len {
                    return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                }
                indices.push(idx as usize);
            }
            selectors.push(SliceSelector::LinearIndices {
                values: indices,
                output_shape: idx_t.shape.clone(),
            });
        } else {
            let idxs = indices_from_value_linear(&materialized, total_len).await?;
            selectors.push(SliceSelector::Indices(idxs));
        }
        return Ok(selectors);
    }

    let mut numeric_iter = 0usize;
    for d in 0..dims {
        let is_colon = selector_mask_has_dim(colon_mask, d);
        if is_colon {
            selectors.push(SliceSelector::Colon);
            continue;
        }
        let dim_len = base_shape.get(d).copied().unwrap_or(1);
        let is_end = selector_mask_has_dim(end_mask, d);
        if is_end {
            selectors.push(SliceSelector::Scalar(dim_len));
            continue;
        }
        let value = numeric
            .get(numeric_iter)
            .ok_or_else(|| mex("MissingNumericIndex", "missing numeric index for slice"))?;
        numeric_iter += 1;
        selectors.push(selector_from_value_dim(value, dim_len).await?);
    }
    Ok(selectors)
}

pub async fn build_cell_scalar_selectors(raw_indices: &[Value]) -> VmResult<Vec<SliceSelector>> {
    let mut selectors = Vec::with_capacity(raw_indices.len());
    for value in raw_indices {
        let idx_val = index_scalar_from_value(value).await?.ok_or_else(|| {
            mex(
                "ScalarIndexRequired",
                "Cell indexing requires scalar numeric indices",
            )
        })?;
        if idx_val < 1 {
            return Err(mex("IndexOutOfBounds", "Index out of bounds"));
        }
        let idx =
            usize::try_from(idx_val).map_err(|_| mex("IndexOutOfBounds", "Index out of bounds"))?;
        selectors.push(SliceSelector::Scalar(idx));
    }
    Ok(selectors)
}

#[cfg(test)]
mod tests {
    use super::{
        build_cell_scalar_selectors, build_slice_selectors, indices_from_value_linear,
        selector_from_value_dim, SliceSelector,
    };
    use runmat_builtins::{Tensor, Value};

    #[test]
    fn selector_from_value_dim_rejects_fractional_numeric_indices() {
        let err =
            futures::executor::block_on(selector_from_value_dim(&Value::Num(2.5), 8)).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:UnsupportedIndexType"));
    }

    #[test]
    fn linear_indices_reject_fractional_tensor_indices() {
        let value = Value::Tensor(
            Tensor::new(vec![1.0, 2.5], vec![1, 2]).expect("fractional index tensor"),
        );
        let err = futures::executor::block_on(indices_from_value_linear(&value, 8)).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:UnsupportedIndexType"));
    }

    #[test]
    fn build_cell_scalar_selectors_rejects_zero_index() {
        let err = futures::executor::block_on(build_cell_scalar_selectors(&[Value::Num(0.0)]))
            .expect_err("zero cell scalar index should fail");
        assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));
    }

    #[test]
    fn build_cell_scalar_selectors_rejects_negative_index() {
        let err = futures::executor::block_on(build_cell_scalar_selectors(&[Value::Num(-2.0)]))
            .expect_err("negative cell scalar index should fail");
        assert_eq!(err.identifier(), Some("RunMat:IndexOutOfBounds"));
    }

    #[test]
    fn index_gather_error_maps_to_acceleration_identifier() {
        let err = super::map_index_gather_error("boom");
        assert_eq!(err.identifier(), Some("RunMat:AccelerationOperationFailed"));
        assert!(err.message().contains("index gather"));
        assert!(err.message().contains("boom"));
    }

    #[test]
    fn build_slice_selectors_supports_dims_beyond_mask_width() {
        let numeric: Vec<Value> = (0..31).map(|v| Value::Num((v + 1) as f64)).collect();
        let base_shape = vec![40usize; 33];
        let selectors = futures::executor::block_on(build_slice_selectors(
            33,
            0b1,
            0b10,
            &numeric,
            &base_shape,
        ))
        .expect("slice selectors for dims beyond mask width");
        assert_eq!(selectors.len(), 33);
        assert!(matches!(selectors[0], SliceSelector::Colon));
        assert!(matches!(selectors[1], SliceSelector::Scalar(40)));
        assert!(matches!(selectors[32], SliceSelector::Scalar(31)));
    }
}
