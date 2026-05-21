use crate::indexing::plan::total_len_from_shape;
use crate::interpreter::errors::mex;
use runmat_builtins::Value;
use runmat_runtime::{
    builtins::common::shape::is_scalar_shape, dispatcher::gather_if_needed_async, RuntimeError,
};

pub type VmResult<T> = Result<T, RuntimeError>;

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
            .map_err(|e| mex("IndexGather", &format!("Failed to gather index value: {e}")));
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
            selectors.push(SliceSelector::Indices((1..=total_len).collect()));
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
        let is_colon = (colon_mask & (1u32 << d)) != 0;
        if is_colon {
            selectors.push(SliceSelector::Colon);
            continue;
        }
        let dim_len = base_shape.get(d).copied().unwrap_or(1);
        let is_end = (end_mask & (1u32 << d)) != 0;
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
        selectors.push(SliceSelector::Scalar(if idx_val <= 0 {
            0
        } else {
            idx_val as usize
        }));
    }
    Ok(selectors)
}

#[cfg(test)]
mod tests {
    use super::{indices_from_value_linear, selector_from_value_dim};
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
}
