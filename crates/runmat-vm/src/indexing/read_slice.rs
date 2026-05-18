use crate::indexing::plan::{build_index_plan, IndexPlan};
use crate::indexing::selectors::{
    build_slice_selectors, index_scalar_from_value, materialize_index_value, SliceSelector,
};
use runmat_builtins::{ComplexTensor, StringArray, Tensor, Value};
use runmat_runtime::RuntimeError;

pub async fn read_tensor_slice_1d(
    tensor: &Tensor,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    let total = tensor.data.len();
    let mut idxs: Vec<usize> = Vec::new();
    let mut idx_shape: Option<Vec<usize>> = None;
    let is_colon = (colon_mask & 1u32) != 0;
    let is_end = (end_mask & 1u32) != 0;
    if is_colon {
        idxs = (1..=total).collect();
    } else if is_end {
        idxs = vec![total];
    } else if let Some(v) = numeric.first() {
        let materialized = materialize_index_value(v).await?;
        if let Some(i) = index_scalar_from_value(&materialized).await? {
            if i < 1 {
                return Err(crate::interpreter::errors::mex(
                    "IndexOutOfBounds",
                    "Index out of bounds",
                ));
            }
            idxs = vec![i as usize];
        } else {
            match &materialized {
                Value::Tensor(idx_t) => {
                    idx_shape = Some(idx_t.shape.clone());
                    for &val in &idx_t.data {
                        let i = val as isize;
                        if i < 1 || (i as usize) > total {
                            return Err(crate::interpreter::errors::mex(
                                "IndexOutOfBounds",
                                "Index out of bounds",
                            ));
                        }
                        idxs.push(i as usize);
                    }
                }
                Value::Bool(b) => {
                    if *b {
                        idxs = vec![1];
                    }
                }
                Value::LogicalArray(la) => {
                    if la.data.len() != total {
                        return Err(crate::interpreter::errors::mex(
                            "IndexShape",
                            "Logical mask length mismatch for linear indexing",
                        ));
                    }
                    for (i, &val) in la.data.iter().enumerate() {
                        if val != 0 {
                            idxs.push(i + 1);
                        }
                    }
                }
                _ => {
                    return Err(crate::interpreter::errors::mex(
                        "UnsupportedIndexType",
                        "Unsupported index type",
                    ))
                }
            }
        }
    } else {
        return Err(crate::interpreter::errors::mex(
            "MissingNumericIndex",
            "missing numeric index",
        ));
    }
    if idxs.iter().any(|&i| i == 0 || i > total) {
        return Err(crate::interpreter::errors::mex(
            "IndexOutOfBounds",
            "Index out of bounds",
        ));
    }
    if idxs.len() == 1 {
        Ok(Value::Num(tensor.data[idxs[0] - 1]))
    } else if idxs.is_empty() {
        let shape = idx_shape.unwrap_or_else(|| vec![0, 1]);
        let tens = Tensor::new(vec![], shape).map_err(|e| format!("Slice error: {e}"))?;
        Ok(Value::Tensor(tens))
    } else {
        let mut out = Vec::with_capacity(idxs.len());
        for &i in &idxs {
            out.push(tensor.data[i - 1]);
        }
        let shape = idx_shape.unwrap_or_else(|| vec![idxs.len(), 1]);
        let tens = Tensor::new(out, shape).map_err(|e| format!("Slice error: {e}"))?;
        Ok(Value::Tensor(tens))
    }
}

pub fn try_tensor_slice_2d_fast_path(
    tensor: &Tensor,
    dims: usize,
    selectors: &[SliceSelector],
) -> Result<Option<Value>, RuntimeError> {
    if dims != 2 {
        return Ok(None);
    }
    let rows = tensor.shape.first().copied().unwrap_or(1);
    let cols = tensor.shape.get(1).copied().unwrap_or(1);
    match (&selectors[0], &selectors[1]) {
        (SliceSelector::Colon, SliceSelector::Scalar(j)) => {
            let j0 = *j - 1;
            if j0 >= cols {
                return Err(crate::interpreter::errors::mex(
                    "IndexOutOfBounds",
                    "Index out of bounds",
                ));
            }
            let start = j0 * rows;
            let out = tensor.data[start..start + rows].to_vec();
            if out.len() == 1 {
                Ok(Some(Value::Num(out[0])))
            } else {
                let tens =
                    Tensor::new(out, vec![rows, 1]).map_err(|e| format!("Slice error: {e}"))?;
                Ok(Some(Value::Tensor(tens)))
            }
        }
        (SliceSelector::Scalar(i), SliceSelector::Colon) => {
            let i0 = *i - 1;
            if i0 >= rows {
                return Err(crate::interpreter::errors::mex(
                    "IndexOutOfBounds",
                    "Index out of bounds",
                ));
            }
            let mut out: Vec<f64> = Vec::with_capacity(cols);
            for c in 0..cols {
                out.push(tensor.data[i0 + c * rows]);
            }
            if out.len() == 1 {
                Ok(Some(Value::Num(out[0])))
            } else {
                let tens =
                    Tensor::new(out, vec![1, cols]).map_err(|e| format!("Slice error: {e}"))?;
                Ok(Some(Value::Tensor(tens)))
            }
        }
        (SliceSelector::Colon, SliceSelector::Indices(js)) => {
            if js.is_empty() {
                let tens = Tensor::new(Vec::new(), vec![rows, 0])
                    .map_err(|e| format!("Slice error: {e}"))?;
                Ok(Some(Value::Tensor(tens)))
            } else {
                let mut out: Vec<f64> = Vec::with_capacity(rows * js.len());
                for &j in js {
                    let j0 = j - 1;
                    if j0 >= cols {
                        return Err(crate::interpreter::errors::mex(
                            "IndexOutOfBounds",
                            "Index out of bounds",
                        ));
                    }
                    let start = j0 * rows;
                    out.extend_from_slice(&tensor.data[start..start + rows]);
                }
                let tens = Tensor::new(out, vec![rows, js.len()])
                    .map_err(|e| format!("Slice error: {e}"))?;
                Ok(Some(Value::Tensor(tens)))
            }
        }
        (SliceSelector::Indices(is), SliceSelector::Colon) => {
            if is.is_empty() {
                let tens = Tensor::new(Vec::new(), vec![0, cols])
                    .map_err(|e| format!("Slice error: {e}"))?;
                Ok(Some(Value::Tensor(tens)))
            } else {
                let mut out: Vec<f64> = Vec::with_capacity(is.len() * cols);
                for c in 0..cols {
                    for &i in is {
                        let i0 = i - 1;
                        if i0 >= rows {
                            return Err(crate::interpreter::errors::mex(
                                "IndexOutOfBounds",
                                "Index out of bounds",
                            ));
                        }
                        out.push(tensor.data[i0 + c * rows]);
                    }
                }
                let tens = Tensor::new(out, vec![is.len(), cols])
                    .map_err(|e| format!("Slice error: {e}"))?;
                Ok(Some(Value::Tensor(tens)))
            }
        }
        _ => Ok(None),
    }
}

pub async fn read_tensor_slice_nd(
    tensor: &Tensor,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    let selectors =
        build_slice_selectors(dims, colon_mask, end_mask, numeric, &tensor.shape).await?;
    if let Some(value) = try_tensor_slice_2d_fast_path(tensor, dims, &selectors)? {
        return Ok(value);
    }
    let plan = build_index_plan(&selectors, dims, &tensor.shape)?;
    if plan.indices.is_empty() {
        let out_tensor =
            Tensor::new(Vec::new(), plan.output_shape).map_err(|e| format!("Slice error: {e}"))?;
        return Ok(Value::Tensor(out_tensor));
    }
    let mut out_data: Vec<f64> = Vec::with_capacity(plan.indices.len());
    for &lin in &plan.indices {
        out_data.push(tensor.data[lin as usize]);
    }
    if out_data.len() == 1 {
        Ok(Value::Num(out_data[0]))
    } else {
        let out_tensor =
            Tensor::new(out_data, plan.output_shape).map_err(|e| format!("Slice error: {e}"))?;
        Ok(Value::Tensor(out_tensor))
    }
}

pub fn read_tensor_slice_from_plan(
    tensor: &Tensor,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    if plan.indices.is_empty() {
        let out_tensor = Tensor::new(Vec::new(), plan.output_shape.clone())
            .map_err(|e| format!("Slice error: {e}"))?;
        return Ok(Value::Tensor(out_tensor));
    }
    let mut out_data: Vec<f64> = Vec::with_capacity(plan.indices.len());
    for &lin in &plan.indices {
        out_data.push(tensor.data[lin as usize]);
    }
    if out_data.len() == 1 {
        Ok(Value::Num(out_data[0]))
    } else {
        let out_tensor = Tensor::new(out_data, plan.output_shape.clone())
            .map_err(|e| format!("Slice error: {e}"))?;
        Ok(Value::Tensor(out_tensor))
    }
}

pub async fn read_complex_slice(
    tensor: &ComplexTensor,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    let selectors =
        build_slice_selectors(dims, colon_mask, end_mask, numeric, &tensor.shape).await?;
    let plan = build_index_plan(&selectors, dims, &tensor.shape)?;
    read_complex_slice_from_plan(tensor, &plan)
}

pub fn read_complex_slice_from_plan(
    tensor: &ComplexTensor,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    if plan.indices.is_empty() {
        let empty = ComplexTensor::new(Vec::new(), plan.output_shape.clone())
            .map_err(|e| format!("Slice error: {e}"))?;
        return Ok(Value::ComplexTensor(empty));
    }
    if plan.indices.len() == 1 {
        let lin = plan.indices[0] as usize;
        let (re, im) = tensor.data.get(lin).copied().ok_or_else(|| {
            crate::interpreter::errors::mex(
                "IndexOutOfBounds",
                "Slice error: complex index out of bounds",
            )
        })?;
        return Ok(Value::Complex(re, im));
    }
    let mut out = Vec::with_capacity(plan.indices.len());
    for &lin in &plan.indices {
        let idx = lin as usize;
        let value = tensor.data.get(idx).copied().ok_or_else(|| {
            crate::interpreter::errors::mex(
                "IndexOutOfBounds",
                "Slice error: complex index out of bounds",
            )
        })?;
        out.push(value);
    }
    let out_ct = ComplexTensor::new(out, plan.output_shape.clone())
        .map_err(|e| format!("Slice error: {e}"))?;
    Ok(Value::ComplexTensor(out_ct))
}

pub async fn read_gpu_slice(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    let base_shape = handle.shape.clone();
    let selectors = build_slice_selectors(dims, colon_mask, end_mask, numeric, &base_shape).await?;
    let plan = build_index_plan(&selectors, dims, &base_shape)?;
    read_gpu_slice_from_plan(handle, &plan)
}

pub fn read_gpu_slice_from_plan(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        crate::interpreter::errors::mex(
            "AccelerationProviderUnavailable",
            "No acceleration provider registered",
        )
    })?;
    if plan.indices.is_empty() {
        let zeros = provider
            .zeros(&plan.output_shape)
            .map_err(|e| format!("slice: {e}"))?;
        Ok(Value::GpuTensor(zeros))
    } else {
        let result = provider
            .gather_linear(handle, &plan.indices, &plan.output_shape)
            .map_err(|e| format!("slice: {e}"))?;
        Ok(Value::GpuTensor(result))
    }
}

pub async fn read_string_slice(
    sa: &StringArray,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    let selectors = build_slice_selectors(dims, colon_mask, end_mask, numeric, &sa.shape).await?;
    let plan = build_index_plan(&selectors, dims, &sa.shape)?;
    gather_string_slice(sa, &plan)
}

pub fn gather_string_slice(sa: &StringArray, plan: &IndexPlan) -> Result<Value, RuntimeError> {
    if plan.indices.is_empty() {
        let empty = StringArray::new(Vec::new(), plan.output_shape.clone())
            .map_err(|e| format!("Slice error: {e}"))?;
        return Ok(Value::StringArray(empty));
    }
    if plan.indices.len() == 1 {
        let lin = plan.indices[0] as usize;
        let value = sa.data.get(lin).cloned().ok_or_else(|| {
            crate::interpreter::errors::mex(
                "IndexOutOfBounds",
                "Slice error: string index out of bounds",
            )
        })?;
        return Ok(Value::String(value));
    }
    let mut out = Vec::with_capacity(plan.indices.len());
    for &lin in &plan.indices {
        let idx = lin as usize;
        let value = sa.data.get(idx).cloned().ok_or_else(|| {
            crate::interpreter::errors::mex(
                "IndexOutOfBounds",
                "Slice error: string index out of bounds",
            )
        })?;
        out.push(value);
    }
    let out_sa = StringArray::new(out, plan.output_shape.clone())
        .map_err(|e| format!("Slice error: {e}"))?;
    Ok(Value::StringArray(out_sa))
}

#[cfg(test)]
mod tests {
    use super::read_string_slice;
    use futures::executor::block_on;
    use runmat_builtins::{StringArray, Tensor, Value};

    #[test]
    fn string_slice_linear_tensor_indices_preserve_selector_shape() {
        let sa = StringArray::new(
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
            vec![2, 2],
        )
        .expect("string array");
        let selector =
            Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![1, 2]).expect("selector tensor"));
        let result = block_on(read_string_slice(&sa, 1, 0, 0, &[selector])).expect("slice");
        match result {
            Value::StringArray(out) => {
                assert_eq!(out.shape, vec![1, 2]);
                assert_eq!(out.data, vec!["a".to_string(), "c".to_string()]);
            }
            other => panic!("expected string array result, got {other:?}"),
        }
    }

    #[test]
    fn string_slice_colon_then_scalar_selects_column() {
        let sa = StringArray::new(
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
            vec![2, 2],
        )
        .expect("string array");
        let result =
            block_on(read_string_slice(&sa, 2, 0b01, 0, &[Value::Num(2.0)])).expect("slice");
        match result {
            Value::StringArray(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data, vec!["c".to_string(), "d".to_string()]);
            }
            other => panic!("expected string array result, got {other:?}"),
        }
    }
}
