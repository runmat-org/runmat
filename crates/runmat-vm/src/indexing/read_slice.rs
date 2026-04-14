use crate::indexing::selectors::{
    build_slice_plan, build_slice_selectors, index_scalar_from_value, materialize_index_value,
    SliceSelector,
};
use runmat_builtins::{CellArray, Tensor, Value};
use runmat_runtime::RuntimeError;

pub fn build_numeric_subsref_cell(numeric: &[Value]) -> Result<Value, RuntimeError> {
    let cell = CellArray::new(numeric.to_vec(), 1, numeric.len())
        .map_err(|e| format!("subsref build error: {e}"))?;
    Ok(Value::Cell(cell))
}

pub async fn object_subsref_paren(base: Value, numeric: &[Value]) -> Result<Value, RuntimeError> {
    let cell = build_numeric_subsref_cell(numeric)?;
    match base {
        Value::Object(obj) => {
            let args = vec![
                Value::Object(obj),
                Value::String("subsref".to_string()),
                Value::String("()".to_string()),
                cell,
            ];
            runmat_runtime::call_builtin_async("call_method", &args).await
        }
        Value::HandleObject(handle) => {
            let args = vec![
                Value::HandleObject(handle),
                Value::String("subsref".to_string()),
                Value::String("()".to_string()),
                cell,
            ];
            runmat_runtime::call_builtin_async("call_method", &args).await
        }
        other => Err(format!("slice subsref requires object/handle, got {other:?}").into()),
    }
}

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
                let tens = Tensor::new(out, vec![rows, 1]).map_err(|e| format!("Slice error: {e}"))?;
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
                let tens = Tensor::new(out, vec![1, cols]).map_err(|e| format!("Slice error: {e}"))?;
                Ok(Some(Value::Tensor(tens)))
            }
        }
        (SliceSelector::Colon, SliceSelector::Indices(js)) => {
            if js.is_empty() {
                let tens = Tensor::new(Vec::new(), vec![rows, 0]).map_err(|e| format!("Slice error: {e}"))?;
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
                let tens = Tensor::new(Vec::new(), vec![0, cols]).map_err(|e| format!("Slice error: {e}"))?;
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
    let selectors = build_slice_selectors(dims, colon_mask, end_mask, numeric, &tensor.shape).await?;
    if let Some(value) = try_tensor_slice_2d_fast_path(tensor, dims, &selectors)? {
        return Ok(value);
    }
    let plan = build_slice_plan(&selectors, dims, &tensor.shape)?;
    if plan.indices.is_empty() {
        let out_tensor = Tensor::new(Vec::new(), plan.output_shape).map_err(|e| format!("Slice error: {e}"))?;
        return Ok(Value::Tensor(out_tensor));
    }
    let mut out_data: Vec<f64> = Vec::with_capacity(plan.indices.len());
    for &lin in &plan.indices {
        out_data.push(tensor.data[lin as usize]);
    }
    if out_data.len() == 1 {
        Ok(Value::Num(out_data[0]))
    } else {
        let out_tensor = Tensor::new(out_data, plan.output_shape).map_err(|e| format!("Slice error: {e}"))?;
        Ok(Value::Tensor(out_tensor))
    }
}
