use crate::indexing::selectors::{index_scalar_from_value, materialize_index_value};
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
