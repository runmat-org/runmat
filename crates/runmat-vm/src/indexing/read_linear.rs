use crate::interpreter::errors::mex;
use crate::indexing::selectors::index_scalar_from_value;
use runmat_builtins::{CellArray, Value};
use runmat_runtime::RuntimeError;

pub async fn collect_linear_indices(
    stack: &mut Vec<Value>,
    count: usize,
) -> Result<Vec<f64>, RuntimeError> {
    let mut indices = Vec::with_capacity(count);
    for _ in 0..count {
        let mut index_value = stack
            .pop()
            .ok_or_else(|| mex("StackUnderflow", "stack underflow"))?;
        let original_value = index_value.clone();
        if matches!(index_value, Value::GpuTensor(_)) {
            index_value = runmat_runtime::dispatcher::gather_if_needed_async(&index_value).await?;
        }
        let index_val = match index_scalar_from_value(&index_value).await? {
            Some(val) => val as f64,
            None => {
                return Err(mex(
                    "UnsupportedIndexType",
                    &format!(
                        "Unsupported index type: expected numeric scalar, got {original_value:?}"
                    ),
                ))
            }
        };
        indices.push(index_val);
    }
    indices.reverse();
    Ok(indices)
}

pub fn build_object_subsref_cell(indices: &[f64]) -> Result<Value, RuntimeError> {
    let cell = CellArray::new(indices.iter().map(|n| Value::Num(*n)).collect(), 1, indices.len())
        .map_err(|e| format!("subsref build error: {e}"))?;
    Ok(Value::Cell(cell))
}

pub async fn generic_index(base: &Value, indices: &[f64]) -> Result<Value, RuntimeError> {
    runmat_runtime::perform_indexing(base, indices).await
}
