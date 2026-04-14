use crate::interpreter::errors::mex;
use crate::indexing::selectors::indices_from_value_linear;
use runmat_builtins::{CellArray, Tensor, Value};
use runmat_runtime::RuntimeError;

pub fn build_subsasgn_paren_cell(numeric: &[Value]) -> Result<Value, RuntimeError> {
    let cell = CellArray::new(numeric.to_vec(), 1, numeric.len())
        .map_err(|e| format!("subsasgn build error: {e}"))?;
    Ok(Value::Cell(cell))
}

pub async fn object_subsasgn_paren(base: Value, numeric: &[Value], rhs: Value) -> Result<Value, RuntimeError> {
    let cell = build_subsasgn_paren_cell(numeric)?;
    match base {
        Value::Object(obj) => {
            let args = vec![
                Value::Object(obj),
                Value::String("subsasgn".to_string()),
                Value::String("()".to_string()),
                cell,
                rhs,
            ];
            runmat_runtime::call_builtin_async("call_method", &args).await
        }
        Value::HandleObject(handle) => {
            let args = vec![
                Value::HandleObject(handle),
                Value::String("subsasgn".to_string()),
                Value::String("()".to_string()),
                cell,
                rhs,
            ];
            runmat_runtime::call_builtin_async("call_method", &args).await
        }
        other => Err(format!("slice subsasgn requires object/handle, got {other:?}").into()),
    }
}

pub async fn assign_tensor_slice_1d(
    mut t: Tensor,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
    rhs: Value,
) -> Result<Value, RuntimeError> {
    let total = t.data.len();
    let is_colon = (colon_mask & 1u32) != 0;
    let is_end = (end_mask & 1u32) != 0;
    let lin_indices: Vec<usize> = if is_colon {
        (1..=total).collect()
    } else if is_end {
        vec![total]
    } else {
        let v = numeric
            .first()
            .ok_or_else(|| mex("MissingNumericIndex", "missing numeric index"))?;
        indices_from_value_linear(v, total).await?
    };
    match rhs {
        Value::Num(v) => {
            for &li in &lin_indices {
                t.data[li - 1] = v;
            }
        }
        Value::Tensor(rt) => {
            if rt.data.len() == 1 {
                let v = rt.data[0];
                for &li in &lin_indices {
                    t.data[li - 1] = v;
                }
            } else if rt.data.len() == lin_indices.len() {
                for (k, &li) in lin_indices.iter().enumerate() {
                    t.data[li - 1] = rt.data[k];
                }
            } else {
                return Err("shape mismatch for linear slice assign".to_string().into());
            }
        }
        _ => return Err("rhs must be numeric or tensor".to_string().into()),
    }
    Ok(Value::Tensor(t))
}
