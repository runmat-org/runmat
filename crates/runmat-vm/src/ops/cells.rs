use crate::interpreter::errors::mex;
use runmat_builtins::{CellArray, Value};
use runmat_runtime::RuntimeError;

fn is_empty_tensor(value: &Value) -> bool {
    matches!(value, Value::Tensor(t) if t.data.is_empty() || t.rows == 0 || t.cols == 0)
}

pub fn create_cell_2d(values: Vec<Value>, rows: usize, cols: usize) -> Result<Value, RuntimeError> {
    runmat_runtime::make_cell_with_shape(values, vec![rows, cols])
        .map_err(|e| format!("Cell creation error: {e}").into())
}

pub fn index_cell_value(ca: &CellArray, indices: &[usize]) -> Result<Value, RuntimeError> {
    match indices.len() {
        1 => {
            let i = indices[0];
            if i == 0 || i > ca.data.len() {
                return Err(mex("CellIndexOutOfBounds", "Cell index out of bounds"));
            }
            Ok((*ca.data[i - 1]).clone())
        }
        2 => {
            let r = indices[0];
            let c = indices[1];
            if r == 0 || r > ca.rows || c == 0 || c > ca.cols {
                return Err(mex(
                    "CellSubscriptOutOfBounds",
                    "Cell subscript out of bounds",
                ));
            }
            Ok((*ca.data[(r - 1) * ca.cols + (c - 1)]).clone())
        }
        _ => Err(mex(
            "UnsupportedCellIndexCount",
            "Unsupported number of cell indices",
        )),
    }
}

pub fn expand_cell_values(
    ca: &CellArray,
    indices: &[usize],
    out_count: usize,
) -> Result<Vec<Value>, RuntimeError> {
    let mut values: Vec<Value> = Vec::new();
    if indices.is_empty() {
        values.extend(ca.data.iter().map(|p| (*(*p)).clone()));
    } else {
        values.push(index_cell_value(ca, indices)?);
    }
    if values.len() >= out_count {
        Ok(values.into_iter().take(out_count).collect())
    } else {
        let mut out = values;
        out.resize(out_count, Value::Num(0.0));
        Ok(out)
    }
}

pub fn assign_cell_value<OnWrite>(
    mut ca: CellArray,
    indices: &[usize],
    rhs: Value,
    mut on_write: OnWrite,
) -> Result<Value, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    match indices.len() {
        1 => {
            let i = indices[0];
            if i == 0 || i > ca.data.len() {
                return Err(mex("CellIndexOutOfBounds", "Cell index out of bounds"));
            }
            if let Some(oldv) = ca.data.get(i - 1) {
                on_write(oldv, &rhs);
            }
            *ca.data[i - 1] = rhs;
            Ok(Value::Cell(ca))
        }
        2 => {
            let i = indices[0];
            let j = indices[1];
            if i == 0 || i > ca.rows || j == 0 || j > ca.cols {
                return Err(mex(
                    "CellSubscriptOutOfBounds",
                    "Cell subscript out of bounds",
                ));
            }
            let lin = (i - 1) * ca.cols + (j - 1);
            if let Some(oldv) = ca.data.get(lin) {
                on_write(oldv, &rhs);
            }
            *ca.data[lin] = rhs;
            Ok(Value::Cell(ca))
        }
        _ => Err(mex(
            "UnsupportedCellIndexCount",
            "Unsupported number of cell indices",
        )),
    }
}

pub fn delete_cell_linear(mut ca: CellArray, idx: usize) -> Result<Value, RuntimeError> {
    let total = ca.data.len();
    if idx == 0 || idx > total {
        return Err(mex("CellIndexOutOfBounds", "Cell index out of bounds"));
    }
    if !(ca.rows == 1 || ca.cols == 1) {
        return Err(mex(
            "UnsupportedCellDeletion",
            "Linear cell deletion is only supported for vectors",
        ));
    }
    ca.data.remove(idx - 1);
    let len = ca.data.len();
    let shape = if len == 0 {
        vec![0, 0]
    } else if ca.rows == 1 {
        vec![1, len]
    } else {
        vec![len, 1]
    };
    Ok(Value::Cell(
        CellArray::new_handles_with_shape(ca.data, shape)
            .map_err(|e| format!("Cell deletion error: {e}"))?,
    ))
}

pub fn assign_cell_paren(
    ca: CellArray,
    indices: &[usize],
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    if indices.len() == 1 && is_empty_tensor(rhs) {
        return delete_cell_linear(ca, indices[0]);
    }
    Err(mex(
        "UnsupportedCellParenAssignment",
        "Only vector cell deletion is supported for cell paren assignment",
    ))
}
