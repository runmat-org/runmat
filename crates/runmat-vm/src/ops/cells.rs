use crate::interpreter::errors::mex;
use runmat_builtins::{CellArray, Value};
use runmat_runtime::RuntimeError;

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
