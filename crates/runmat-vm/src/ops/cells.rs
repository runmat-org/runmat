use crate::interpreter::errors::mex;
use runmat_builtins::{CellArray, StructValue, Value};
use runmat_runtime::RuntimeError;

const CELL_END_PLUS_TAG_MASK: u64 = 0x7ff8_0000_0000_0000;
const CELL_END_PLUS_TAG_VALUE: u64 = 0x7ff8_c311_0000_0000;
const CELL_END_PLUS_OFFSET_MASK: u64 = 0x0000_0000_ffff_ffff;

fn decode_cell_end_plus(value: f64) -> Option<usize> {
    if !value.is_nan() {
        return None;
    }
    let bits = value.to_bits();
    if (bits & CELL_END_PLUS_TAG_MASK) != CELL_END_PLUS_TAG_VALUE {
        return None;
    }
    Some((bits & CELL_END_PLUS_OFFSET_MASK) as usize)
}

fn resolve_cell_end_relative_index(value: f64, len: usize) -> Result<Option<usize>, RuntimeError> {
    if let Some(offset) = decode_cell_end_plus(value) {
        let idx = len + offset;
        if idx < 1 || idx > len {
            return Err(mex("CellIndexOutOfBounds", "Cell index out of bounds"));
        }
        return Ok(Some(idx));
    }
    if value == 0.0 && value.is_sign_negative() {
        return Ok(Some(len));
    }
    if value < 0.0 {
        let idx = len as isize + value as isize;
        if idx < 1 || idx as usize > len {
            return Err(mex("CellIndexOutOfBounds", "Cell index out of bounds"));
        }
        return Ok(Some(idx as usize));
    }
    Ok(None)
}

fn is_empty_tensor(value: &Value) -> bool {
    matches!(value, Value::Tensor(t) if t.data.is_empty() || t.rows == 0 || t.cols == 0)
}

fn row_major_pos_from_linear(ca: &CellArray, idx: usize) -> Result<usize, RuntimeError> {
    if idx == 0 || idx > ca.data.len() {
        return Err(mex("CellIndexOutOfBounds", "Cell index out of bounds"));
    }
    if ca.rows <= 1 || ca.cols <= 1 {
        return Ok(idx - 1);
    }
    let zero = idx - 1;
    let row = zero % ca.rows;
    let col = zero / ca.rows;
    Ok(row * ca.cols + col)
}

pub fn create_cell_2d(values: Vec<Value>, rows: usize, cols: usize) -> Result<Value, RuntimeError> {
    runmat_runtime::make_cell_with_shape(values, vec![rows, cols])
        .map_err(|e| format!("Cell creation error: {e}").into())
}

pub fn index_cell_value(ca: &CellArray, indices: &[usize]) -> Result<Value, RuntimeError> {
    match indices.len() {
        1 => {
            let i = indices[0];
            Ok((*ca.data[row_major_pos_from_linear(ca, i)?]).clone())
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
        for idx in 1..=ca.data.len() {
            values.push(index_cell_value(ca, &[idx])?);
        }
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

pub fn expand_all_cell_values(ca: &CellArray) -> Result<Vec<Value>, RuntimeError> {
    (1..=ca.data.len())
        .map(|idx| index_cell_value(ca, &[idx]))
        .collect()
}

pub fn gather_cell_member(ca: &CellArray, field: &str) -> Result<Value, RuntimeError> {
    let mut out: Vec<Value> = Vec::with_capacity(ca.data.len());
    for value in &ca.data {
        match &**value {
            Value::Struct(st) => out.push(st.fields.get(field).cloned().unwrap_or(Value::Num(0.0))),
            other => out.push(other.clone()),
        }
    }
    let cell =
        CellArray::new(out, ca.rows, ca.cols).map_err(|e| format!("cell field gather: {e}"))?;
    Ok(Value::Cell(cell))
}

pub fn assign_cell_member<OnWrite>(
    mut ca: CellArray,
    field: String,
    rhs: Value,
    mut on_write: OnWrite,
) -> Result<Value, RuntimeError>
where
    OnWrite: FnMut(&Value, &Value),
{
    let rhs_cell = if let Value::Cell(rc) = &rhs {
        if rc.rows != ca.rows || rc.cols != ca.cols {
            return Err("Field assignment: cell rhs shape mismatch".into());
        }
        Some(rc)
    } else {
        None
    };

    for i in 0..ca.data.len() {
        let rv = if let Some(rc) = rhs_cell {
            (*rc.data[i]).clone()
        } else {
            rhs.clone()
        };
        match &mut *ca.data[i] {
            Value::Struct(st) => {
                if let Some(oldv) = st.fields.get(&field) {
                    on_write(oldv, &rv);
                }
                st.fields.insert(field.clone(), rv);
            }
            other => {
                let mut st = StructValue::new();
                st.fields.insert(field.clone(), rv);
                *other = Value::Struct(st);
            }
        }
    }
    Ok(Value::Cell(ca))
}

pub fn expand_cell_indices(ca: &CellArray, indices: &[Value]) -> Result<Vec<Value>, RuntimeError> {
    match indices.len() {
        1 => match &indices[0] {
            Value::Num(n) => {
                if let Some(idx) = resolve_cell_end_relative_index(*n, ca.data.len())? {
                    return Ok(vec![index_cell_value(ca, &[idx])?]);
                }
                Ok(vec![index_cell_value(ca, &[*n as usize])?])
            }
            Value::Int(i) => Ok(vec![index_cell_value(ca, &[i.to_i64() as usize])?]),
            Value::Tensor(t) => t
                .data
                .iter()
                .map(|&val| index_cell_value(ca, &[val as usize]))
                .collect(),
            _ => Err(mex("CellIndexType", "Unsupported cell index type")),
        },
        2 => {
            let r: f64 = (&indices[0]).try_into()?;
            let c: f64 = (&indices[1]).try_into()?;
            Ok(vec![index_cell_value(ca, &[r as usize, c as usize])?])
        }
        _ => Err(mex("CellIndexType", "Unsupported cell index type")),
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
            let pos = row_major_pos_from_linear(&ca, i)?;
            if let Some(oldv) = ca.data.get(pos) {
                on_write(oldv, &rhs);
            }
            *ca.data[pos] = rhs;
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
    if let Value::Cell(rhs_cell) = rhs {
        return assign_cell_paren_from_cell(ca, indices, rhs_cell);
    }
    Err(mex(
        "UnsupportedCellParenAssignment",
        "Only vector cell deletion is supported for cell paren assignment",
    ))
}

pub fn assign_cell_paren_linear_indices(
    mut ca: CellArray,
    indices: &[usize],
    rhs: &Value,
) -> Result<Value, RuntimeError> {
    if is_empty_tensor(rhs) {
        if !(ca.rows == 1 || ca.cols == 1) {
            return Err(mex(
                "UnsupportedCellDeletion",
                "Linear cell deletion is only supported for vectors",
            ));
        }
        let mut positions = indices
            .iter()
            .map(|&idx| row_major_pos_from_linear(&ca, idx))
            .collect::<Result<Vec<_>, _>>()?;
        positions.sort_unstable();
        positions.dedup();
        for pos in positions.into_iter().rev() {
            ca.data.remove(pos);
        }
        let len = ca.data.len();
        let shape = if len == 0 {
            vec![0, 0]
        } else if ca.rows == 1 {
            vec![1, len]
        } else {
            vec![len, 1]
        };
        return Ok(Value::Cell(
            CellArray::new_handles_with_shape(ca.data, shape)
                .map_err(|e| format!("Cell deletion error: {e}"))?,
        ));
    }
    let Value::Cell(rhs_cell) = rhs else {
        return Err(mex(
            "UnsupportedCellParenAssignment",
            "Cell paren assignment requires a cell RHS",
        ));
    };
    if rhs_cell.data.len() != indices.len() && rhs_cell.data.len() != 1 {
        return Err(mex(
            "UnsupportedCellParenAssignment",
            "Cell RHS must be scalar or match assignment size",
        ));
    }
    for (k, &idx) in indices.iter().enumerate() {
        let pos = row_major_pos_from_linear(&ca, idx)?;
        let rhs_pos = if rhs_cell.data.len() == 1 { 0 } else { k };
        let newv = (*rhs_cell.data[rhs_pos]).clone();
        if let Some(oldv) = ca.data.get(pos) {
            runmat_gc::gc_record_write(oldv, &newv);
        }
        *ca.data[pos] = newv;
    }
    Ok(Value::Cell(ca))
}

fn assign_cell_paren_from_cell(
    mut ca: CellArray,
    indices: &[usize],
    rhs: &CellArray,
) -> Result<Value, RuntimeError> {
    if rhs.data.len() != 1 {
        return Err(mex(
            "UnsupportedCellParenAssignment",
            "Only scalar cell paren assignment is supported",
        ));
    }
    let newv = (*rhs.data[0]).clone();
    let lin = match indices.len() {
        1 => {
            let i = indices[0];
            row_major_pos_from_linear(&ca, i)?
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
            (i - 1) * ca.cols + (j - 1)
        }
        _ => {
            return Err(mex(
                "UnsupportedCellIndexCount",
                "Unsupported number of cell indices",
            ))
        }
    };
    if let Some(oldv) = ca.data.get(lin) {
        runmat_gc::gc_record_write(oldv, &newv);
    }
    *ca.data[lin] = newv;
    Ok(Value::Cell(ca))
}
