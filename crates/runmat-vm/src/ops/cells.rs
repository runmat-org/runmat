use crate::interpreter::errors::mex;
use runmat_builtins::{CellArray, StructValue, Tensor, Value};
use runmat_runtime::RuntimeError;

const CELL_END_PLUS_TAG_MASK: u64 = 0x7ff8_0000_0000_0000;
const CELL_END_PLUS_TAG_VALUE: u64 = 0x7ff8_c311_0000_0000;
const CELL_END_PLUS_OFFSET_MASK: u64 = 0x0000_0000_ffff_ffff;

fn map_cell_shape_error(context: &str, err: impl std::fmt::Display) -> RuntimeError {
    mex("ShapeMismatch", &format!("{context}: {err}"))
}

fn allocate_cell_handle(value: Value) -> Result<runmat_gc::GcPtr<Value>, RuntimeError> {
    runmat_gc::gc_allocate(value).map_err(|e| {
        mex(
            "CellAllocationFailed",
            &format!("failed to allocate cell element handle: {e}"),
        )
    })
}

fn empty_numeric_cell_value() -> Result<Value, RuntimeError> {
    Tensor::new(Vec::new(), vec![0, 0])
        .map(Value::Tensor)
        .map_err(|e| map_cell_shape_error("cell growth empty filler", e))
}

fn allocate_empty_cell_handle() -> Result<runmat_gc::GcPtr<Value>, RuntimeError> {
    allocate_cell_handle(empty_numeric_cell_value()?)
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

fn parse_positive_cell_index(index: i64) -> Result<usize, RuntimeError> {
    if index < 1 {
        return Err(mex("CellIndexOutOfBounds", "Cell index out of bounds"));
    }
    usize::try_from(index).map_err(|_| mex("CellIndexOutOfBounds", "Cell index out of bounds"))
}

fn parse_cell_index_value(value: &Value) -> Result<usize, RuntimeError> {
    let index = match value {
        Value::Num(n) => exact_index_from_f64(*n)
            .ok_or_else(|| mex("CellIndexType", "Unsupported cell index type"))?,
        Value::Int(i) => i.to_i64(),
        Value::Tensor(t) if t.data.len() == 1 && t.shape.iter().product::<usize>() == 1 => {
            exact_index_from_f64(t.data[0])
                .ok_or_else(|| mex("CellIndexType", "Unsupported cell index type"))?
        }
        other => {
            let n: f64 = other
                .try_into()
                .map_err(|_| mex("CellIndexType", "Unsupported cell index type"))?;
            exact_index_from_f64(n)
                .ok_or_else(|| mex("CellIndexType", "Unsupported cell index type"))?
        }
    };
    parse_positive_cell_index(index)
}

fn parse_cell_index_value_for_len(value: &Value, len: usize) -> Result<usize, RuntimeError> {
    match value {
        Value::Num(n) => {
            if let Some(idx) = resolve_cell_end_relative_index(*n, len)? {
                return Ok(idx);
            }
            if n.is_nan() {
                return Err(mex("CellIndexOutOfBounds", "Cell index out of bounds"));
            }
        }
        Value::Tensor(t) if t.data.len() == 1 && t.shape.iter().product::<usize>() == 1 => {
            if let Some(idx) = resolve_cell_end_relative_index(t.data[0], len)? {
                return Ok(idx);
            }
            if t.data[0].is_nan() {
                return Err(mex("CellIndexOutOfBounds", "Cell index out of bounds"));
            }
        }
        _ => {}
    }
    parse_cell_index_value(value)
}

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
        let offset = exact_index_from_f64(value)
            .ok_or_else(|| mex("CellIndexType", "Unsupported cell index type"))?;
        let idx = len as i64 + offset;
        if idx < 1 || idx > len as i64 {
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
        .map_err(|e| map_cell_shape_error("cell creation error", e))
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

pub fn gather_cell_paren_linear_indices(
    ca: &CellArray,
    indices: &[usize],
    output_shape: &[usize],
) -> Result<Value, RuntimeError> {
    let mut handles = Vec::with_capacity(indices.len());
    for &idx in indices {
        let pos = row_major_pos_from_linear(ca, idx)?;
        handles.push(ca.data[pos].clone());
    }
    let shape = if output_shape.is_empty() {
        vec![1, handles.len().max(1)]
    } else {
        output_shape.to_vec()
    };
    Ok(Value::Cell(
        CellArray::new_handles_with_shape(handles, shape)
            .map_err(|e| map_cell_shape_error("cell paren indexing error", e))?,
    ))
}

pub fn gather_cell_member(ca: &CellArray, field: &str) -> Result<Value, RuntimeError> {
    if ca.data.len() == 1 {
        return Ok(match &*ca.data[0] {
            Value::Struct(st) => st.fields.get(field).cloned().unwrap_or(Value::Num(0.0)),
            other => other.clone(),
        });
    }

    let mut out: Vec<Value> = Vec::with_capacity(ca.data.len());
    for value in &ca.data {
        match &**value {
            Value::Struct(st) => out.push(st.fields.get(field).cloned().unwrap_or(Value::Num(0.0))),
            other => out.push(other.clone()),
        }
    }
    let cell = CellArray::new(out, ca.rows, ca.cols)
        .map_err(|e| map_cell_shape_error("cell field gather", e))?;
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
            return Err(mex(
                "CellMemberRhsShapeMismatch",
                "field assignment cell RHS shape mismatch",
            ));
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
    fn is_colon_selector(value: &Value) -> bool {
        matches!(value, Value::String(text) if text == ":")
            || matches!(value, Value::CharArray(chars) if chars.to_string() == ":")
    }

    match indices.len() {
        1 => match &indices[0] {
            value if is_colon_selector(value) => expand_all_cell_values(ca),
            Value::Num(n) => {
                if let Some(idx) = resolve_cell_end_relative_index(*n, ca.data.len())? {
                    return Ok(vec![index_cell_value(ca, &[idx])?]);
                }
                if n.is_nan() {
                    return Err(mex("CellIndexOutOfBounds", "Cell index out of bounds"));
                }
                let idx = parse_cell_index_value(&indices[0])?;
                Ok(vec![index_cell_value(ca, &[idx])?])
            }
            Value::Int(_) => {
                let idx = parse_cell_index_value(&indices[0])?;
                Ok(vec![index_cell_value(ca, &[idx])?])
            }
            Value::Tensor(t) => t
                .data
                .iter()
                .map(|&val| {
                    if t.data.len() == 1 && t.shape.iter().product::<usize>() == 1 {
                        let idx = parse_cell_index_value_for_len(&indices[0], ca.data.len())?;
                        return index_cell_value(ca, &[idx]);
                    }
                    let idx = exact_index_from_f64(val)
                        .ok_or_else(|| mex("CellIndexType", "Unsupported cell index type"))?;
                    let idx = parse_positive_cell_index(idx)?;
                    index_cell_value(ca, &[idx])
                })
                .collect(),
            _ => Err(mex("CellIndexType", "Unsupported cell index type")),
        },
        2 => {
            let row_colon = is_colon_selector(&indices[0]);
            let col_colon = is_colon_selector(&indices[1]);
            if row_colon && col_colon {
                return expand_all_cell_values(ca);
            }
            if row_colon {
                let c = parse_cell_index_value_for_len(&indices[1], ca.cols)?;
                let mut values = Vec::with_capacity(ca.rows);
                for r in 1..=ca.rows {
                    values.push(index_cell_value(ca, &[r, c])?);
                }
                return Ok(values);
            }
            if col_colon {
                let r = parse_cell_index_value_for_len(&indices[0], ca.rows)?;
                let mut values = Vec::with_capacity(ca.cols);
                for c in 1..=ca.cols {
                    values.push(index_cell_value(ca, &[r, c])?);
                }
                return Ok(values);
            }
            let r = parse_cell_index_value_for_len(&indices[0], ca.rows)?;
            let c = parse_cell_index_value_for_len(&indices[1], ca.cols)?;
            Ok(vec![index_cell_value(ca, &[r, c])?])
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
            let old_len = ca.data.len();
            if i > old_len {
                if !(ca.data.is_empty() || ca.rows <= 1 || ca.cols <= 1) {
                    return Err(mex(
                        "UnsupportedCellGrowth",
                        "Cell growth via linear brace assignment is only supported for vectors",
                    ));
                }
                while ca.data.len() < i {
                    ca.data.push(allocate_empty_cell_handle()?);
                }
                let len = ca.data.len();
                if old_len == 0 {
                    // MATLAB linear growth from empty cell shapes (for example `0x5` or `5x0`)
                    // normalizes to a row vector.
                    ca.rows = 1;
                    ca.cols = len;
                    ca.shape = vec![1, len];
                } else if ca.rows <= 1 {
                    ca.rows = 1;
                    ca.cols = len;
                    ca.shape = vec![1, len];
                } else {
                    ca.rows = len;
                    ca.cols = 1;
                    ca.shape = vec![len, 1];
                }
            }
            let pos = row_major_pos_from_linear(&ca, i)?;
            if i <= old_len {
                if let Some(oldv) = ca.data.get(pos) {
                    on_write(oldv, &rhs);
                }
            }
            ca.data[pos] = allocate_cell_handle(rhs)?;
            Ok(Value::Cell(ca))
        }
        2 => {
            let i = indices[0];
            let j = indices[1];
            if i == 0 || j == 0 {
                return Err(mex(
                    "CellSubscriptOutOfBounds",
                    "Cell subscript out of bounds",
                ));
            }
            if i > ca.rows || j > ca.cols {
                let old_rows = ca.rows;
                let old_cols = ca.cols;
                let new_rows = old_rows.max(i);
                let new_cols = old_cols.max(j);
                let total = new_rows.checked_mul(new_cols).ok_or_else(|| {
                    mex(
                        "CellSubscriptOutOfBounds",
                        "Cell array expansion exceeds supported size",
                    )
                })?;
                let mut grown = Vec::with_capacity(total);
                for _ in 0..total {
                    grown.push(allocate_empty_cell_handle()?);
                }
                for row in 0..old_rows {
                    for col in 0..old_cols {
                        let old_lin = row * old_cols + col;
                        let new_lin = row * new_cols + col;
                        grown[new_lin] = ca.data[old_lin].clone();
                    }
                }
                ca.data = grown;
                ca.rows = new_rows;
                ca.cols = new_cols;
                ca.shape = vec![new_rows, new_cols];
            }
            let lin = (i - 1) * ca.cols + (j - 1);
            if let Some(oldv) = ca.data.get(lin) {
                on_write(oldv, &rhs);
            }
            ca.data[lin] = allocate_cell_handle(rhs)?;
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
            .map_err(|e| map_cell_shape_error("cell deletion error", e))?,
    ))
}

pub fn assign_cell_paren_with_policy(
    ca: CellArray,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    if delete {
        if indices.len() != 1 {
            return Err(mex(
                "UnsupportedCellDeletion",
                "Linear cell deletion is only supported for vector indices",
            ));
        }
        if !is_empty_tensor(rhs) {
            return Err(mex(
                "DeletionRequiresEmptyRhs",
                "Cell deletion requires empty RHS",
            ));
        }
        return delete_cell_linear(ca, indices[0]);
    }
    if let Value::Cell(rhs_cell) = rhs {
        return assign_cell_paren_from_cell(ca, indices, rhs_cell);
    }
    Err(mex(
        "UnsupportedCellParenAssignment",
        "Cell paren assignment requires a cell RHS",
    ))
}

pub fn assign_cell_paren_linear_indices_with_policy(
    mut ca: CellArray,
    indices: &[usize],
    rhs: &Value,
    delete: bool,
) -> Result<Value, RuntimeError> {
    if delete {
        if !is_empty_tensor(rhs) {
            return Err(mex(
                "DeletionRequiresEmptyRhs",
                "Cell deletion requires empty RHS",
            ));
        }
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
                .map_err(|e| map_cell_shape_error("cell deletion error", e))?,
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
        ca.data[pos] = allocate_cell_handle(newv)?;
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
    ca.data[lin] = allocate_cell_handle(newv)?;
    Ok(Value::Cell(ca))
}

#[cfg(test)]
mod tests {
    use super::{assign_cell_member, expand_cell_indices, map_cell_shape_error};
    use runmat_builtins::{CellArray, StructValue, Tensor, Value};

    #[test]
    fn assign_cell_member_rejects_shape_mismatch_cell_rhs() {
        let base = CellArray::new(
            vec![
                Value::Struct(StructValue::new()),
                Value::Struct(StructValue::new()),
            ],
            1,
            2,
        )
        .expect("base cell");
        let rhs = CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("rhs cell");
        let err = assign_cell_member(base, "field".to_string(), Value::Cell(rhs), |_old, _new| {})
            .expect_err("shape mismatch should fail");
        assert_eq!(err.identifier(), Some("RunMat:CellMemberRhsShapeMismatch"));
    }

    #[test]
    fn cell_shape_error_mapping_reports_identifier() {
        let err = map_cell_shape_error("cell creation", "invalid shape");
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn expand_cell_indices_rejects_fractional_linear_index() {
        let cell = CellArray::new(vec![Value::Num(10.0), Value::Num(20.0)], 1, 2).expect("cell");
        let err = expand_cell_indices(&cell, &[Value::Num(1.5)])
            .expect_err("fractional index should fail");
        assert_eq!(err.identifier(), Some("RunMat:CellIndexType"));
    }

    #[test]
    fn expand_cell_indices_rejects_fractional_tensor_indices() {
        let cell = CellArray::new(vec![Value::Num(10.0), Value::Num(20.0)], 1, 2).expect("cell");
        let tensor = Tensor::new(vec![1.0, 1.25], vec![1, 2]).expect("tensor");
        let err = expand_cell_indices(&cell, &[Value::Tensor(tensor)])
            .expect_err("fractional tensor index should fail");
        assert_eq!(err.identifier(), Some("RunMat:CellIndexType"));
    }

    #[test]
    fn expand_cell_indices_accepts_scalar_tensor_subscripts_for_2d_cells() {
        let cell = CellArray::new(
            vec![
                Value::Num(11.0),
                Value::Num(12.0),
                Value::Num(21.0),
                Value::Num(22.0),
            ],
            2,
            2,
        )
        .expect("2d cell");
        let row = Tensor::new(vec![2.0], vec![1, 1]).expect("row scalar tensor");
        let col = Tensor::new(vec![1.0], vec![1, 1]).expect("col scalar tensor");
        let values = expand_cell_indices(&cell, &[Value::Tensor(row), Value::Tensor(col)])
            .expect("scalar tensor selectors should index 2d cell expansion");
        assert_eq!(values, vec![Value::Num(21.0)]);
    }

    #[test]
    fn expand_cell_indices_rejects_nonscalar_tensor_subscripts_for_2d_cells() {
        let cell = CellArray::new(
            vec![
                Value::Num(11.0),
                Value::Num(12.0),
                Value::Num(21.0),
                Value::Num(22.0),
            ],
            2,
            2,
        )
        .expect("2d cell");
        let row = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("non-scalar row tensor");
        let err = expand_cell_indices(&cell, &[Value::Tensor(row), Value::Num(1.0)])
            .expect_err("non-scalar tensor row selector should fail");
        assert_eq!(err.identifier(), Some("RunMat:CellIndexType"));
    }

    #[test]
    fn expand_cell_indices_supports_end_selectors_for_2d_cells() {
        let cell = CellArray::new(
            vec![
                Value::Num(11.0),
                Value::Num(12.0),
                Value::Num(21.0),
                Value::Num(22.0),
            ],
            2,
            2,
        )
        .expect("2d cell");
        let row_end = expand_cell_indices(&cell, &[Value::Num(-0.0), Value::Num(1.0)])
            .expect("row end selector should resolve");
        assert_eq!(row_end, vec![Value::Num(21.0)]);

        let col_end = expand_cell_indices(&cell, &[Value::Num(1.0), Value::Num(-0.0)])
            .expect("col end selector should resolve");
        assert_eq!(col_end, vec![Value::Num(12.0)]);
    }
}
