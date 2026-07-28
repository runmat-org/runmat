use crate::indexing::plan::{build_index_plan, IndexPlan};
use crate::indexing::selectors::{build_slice_selectors, SliceSelector};
use runmat_builtins::{
    ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, SparseTensor, StringArray,
    Tensor, Value,
};
use runmat_runtime::RuntimeError;
use std::collections::HashMap;

fn map_slice_shape_error(err: impl std::fmt::Display) -> RuntimeError {
    crate::interpreter::errors::mex(
        "ShapeMismatch",
        &format!("shape mismatch for slice result: {err}"),
    )
}

fn map_slice_acceleration_error(err: impl std::fmt::Display) -> RuntimeError {
    crate::interpreter::errors::mex("AccelerationOperationFailed", &format!("slice: {err}"))
}

fn integer_selection_value(
    tensor: &Tensor,
    indices: &[usize],
    output_shape: Vec<usize>,
) -> Result<Value, RuntimeError> {
    let storage = tensor
        .integer_storage()
        .expect("integer selection requires exact integer storage");
    macro_rules! select_integer_storage {
        ($values:expr, $storage:ident, $int:ident) => {{
            let selected: Vec<_> = indices.iter().map(|&index| $values[index]).collect();
            if let [value] = selected.as_slice() {
                Ok(Value::Int(IntValue::$int(*value)))
            } else {
                let tensor = Tensor::new_integer(IntegerStorage::$storage(selected), output_shape)
                    .map_err(map_slice_shape_error)?;
                Ok(Value::Tensor(tensor))
            }
        }};
    }

    match storage {
        IntegerStorage::I8(values) => select_integer_storage!(values, I8, I8),
        IntegerStorage::I16(values) => select_integer_storage!(values, I16, I16),
        IntegerStorage::I32(values) => select_integer_storage!(values, I32, I32),
        IntegerStorage::I64(values) => select_integer_storage!(values, I64, I64),
        IntegerStorage::U8(values) => select_integer_storage!(values, U8, U8),
        IntegerStorage::U16(values) => select_integer_storage!(values, U16, U16),
        IntegerStorage::U32(values) => select_integer_storage!(values, U32, U32),
        IntegerStorage::U64(values) => select_integer_storage!(values, U64, U64),
    }
}

pub async fn read_tensor_slice_1d(
    tensor: &Tensor,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    read_tensor_slice_nd(tensor, 1, colon_mask, end_mask, numeric).await
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
            if tensor.integer_storage().is_some() {
                let indices: Vec<usize> = (start..start + rows).collect();
                integer_selection_value(tensor, &indices, vec![rows, 1]).map(Some)
            } else {
                let out = tensor.data[start..start + rows].to_vec();
                if out.len() == 1 {
                    Ok(Some(Value::Num(out[0])))
                } else {
                    let tens = Tensor::new(out, vec![rows, 1]).map_err(map_slice_shape_error)?;
                    Ok(Some(Value::Tensor(tens)))
                }
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
            if tensor.integer_storage().is_some() {
                let indices: Vec<usize> = (0..cols).map(|col| i0 + col * rows).collect();
                integer_selection_value(tensor, &indices, vec![1, cols]).map(Some)
            } else {
                let mut out: Vec<f64> = Vec::with_capacity(cols);
                for col in 0..cols {
                    out.push(tensor.data[i0 + col * rows]);
                }
                if out.len() == 1 {
                    Ok(Some(Value::Num(out[0])))
                } else {
                    let tens = Tensor::new(out, vec![1, cols]).map_err(map_slice_shape_error)?;
                    Ok(Some(Value::Tensor(tens)))
                }
            }
        }
        (SliceSelector::Colon, SliceSelector::Indices(js)) => {
            if js.is_empty() {
                if tensor.integer_storage().is_some() {
                    integer_selection_value(tensor, &[], vec![rows, 0]).map(Some)
                } else {
                    let tens =
                        Tensor::new(Vec::new(), vec![rows, 0]).map_err(map_slice_shape_error)?;
                    Ok(Some(Value::Tensor(tens)))
                }
            } else if tensor.integer_storage().is_some() {
                let mut indices = Vec::with_capacity(rows * js.len());
                for &j in js {
                    let j0 = j - 1;
                    if j0 >= cols {
                        return Err(crate::interpreter::errors::mex(
                            "IndexOutOfBounds",
                            "Index out of bounds",
                        ));
                    }
                    let start = j0 * rows;
                    indices.extend(start..start + rows);
                }
                integer_selection_value(tensor, &indices, vec![rows, js.len()]).map(Some)
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
                let tens = Tensor::new(out, vec![rows, js.len()]).map_err(map_slice_shape_error)?;
                Ok(Some(Value::Tensor(tens)))
            }
        }
        (SliceSelector::Indices(is), SliceSelector::Colon) => {
            if is.is_empty() {
                if tensor.integer_storage().is_some() {
                    integer_selection_value(tensor, &[], vec![0, cols]).map(Some)
                } else {
                    let tens =
                        Tensor::new(Vec::new(), vec![0, cols]).map_err(map_slice_shape_error)?;
                    Ok(Some(Value::Tensor(tens)))
                }
            } else if tensor.integer_storage().is_some() {
                let mut indices = Vec::with_capacity(is.len() * cols);
                for col in 0..cols {
                    for &i in is {
                        let i0 = i - 1;
                        if i0 >= rows {
                            return Err(crate::interpreter::errors::mex(
                                "IndexOutOfBounds",
                                "Index out of bounds",
                            ));
                        }
                        indices.push(i0 + col * rows);
                    }
                }
                integer_selection_value(tensor, &indices, vec![is.len(), cols]).map(Some)
            } else {
                let mut out: Vec<f64> = Vec::with_capacity(is.len() * cols);
                for col in 0..cols {
                    for &i in is {
                        let i0 = i - 1;
                        if i0 >= rows {
                            return Err(crate::interpreter::errors::mex(
                                "IndexOutOfBounds",
                                "Index out of bounds",
                            ));
                        }
                        out.push(tensor.data[i0 + col * rows]);
                    }
                }
                let tens = Tensor::new(out, vec![is.len(), cols]).map_err(map_slice_shape_error)?;
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
    if tensor.integer_storage().is_some() {
        let indices: Vec<usize> = plan.indices.iter().map(|&index| index as usize).collect();
        integer_selection_value(tensor, &indices, plan.output_shape)
    } else if plan.indices.is_empty() {
        let out_tensor =
            Tensor::new(Vec::new(), plan.output_shape).map_err(map_slice_shape_error)?;
        Ok(Value::Tensor(out_tensor))
    } else {
        let mut out_data: Vec<f64> = Vec::with_capacity(plan.indices.len());
        for &lin in &plan.indices {
            out_data.push(tensor.data[lin as usize]);
        }
        if out_data.len() == 1 {
            Ok(Value::Num(out_data[0]))
        } else {
            let out_tensor =
                Tensor::new(out_data, plan.output_shape).map_err(map_slice_shape_error)?;
            Ok(Value::Tensor(out_tensor))
        }
    }
}

pub fn read_tensor_slice_from_plan(
    tensor: &Tensor,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    if tensor.integer_storage().is_some() {
        let indices: Vec<usize> = plan.indices.iter().map(|&index| index as usize).collect();
        integer_selection_value(tensor, &indices, plan.output_shape.clone())
    } else if plan.indices.is_empty() {
        let out_tensor =
            Tensor::new(Vec::new(), plan.output_shape.clone()).map_err(map_slice_shape_error)?;
        Ok(Value::Tensor(out_tensor))
    } else {
        let mut out_data: Vec<f64> = Vec::with_capacity(plan.indices.len());
        for &lin in &plan.indices {
            out_data.push(tensor.data[lin as usize]);
        }
        if out_data.len() == 1 {
            Ok(Value::Num(out_data[0]))
        } else {
            let out_tensor =
                Tensor::new(out_data, plan.output_shape.clone()).map_err(map_slice_shape_error)?;
            Ok(Value::Tensor(out_tensor))
        }
    }
}

fn sparse_output_shape(plan: &IndexPlan) -> Result<(usize, usize), RuntimeError> {
    match plan.output_shape.as_slice() {
        [rows, cols] => Ok((*rows, *cols)),
        [len] => Ok((*len, 1)),
        _ => Err(crate::interpreter::errors::mex(
            "UnsupportedSparseIndexRank",
            "Sparse indexing currently supports two-dimensional outputs",
        )),
    }
}

fn sparse_scalar_value(
    sparse: &SparseTensor,
    row: usize,
    col: usize,
) -> Result<Value, RuntimeError> {
    if let Some(storage) = sparse.integer_storage() {
        let scalar = match sparse.integer_at(row, col) {
            Some(value) => {
                SparseTensor::new_integer_like(1, 1, vec![0, 1], vec![0], vec![value], storage)
            }
            None => Ok(SparseTensor::zeros_with_integer_storage(1, 1, storage)),
        }
        .map_err(map_slice_shape_error)?;
        return Ok(Value::SparseTensor(scalar));
    }

    let value = sparse.get(row, col).unwrap_or(0.0);
    if value == 0.0 {
        return Ok(Value::SparseTensor(SparseTensor::zeros(1, 1)));
    }
    let sparse =
        SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![value]).map_err(map_slice_shape_error)?;
    Ok(Value::SparseTensor(sparse))
}

fn checked_sparse_numel(sparse: &SparseTensor) -> Result<usize, RuntimeError> {
    sparse.rows.checked_mul(sparse.cols).ok_or_else(|| {
        crate::interpreter::errors::mex("IndexOutOfBounds", "Sparse dimensions overflow")
    })
}

fn sparse_zeros_like(sparse: &SparseTensor, rows: usize, cols: usize) -> SparseTensor {
    sparse.integer_storage().map_or_else(
        || SparseTensor::zeros(rows, cols),
        |storage| SparseTensor::zeros_with_integer_storage(rows, cols, storage),
    )
}

fn typed_sparse_from_column_entries(
    rows: usize,
    cols: usize,
    mut col_entries: Vec<Vec<(usize, IntValue)>>,
    prototype: &IntegerStorage,
) -> Result<Value, RuntimeError> {
    let mut col_ptrs = Vec::with_capacity(cols.saturating_add(1));
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for entries in col_entries.iter_mut().take(cols) {
        entries.sort_by_key(|(row, _)| *row);
        for (row, value) in entries.drain(..) {
            if !value.is_zero() {
                row_indices.push(row);
                values.push(value);
            }
        }
        col_ptrs.push(values.len());
    }
    let sparse =
        SparseTensor::new_integer_like(rows, cols, col_ptrs, row_indices, values, prototype)
            .map_err(map_slice_shape_error)?;
    Ok(Value::SparseTensor(sparse))
}

fn linear_sparse_slice(
    sparse: &SparseTensor,
    selector: &SliceSelector,
) -> Result<Value, RuntimeError> {
    let total = checked_sparse_numel(sparse)?;
    let base_is_row_vector = sparse.rows == 1 && sparse.cols > 1;
    if matches!(selector, SliceSelector::Colon) {
        let mut row_indices = Vec::with_capacity(sparse.values.len());
        for col in 0..sparse.cols {
            for entry in sparse.col_ptrs[col]..sparse.col_ptrs[col + 1] {
                row_indices.push(sparse.row_indices[entry] + col * sparse.rows);
            }
        }
        let sparse = if let Some(storage) = sparse.integer_storage() {
            SparseTensor::new_integer(
                total,
                1,
                vec![0, sparse.nnz()],
                row_indices,
                storage.clone(),
            )
        } else {
            SparseTensor::new(
                total,
                1,
                vec![0, sparse.values.len()],
                row_indices,
                sparse.values.clone(),
            )
        }
        .map_err(map_slice_shape_error)?;
        return Ok(Value::SparseTensor(sparse));
    }
    let (indices, output_shape) = match selector {
        SliceSelector::Colon => unreachable!("colon sparse linear slices return early"),
        SliceSelector::Scalar(index) => (vec![*index], vec![1, 1]),
        SliceSelector::Indices(indices) => {
            let shape = if indices.is_empty() {
                vec![0, 1]
            } else if indices.len() == 1 {
                vec![1, 1]
            } else if base_is_row_vector {
                vec![1, indices.len()]
            } else {
                vec![indices.len(), 1]
            };
            (indices.clone(), shape)
        }
        SliceSelector::LinearIndices {
            values,
            output_shape,
        } => (values.clone(), output_shape.clone()),
    };
    if indices.iter().any(|&index| index == 0 || index > total) {
        return Err(crate::interpreter::errors::mex(
            "IndexOutOfBounds",
            "Index out of bounds",
        ));
    }
    if indices.len() == 1 {
        let lin = indices[0] - 1;
        let row = lin % sparse.rows;
        let col = lin / sparse.rows;
        return sparse_scalar_value(sparse, row, col);
    }
    let (out_rows, out_cols) = match output_shape.as_slice() {
        [rows, cols] => (*rows, *cols),
        [len] => (*len, 1),
        _ => {
            return Err(crate::interpreter::errors::mex(
                "UnsupportedSparseIndexRank",
                "Sparse indexing currently supports two-dimensional outputs",
            ))
        }
    };
    if indices.is_empty() {
        return Ok(Value::SparseTensor(sparse_zeros_like(
            sparse, out_rows, out_cols,
        )));
    }

    if let Some(storage) = sparse.integer_storage() {
        let mut col_entries: Vec<Vec<(usize, IntValue)>> = vec![Vec::new(); out_cols];
        for (out_pos, &index) in indices.iter().enumerate() {
            let base_lin = index - 1;
            let base_row = base_lin % sparse.rows;
            let base_col = base_lin / sparse.rows;
            if let Some(value) = sparse.integer_at(base_row, base_col) {
                let out_row = out_pos % out_rows;
                let out_col = out_pos / out_rows;
                col_entries[out_col].push((out_row, value));
            }
        }
        return typed_sparse_from_column_entries(out_rows, out_cols, col_entries, storage);
    }

    let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); out_cols];
    for (out_pos, &index) in indices.iter().enumerate() {
        let base_lin = index - 1;
        let base_row = base_lin % sparse.rows;
        let base_col = base_lin / sparse.rows;
        if let Some(value) = sparse.get(base_row, base_col) {
            if value != 0.0 {
                let out_row = out_pos % out_rows;
                let out_col = out_pos / out_rows;
                col_entries[out_col].push((out_row, value));
            }
        }
    }
    sparse_from_column_entries(out_rows, out_cols, col_entries)
}

fn selector_indices(selector: &SliceSelector, dim_len: usize) -> Vec<usize> {
    match selector {
        SliceSelector::Colon => (1..=dim_len).collect(),
        SliceSelector::Scalar(index) => vec![*index],
        SliceSelector::Indices(indices)
        | SliceSelector::LinearIndices {
            values: indices, ..
        } => indices.clone(),
    }
}

fn sparse_from_column_entries(
    rows: usize,
    cols: usize,
    mut col_entries: Vec<Vec<(usize, f64)>>,
) -> Result<Value, RuntimeError> {
    let mut col_ptrs = Vec::with_capacity(cols.saturating_add(1));
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for entries in col_entries.iter_mut().take(cols) {
        entries.sort_by_key(|(row, _)| *row);
        for &(row, value) in entries.iter() {
            if value != 0.0 {
                row_indices.push(row);
                values.push(value);
            }
        }
        col_ptrs.push(values.len());
    }
    let sparse = SparseTensor::new(rows, cols, col_ptrs, row_indices, values)
        .map_err(map_slice_shape_error)?;
    Ok(Value::SparseTensor(sparse))
}

fn matrix_sparse_slice(
    sparse: &SparseTensor,
    selectors: &[SliceSelector],
) -> Result<Value, RuntimeError> {
    let row_selector = selectors.first().unwrap_or(&SliceSelector::Colon);
    let col_selector = selectors.get(1).unwrap_or(&SliceSelector::Colon);
    let all_rows = matches!(row_selector, SliceSelector::Colon);
    let rows = if all_rows {
        Vec::new()
    } else {
        selector_indices(row_selector, sparse.rows)
    };
    let cols = selector_indices(col_selector, sparse.cols);
    if (!all_rows && rows.iter().any(|&row| row == 0 || row > sparse.rows))
        || cols.iter().any(|&col| col == 0 || col > sparse.cols)
    {
        return Err(crate::interpreter::errors::mex(
            "IndexOutOfBounds",
            "Index out of bounds",
        ));
    }
    let out_rows = if all_rows { sparse.rows } else { rows.len() };
    let out_cols = cols.len();
    if out_rows == 0 || out_cols == 0 {
        return Ok(Value::SparseTensor(sparse_zeros_like(
            sparse, out_rows, out_cols,
        )));
    }

    let mut row_positions: HashMap<usize, Vec<usize>> = HashMap::new();
    if !all_rows {
        for (out_row, &row) in rows.iter().enumerate() {
            row_positions.entry(row - 1).or_default().push(out_row);
        }
    }
    if let Some(storage) = sparse.integer_storage() {
        let mut col_entries: Vec<Vec<(usize, IntValue)>> = vec![Vec::new(); out_cols];
        for (out_col, &col) in cols.iter().enumerate() {
            let base_col = col - 1;
            for entry in sparse.col_ptrs[base_col]..sparse.col_ptrs[base_col + 1] {
                let base_row = sparse.row_indices[entry];
                let value = storage
                    .value_at(entry)
                    .expect("typed sparse entry is present");
                if all_rows {
                    col_entries[out_col].push((base_row, value));
                } else if let Some(output_rows) = row_positions.get(&base_row) {
                    for &out_row in output_rows {
                        col_entries[out_col].push((out_row, value.clone()));
                    }
                }
            }
        }
        if out_rows == 1 && out_cols == 1 {
            let value = col_entries
                .first()
                .and_then(|entries| entries.first())
                .map(|(_, value)| value.clone());
            let scalar = match value {
                Some(value) => {
                    SparseTensor::new_integer_like(1, 1, vec![0, 1], vec![0], vec![value], storage)
                }
                None => Ok(SparseTensor::zeros_with_integer_storage(1, 1, storage)),
            }
            .map_err(map_slice_shape_error)?;
            return Ok(Value::SparseTensor(scalar));
        }
        return typed_sparse_from_column_entries(out_rows, out_cols, col_entries, storage);
    }

    let mut col_entries = vec![Vec::new(); out_cols];
    for (out_col, &col) in cols.iter().enumerate() {
        let base_col = col - 1;
        for entry in sparse.col_ptrs[base_col]..sparse.col_ptrs[base_col + 1] {
            let base_row = sparse.row_indices[entry];
            let value = sparse.values[entry];
            if all_rows {
                col_entries[out_col].push((base_row, value));
            } else if let Some(output_rows) = row_positions.get(&base_row) {
                for &out_row in output_rows {
                    col_entries[out_col].push((out_row, value));
                }
            }
        }
    }
    if out_rows == 1 && out_cols == 1 {
        let value = col_entries
            .first()
            .and_then(|entries| entries.first())
            .map(|(_, value)| *value)
            .unwrap_or(0.0);
        if value == 0.0 {
            return Ok(Value::SparseTensor(SparseTensor::zeros(1, 1)));
        }
        let scalar = SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![value])
            .map_err(map_slice_shape_error)?;
        return Ok(Value::SparseTensor(scalar));
    }
    sparse_from_column_entries(out_rows, out_cols, col_entries)
}

pub async fn read_sparse_slice(
    sparse: &SparseTensor,
    dims: usize,
    colon_mask: u32,
    end_mask: u32,
    numeric: &[Value],
) -> Result<Value, RuntimeError> {
    let selectors =
        build_slice_selectors(dims, colon_mask, end_mask, numeric, &sparse.shape()).await?;
    match dims {
        1 => linear_sparse_slice(
            sparse,
            selectors
                .first()
                .unwrap_or(&SliceSelector::Indices(Vec::new())),
        ),
        2 => matrix_sparse_slice(sparse, &selectors),
        _ => {
            let plan = build_index_plan(&selectors, dims, &sparse.shape())?;
            read_sparse_slice_from_plan(sparse, &plan)
        }
    }
}

pub fn read_sparse_slice_from_plan(
    sparse: &SparseTensor,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    if plan.indices.len() == 1 {
        let lin = plan.indices[0] as usize;
        if sparse.rows == 0 || lin >= sparse.rows.saturating_mul(sparse.cols) {
            return Err(crate::interpreter::errors::mex(
                "IndexOutOfBounds",
                "Index out of bounds",
            ));
        }
        let row = lin % sparse.rows;
        let col = lin / sparse.rows;
        return sparse_scalar_value(sparse, row, col);
    }

    let (out_rows, out_cols) = sparse_output_shape(plan)?;
    if plan.indices.is_empty() {
        return Ok(Value::SparseTensor(sparse_zeros_like(
            sparse, out_rows, out_cols,
        )));
    }

    let total = sparse.rows.checked_mul(sparse.cols).ok_or_else(|| {
        crate::interpreter::errors::mex("IndexOutOfBounds", "Sparse dimensions overflow")
    })?;
    let mut col_ptrs = Vec::with_capacity(out_cols.saturating_add(1));
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    if let Some(storage) = sparse.integer_storage() {
        let mut integer_values = Vec::new();
        for out_col in 0..out_cols {
            for out_row in 0..out_rows {
                let out_lin = out_row + out_col * out_rows;
                let Some(&base_lin) = plan.indices.get(out_lin) else {
                    return Err(crate::interpreter::errors::mex(
                        "ShapeMismatch",
                        "sparse slice plan output shape does not match selected indices",
                    ));
                };
                let base_lin = base_lin as usize;
                if sparse.rows == 0 || base_lin >= total {
                    return Err(crate::interpreter::errors::mex(
                        "IndexOutOfBounds",
                        "Index out of bounds",
                    ));
                }
                let base_row = base_lin % sparse.rows;
                let base_col = base_lin / sparse.rows;
                if let Some(value) = sparse.integer_at(base_row, base_col) {
                    row_indices.push(out_row);
                    integer_values.push(value);
                }
            }
            col_ptrs.push(integer_values.len());
        }
        let out = SparseTensor::new_integer_like(
            out_rows,
            out_cols,
            col_ptrs,
            row_indices,
            integer_values,
            storage,
        )
        .map_err(map_slice_shape_error)?;
        return Ok(Value::SparseTensor(out));
    }
    for out_col in 0..out_cols {
        for out_row in 0..out_rows {
            let out_lin = out_row + out_col * out_rows;
            let Some(&base_lin) = plan.indices.get(out_lin) else {
                return Err(crate::interpreter::errors::mex(
                    "ShapeMismatch",
                    "sparse slice plan output shape does not match selected indices",
                ));
            };
            let base_lin = base_lin as usize;
            if sparse.rows == 0 || base_lin >= total {
                return Err(crate::interpreter::errors::mex(
                    "IndexOutOfBounds",
                    "Index out of bounds",
                ));
            }
            let base_row = base_lin % sparse.rows;
            let base_col = base_lin / sparse.rows;
            if let Some(value) = sparse.get(base_row, base_col) {
                if value != 0.0 {
                    row_indices.push(out_row);
                    values.push(value);
                }
            }
        }
        col_ptrs.push(values.len());
    }

    let out = SparseTensor::new(out_rows, out_cols, col_ptrs, row_indices, values)
        .map_err(map_slice_shape_error)?;
    Ok(Value::SparseTensor(out))
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
    if let Some(storage) = tensor.integer_data.as_ref() {
        return read_integer_complex_slice_from_plan(storage, plan);
    }
    if plan.indices.is_empty() {
        let empty = ComplexTensor::new(Vec::new(), plan.output_shape.clone())
            .map_err(map_slice_shape_error)?;
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
    let out_ct =
        ComplexTensor::new(out, plan.output_shape.clone()).map_err(map_slice_shape_error)?;
    Ok(Value::ComplexTensor(out_ct))
}

fn read_integer_complex_slice_from_plan(
    storage: &IntegerComplexStorage,
    plan: &IndexPlan,
) -> Result<Value, RuntimeError> {
    let mut real_values = Vec::with_capacity(plan.indices.len());
    let mut imag_values = Vec::with_capacity(plan.indices.len());
    for &linear_index in &plan.indices {
        let index = linear_index as usize;
        let real = storage.real.value_at(index).ok_or_else(|| {
            crate::interpreter::errors::mex(
                "IndexOutOfBounds",
                "Slice error: complex index out of bounds",
            )
        })?;
        let imag = storage.imag.value_at(index).ok_or_else(|| {
            crate::interpreter::errors::mex(
                "IndexOutOfBounds",
                "Slice error: complex index out of bounds",
            )
        })?;
        real_values.push(real);
        imag_values.push(imag);
    }

    let real = storage
        .real
        .from_same_class_values(real_values)
        .map_err(map_slice_shape_error)?;
    let imag = storage
        .imag
        .from_same_class_values(imag_values)
        .map_err(map_slice_shape_error)?;
    let result = ComplexTensor::new_integer(
        IntegerComplexStorage::new(real, imag).map_err(map_slice_shape_error)?,
        plan.output_shape.clone(),
    )
    .map_err(map_slice_shape_error)?;
    Ok(Value::ComplexTensor(result))
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
        if let Some(integer_type) = runmat_accelerate_api::handle_integer_type(handle) {
            return upload_empty_integer_gpu_slice(provider, integer_type, &plan.output_shape);
        }
        let zeros = provider
            .zeros(&plan.output_shape)
            .map_err(map_slice_acceleration_error)?;
        Ok(Value::GpuTensor(zeros))
    } else {
        let result = provider
            .gather_linear(handle, &plan.indices, &plan.output_shape)
            .map_err(map_slice_acceleration_error)?;
        Ok(Value::GpuTensor(result))
    }
}

fn upload_empty_integer_gpu_slice(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    integer_type: runmat_accelerate_api::IntegerElementType,
    output_shape: &[usize],
) -> Result<Value, RuntimeError> {
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView, IntegerElementType};

    let data = match integer_type {
        IntegerElementType::I8 => HostIntegerDataView::I8(&[]),
        IntegerElementType::I16 => HostIntegerDataView::I16(&[]),
        IntegerElementType::I32 => HostIntegerDataView::I32(&[]),
        IntegerElementType::I64 => HostIntegerDataView::I64(&[]),
        IntegerElementType::U8 => HostIntegerDataView::U8(&[]),
        IntegerElementType::U16 => HostIntegerDataView::U16(&[]),
        IntegerElementType::U32 => HostIntegerDataView::U32(&[]),
        IntegerElementType::U64 => HostIntegerDataView::U64(&[]),
    };
    provider
        .upload_integer(&HostIntegerTensorView {
            data,
            shape: output_shape,
        })
        .map(Value::GpuTensor)
        .map_err(map_slice_acceleration_error)
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
            .map_err(map_slice_shape_error)?;
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
    let out_sa = StringArray::new(out, plan.output_shape.clone()).map_err(map_slice_shape_error)?;
    Ok(Value::StringArray(out_sa))
}

#[cfg(test)]
mod tests {
    use super::{
        gather_string_slice, map_slice_acceleration_error, read_complex_slice_from_plan,
        read_gpu_slice_from_plan, read_sparse_slice_from_plan, read_string_slice,
        read_tensor_slice_from_plan, try_tensor_slice_2d_fast_path,
    };
    use crate::indexing::plan::IndexPlan;
    use crate::indexing::selectors::SliceSelector;
    use futures::executor::block_on;
    use runmat_builtins::{
        ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, SparseTensor, StringArray,
        Tensor, Value,
    };

    #[test]
    fn tensor_slice_plan_preserves_exact_uint64_storage() {
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![1, u64::MAX, 3, 4]), vec![2, 2])
            .expect("tensor");
        let plan = IndexPlan::new(vec![0, 2, 1, 3], vec![2, 2], vec![2, 2], 2, vec![2, 2]);
        let result = read_tensor_slice_from_plan(&tensor, &plan).expect("slice");

        let Value::Tensor(output) = result else {
            panic!("expected tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![1, 3, u64::MAX, 4]))
        );
    }

    #[test]
    fn sparse_slice_plan_preserves_exact_uint64_storage_and_empty_class() {
        let sparse = SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 2],
            vec![0, 1],
            IntegerStorage::U64(vec![1, u64::MAX]),
        )
        .expect("sparse");
        let plan = IndexPlan::new(vec![0, 3, 1, 2], vec![2, 2], vec![2, 2], 2, vec![2, 2]);
        let result = read_sparse_slice_from_plan(&sparse, &plan).expect("slice");
        let Value::SparseTensor(output) = result else {
            panic!("expected sparse output");
        };
        assert_eq!(output.shape(), vec![2, 2]);
        assert_eq!(output.col_ptrs, vec![0, 2, 2]);
        assert_eq!(output.row_indices, vec![0, 1]);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![1, u64::MAX]))
        );

        let empty_plan = IndexPlan::new(Vec::new(), vec![0, 1], vec![0], 1, vec![2, 2]);
        let empty = read_sparse_slice_from_plan(&sparse, &empty_plan).expect("empty slice");
        let Value::SparseTensor(empty) = empty else {
            panic!("expected sparse output");
        };
        assert_eq!(empty.shape(), vec![0, 1]);
        assert_eq!(empty.integer_storage(), Some(&IntegerStorage::U64(vec![])));
    }

    #[test]
    fn tensor_slice_fast_path_returns_exact_integer_scalar() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I64(vec![i64::MAX]), vec![1, 1]).expect("tensor");
        let result = try_tensor_slice_2d_fast_path(
            &tensor,
            2,
            &[SliceSelector::Colon, SliceSelector::Scalar(1)],
        )
        .expect("fast path")
        .expect("fast path result");
        assert_eq!(result, Value::Int(IntValue::I64(i64::MAX)));
    }

    #[test]
    fn gpu_integer_slice_preserves_class_for_empty_and_nonempty_plans() {
        runmat_accelerate_api::set_thread_provider(None);
        runmat_accelerate_api::clear_provider();
        runmat_accelerate::simple_provider::register_inprocess_provider();
        let provider = runmat_accelerate_api::provider().expect("test provider");
        let _thread_provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));

        {
            let source = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U64(&[
                        0,
                        1_u64 << 63,
                        u64::MAX,
                    ]),
                    shape: &[1, 3],
                })
                .expect("upload integer gpu source");

            let nonempty = IndexPlan::new(vec![2, 1], vec![1, 2], vec![2], 1, vec![1, 3]);
            let Value::GpuTensor(gathered) =
                read_gpu_slice_from_plan(&source, &nonempty).expect("gpu integer gather")
            else {
                panic!("expected gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&gathered),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            let host = block_on(provider.download_integer(&gathered)).expect("download gathered");
            assert_eq!(host.shape, vec![1, 2]);
            assert_eq!(
                host.data,
                runmat_accelerate_api::HostIntegerDataOwned::U64(vec![u64::MAX, 1_u64 << 63])
            );

            let empty = IndexPlan::new(Vec::new(), vec![1, 0], vec![0], 1, vec![1, 3]);
            let Value::GpuTensor(empty_handle) =
                read_gpu_slice_from_plan(&source, &empty).expect("empty gpu integer slice")
            else {
                panic!("expected gpu tensor");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&empty_handle),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            let host = block_on(provider.download_integer(&empty_handle)).expect("download empty");
            assert_eq!(host.shape, vec![1, 0]);
            assert_eq!(
                host.data,
                runmat_accelerate_api::HostIntegerDataOwned::U64(Vec::new())
            );
        }
    }

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

    #[test]
    fn tensor_slice_plan_shape_mismatch_reports_identifier() {
        let tensor = Tensor::new(vec![10.0, 20.0], vec![1, 2]).expect("tensor");
        let plan = IndexPlan::new(vec![0, 1], vec![1, 1], vec![2], 1, vec![1, 2]);
        let err = read_tensor_slice_from_plan(&tensor, &plan)
            .expect_err("shape-mismatch plan should fail");
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn string_slice_plan_shape_mismatch_reports_identifier() {
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
        let plan = IndexPlan::new(vec![0, 1], vec![1, 1], vec![2], 1, vec![2, 2]);
        let err = gather_string_slice(&sa, &plan).expect_err("shape-mismatch plan should fail");
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn complex_slice_plan_shape_mismatch_reports_identifier() {
        let ct = ComplexTensor::new(vec![(1.0, 0.0), (2.0, 0.0)], vec![1, 2]).expect("complex");
        let plan = IndexPlan::new(vec![0, 1], vec![1, 1], vec![2], 1, vec![1, 2]);
        let err =
            read_complex_slice_from_plan(&ct, &plan).expect_err("shape-mismatch plan should fail");
        assert_eq!(err.identifier(), Some("RunMat:ShapeMismatch"));
    }

    #[test]
    fn integer_complex_slice_preserves_exact_reordered_and_empty_components() {
        let complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![1, 9_223_372_036_854_775_809, u64::MAX]),
                IntegerStorage::U64(vec![7, 8, 9]),
            )
            .unwrap(),
            vec![1, 3],
        )
        .unwrap();
        let reordered = IndexPlan::new(vec![2, 0], vec![1, 2], vec![2], 1, vec![1, 3]);
        let Value::ComplexTensor(result) =
            read_complex_slice_from_plan(&complex, &reordered).expect("typed complex slice")
        else {
            panic!("typed complex selection must remain a complex tensor");
        };
        assert_eq!(result.shape, vec![1, 2]);
        assert_eq!(
            result.integer_data,
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![u64::MAX, 1]),
                    IntegerStorage::U64(vec![9, 7]),
                )
                .unwrap()
            )
        );

        let scalar = IndexPlan::new(vec![1], vec![1, 1], vec![1], 1, vec![1, 3]);
        let Value::ComplexTensor(result) =
            read_complex_slice_from_plan(&complex, &scalar).expect("typed scalar selection")
        else {
            panic!("typed complex scalar must retain exact complex storage");
        };
        assert_eq!(
            result.integer_data,
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![9_223_372_036_854_775_809]),
                    IntegerStorage::U64(vec![8]),
                )
                .unwrap()
            )
        );

        let empty = IndexPlan::new(Vec::new(), vec![0, 1], vec![0], 1, vec![1, 3]);
        let Value::ComplexTensor(result) =
            read_complex_slice_from_plan(&complex, &empty).expect("empty typed selection")
        else {
            panic!("empty typed complex selection must retain its class");
        };
        assert_eq!(
            result.integer_data,
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(Vec::new()),
                    IntegerStorage::U64(Vec::new())
                )
                .unwrap()
            )
        );
    }

    #[test]
    fn slice_acceleration_error_mapping_reports_identifier() {
        let err = map_slice_acceleration_error("provider failed");
        assert_eq!(err.identifier(), Some("RunMat:AccelerationOperationFailed"));
    }
}
