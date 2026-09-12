use super::detection::{any_missing, logical_mask_for_rows};
use super::numeric::{is_missing_text, scalar_text, scalar_usize};
use super::*;

#[derive(Clone, Copy)]
pub(super) enum RemoveDim {
    Auto,
    Rows,
    Columns,
}

#[derive(Clone, Copy)]
pub(super) struct RemoveOptions {
    dim: RemoveDim,
}

impl RemoveOptions {
    pub(super) fn parse(args: &[Value]) -> BuiltinResult<Self> {
        let mut dim = RemoveDim::Auto;
        let mut idx = 0;
        while idx < args.len() {
            if let Some(text) = scalar_text(&args[idx]) {
                match text.to_ascii_lowercase().as_str() {
                    "dim" if idx + 1 < args.len() => {
                        dim = dim_from_value(&args[idx + 1])?;
                        idx += 2;
                        continue;
                    }
                    "dim" => return Err(invalid_argument("rmmissing: 'dim' requires a value")),
                    "rows" => {
                        dim = RemoveDim::Rows;
                        idx += 1;
                        continue;
                    }
                    "columns" | "cols" => {
                        dim = RemoveDim::Columns;
                        idx += 1;
                        continue;
                    }
                    other => {
                        return Err(invalid_argument(format!(
                            "rmmissing: unsupported option '{other}'"
                        )))
                    }
                }
            }
            if matches!(args[idx], Value::Num(_) | Value::Int(_)) {
                dim = dim_from_value(&args[idx])?;
                idx += 1;
                continue;
            }
            return Err(invalid_argument(format!(
                "rmmissing: unsupported option argument {:?}",
                args[idx]
            )));
        }
        Ok(Self { dim })
    }
}

pub(super) fn dim_from_value(value: &Value) -> BuiltinResult<RemoveDim> {
    match scalar_usize(value, "dimension")? {
        1 => Ok(RemoveDim::Rows),
        2 => Ok(RemoveDim::Columns),
        _ => Err(invalid_argument("dimension must be 1 or 2 for rmmissing")),
    }
}

pub(super) fn remove_missing_value(
    value: Value,
    options: RemoveOptions,
) -> BuiltinResult<(Value, LogicalArray)> {
    match value {
        Value::Object(object) if is_tabular_object(&object) => {
            remove_missing_table(object, options)
        }
        Value::Tensor(tensor) => remove_missing_tensor(tensor, options),
        Value::StringArray(array) => remove_missing_string_array(array, options),
        Value::LogicalArray(array) => remove_missing_logical_array(array, options),
        Value::Cell(cell) => remove_missing_cell(cell, options),
        other => {
            if any_missing(&other)? {
                Ok((
                    empty_like(other)?,
                    LogicalArray::new(vec![1], vec![1, 1]).map_err(internal_error)?,
                ))
            } else {
                Ok((
                    other,
                    LogicalArray::new(vec![0], vec![1, 1]).map_err(internal_error)?,
                ))
            }
        }
    }
}

pub(super) fn remove_missing_table(
    object: ObjectInstance,
    options: RemoveOptions,
) -> BuiltinResult<(Value, LogicalArray)> {
    let height = table_height(&object)?;
    let width = table_width(&object)?;
    let names = table_variable_names_from_object(&object)?;
    let variables = table_variables(&object)?;
    let remove_columns = matches!(options.dim, RemoveDim::Columns);
    if remove_columns {
        let mut keep_names = Vec::new();
        let mut removed = Vec::with_capacity(width);
        let mut keep_values = Vec::new();
        for name in &names {
            let value = variables
                .fields
                .get(name)
                .ok_or_else(|| internal_error(format!("table missing variable {name}")))?;
            let has_missing = any_missing(value)?;
            removed.push(u8::from(has_missing));
            if !has_missing {
                keep_names.push(name.clone());
                keep_values.push(value.clone());
            }
        }
        let row_names = selected_row_names(&object, &(0..height).collect::<Vec<_>>())?;
        Ok((
            table_from_columns_like(&object, keep_names, keep_values, row_names, None)?,
            LogicalArray::new(removed, vec![1, width]).map_err(internal_error)?,
        ))
    } else {
        let mut row_has_missing = vec![0u8; height];
        for name in &names {
            if let Some(value) = variables.fields.get(name) {
                let mask = logical_mask_for_rows(value, height)?;
                for row in 0..height {
                    if mask[row] != 0 {
                        row_has_missing[row] = 1;
                    }
                }
            }
        }
        let keep_rows: Vec<usize> = row_has_missing
            .iter()
            .enumerate()
            .filter_map(|(idx, flag)| (*flag == 0).then_some(idx))
            .collect();
        let mut values = Vec::with_capacity(names.len());
        for name in &names {
            let value = variables
                .fields
                .get(name)
                .ok_or_else(|| internal_error(format!("table missing variable {name}")))?;
            values.push(select_rows(value, &keep_rows)?);
        }
        let row_names = selected_row_names(&object, &keep_rows)?;
        Ok((
            table_from_columns_like(&object, names, values, row_names, Some(&keep_rows))?,
            LogicalArray::new(row_has_missing, vec![height, 1]).map_err(internal_error)?,
        ))
    }
}

pub(super) fn remove_missing_tensor(
    tensor: Tensor,
    options: RemoveOptions,
) -> BuiltinResult<(Value, LogicalArray)> {
    if tensor.integer_storage().is_some() {
        let rows = tensor.rows();
        let cols = tensor.cols();
        let mask = if rows == 1 || cols == 1 {
            let len = tensor_utils::tensor_element_len(&tensor);
            LogicalArray::new(vec![0; len], vec![1, len])
        } else if matches!(options.dim, RemoveDim::Columns) {
            LogicalArray::new(vec![0; cols], vec![1, cols])
        } else {
            LogicalArray::new(vec![0; rows], vec![rows, 1])
        }
        .map_err(internal_error)?;
        return Ok((Value::Tensor(tensor), mask));
    }
    if tensor.rows() == 1 || tensor.cols() == 1 {
        let dtype = tensor.numeric_dtype();
        let source_rows = tensor.rows();
        let values = tensor_utils::tensor_into_values_f64(tensor);
        let mut data = Vec::new();
        let mut removed = Vec::with_capacity(values.len());
        let source_is_row = source_rows == 1;
        for value in values {
            if value.is_nan() {
                removed.push(1);
            } else {
                removed.push(0);
                data.push(value);
            }
        }
        let shape = if source_is_row {
            vec![1, data.len()]
        } else {
            vec![data.len(), 1]
        };
        let removed_len = removed.len();
        return Ok((
            Value::Tensor(Tensor::new_with_dtype(data, shape, dtype).map_err(internal_error)?),
            LogicalArray::new(removed, vec![1, removed_len]).map_err(internal_error)?,
        ));
    }
    let remove_columns = matches!(options.dim, RemoveDim::Columns);
    if remove_columns {
        remove_missing_columns_tensor(tensor)
    } else {
        remove_missing_rows_tensor(tensor)
    }
}

pub(super) fn remove_missing_rows_tensor(tensor: Tensor) -> BuiltinResult<(Value, LogicalArray)> {
    let rows = tensor.rows();
    let cols = tensor.cols();
    let mut removed = vec![0u8; rows];
    for col in 0..cols {
        for (row, slot) in removed.iter_mut().enumerate().take(rows) {
            if tensor.get2(row, col).map_err(internal_error)?.is_nan() {
                *slot = 1;
            }
        }
    }
    let keep: Vec<usize> = removed
        .iter()
        .enumerate()
        .filter_map(|(idx, flag)| (*flag == 0).then_some(idx))
        .collect();
    let out = select_rows(&Value::Tensor(tensor), &keep)?;
    Ok((
        out,
        LogicalArray::new(removed, vec![rows, 1]).map_err(internal_error)?,
    ))
}

pub(super) fn remove_missing_columns_tensor(
    tensor: Tensor,
) -> BuiltinResult<(Value, LogicalArray)> {
    let rows = tensor.rows();
    let cols = tensor.cols();
    let mut removed = vec![0u8; cols];
    for (col, slot) in removed.iter_mut().enumerate().take(cols) {
        for row in 0..rows {
            if tensor.get2(row, col).map_err(internal_error)?.is_nan() {
                *slot = 1;
            }
        }
    }
    let keep_cols: Vec<usize> = removed
        .iter()
        .enumerate()
        .filter_map(|(idx, flag)| (*flag == 0).then_some(idx))
        .collect();
    let mut data = Vec::with_capacity(rows * keep_cols.len());
    for col in keep_cols {
        for row in 0..rows {
            data.push(tensor.get2(row, col).map_err(internal_error)?);
        }
    }
    Ok((
        Value::Tensor(
            Tensor::new_with_dtype(
                data,
                vec![rows, removed.iter().filter(|f| **f == 0).count()],
                tensor.numeric_dtype(),
            )
            .map_err(internal_error)?,
        ),
        LogicalArray::new(removed, vec![1, cols]).map_err(internal_error)?,
    ))
}

pub(super) fn remove_missing_string_array(
    array: StringArray,
    options: RemoveOptions,
) -> BuiltinResult<(Value, LogicalArray)> {
    let rows = array.rows();
    let cols = array.cols();
    let shape = array.shape.clone();
    remove_missing_column_major(
        array.data,
        rows,
        cols,
        shape,
        options,
        |text| is_missing_text(text),
        |data, shape| {
            StringArray::new(data, shape)
                .map(Value::StringArray)
                .map_err(internal_error)
        },
    )
}

pub(super) fn remove_missing_logical_array(
    array: LogicalArray,
    options: RemoveOptions,
) -> BuiltinResult<(Value, LogicalArray)> {
    let rows = array.shape.first().copied().unwrap_or(array.data.len());
    let cols = array.shape.get(1).copied().unwrap_or(1);
    remove_missing_column_major(
        array.data.into_vec(),
        rows,
        cols,
        array.shape.clone(),
        options,
        |_| false,
        |data, shape| {
            LogicalArray::new(data, shape)
                .map(Value::LogicalArray)
                .map_err(internal_error)
        },
    )
}

pub(super) fn remove_missing_cell(
    cell: CellArray,
    options: RemoveOptions,
) -> BuiltinResult<(Value, LogicalArray)> {
    let rows = cell.rows;
    let cols = cell.cols;
    let is_missing = |row: usize, col: usize| -> bool {
        cell.get(row, col)
            .ok()
            .and_then(|value| any_missing(&value).ok())
            .unwrap_or(false)
    };

    if rows == 1 || cols == 1 {
        let mut out = Vec::new();
        let mut removed = Vec::with_capacity(cell.data.len());
        for value in cell.data {
            if any_missing(&value).unwrap_or(false) {
                removed.push(1);
            } else {
                removed.push(0);
                out.push(value);
            }
        }
        let out_shape = if rows == 1 {
            vec![1, out.len()]
        } else {
            vec![out.len(), 1]
        };
        let removed_len = removed.len();
        let out_rows = out_shape.first().copied().unwrap_or(0);
        let out_cols = out_shape.get(1).copied().unwrap_or(0);
        return Ok((
            CellArray::new(out, out_rows, out_cols)
                .map(Value::Cell)
                .map_err(internal_error)?,
            LogicalArray::new(removed, vec![1, removed_len]).map_err(internal_error)?,
        ));
    }

    if matches!(options.dim, RemoveDim::Columns) {
        let mut removed = vec![0u8; cols];
        for col in 0..cols {
            for row in 0..rows {
                if is_missing(row, col) {
                    removed[col] = 1;
                }
            }
        }
        let kept_cols: Vec<usize> = removed
            .iter()
            .enumerate()
            .filter_map(|(idx, flag)| (*flag == 0).then_some(idx))
            .collect();
        let mut out = Vec::with_capacity(rows * kept_cols.len());
        for row in 0..rows {
            for col in &kept_cols {
                out.push(cell.get(row, *col).map_err(internal_error)?);
            }
        }
        let kept = kept_cols.len();
        Ok((
            CellArray::new(out, rows, kept)
                .map(Value::Cell)
                .map_err(internal_error)?,
            LogicalArray::new(removed, vec![1, cols]).map_err(internal_error)?,
        ))
    } else {
        let mut removed = vec![0u8; rows];
        for row in 0..rows {
            for col in 0..cols {
                if is_missing(row, col) {
                    removed[row] = 1;
                }
            }
        }
        let kept_rows: Vec<usize> = removed
            .iter()
            .enumerate()
            .filter_map(|(idx, flag)| (*flag == 0).then_some(idx))
            .collect();
        let mut out = Vec::with_capacity(kept_rows.len() * cols);
        for row in &kept_rows {
            for col in 0..cols {
                out.push(cell.get(*row, col).map_err(internal_error)?);
            }
        }
        Ok((
            CellArray::new(out, kept_rows.len(), cols)
                .map(Value::Cell)
                .map_err(internal_error)?,
            LogicalArray::new(removed, vec![rows, 1]).map_err(internal_error)?,
        ))
    }
}

pub(super) fn remove_missing_column_major<T: Clone>(
    data: Vec<T>,
    rows: usize,
    cols: usize,
    _shape: Vec<usize>,
    options: RemoveOptions,
    is_missing: impl Fn(&T) -> bool,
    build: impl Fn(Vec<T>, Vec<usize>) -> BuiltinResult<Value>,
) -> BuiltinResult<(Value, LogicalArray)> {
    if rows == 1 || cols == 1 {
        let mut out = Vec::new();
        let mut removed = Vec::with_capacity(data.len());
        for value in data {
            if is_missing(&value) {
                removed.push(1);
            } else {
                removed.push(0);
                out.push(value);
            }
        }
        let out_shape = if rows == 1 {
            vec![1, out.len()]
        } else {
            vec![out.len(), 1]
        };
        let removed_len = removed.len();
        return Ok((
            build(out, out_shape)?,
            LogicalArray::new(removed, vec![1, removed_len]).map_err(internal_error)?,
        ));
    }
    if matches!(options.dim, RemoveDim::Columns) {
        let mut removed = vec![0u8; cols];
        for col in 0..cols {
            for row in 0..rows {
                if is_missing(&data[row + col * rows]) {
                    removed[col] = 1;
                }
            }
        }
        let kept_cols: Vec<usize> = removed
            .iter()
            .enumerate()
            .filter_map(|(idx, flag)| (*flag == 0).then_some(idx))
            .collect();
        let mut out = Vec::with_capacity(rows * kept_cols.len());
        for col in kept_cols {
            for row in 0..rows {
                out.push(data[row + col * rows].clone());
            }
        }
        let kept = removed.iter().filter(|flag| **flag == 0).count();
        Ok((
            build(out, vec![rows, kept])?,
            LogicalArray::new(removed, vec![1, cols]).map_err(internal_error)?,
        ))
    } else {
        let mut removed = vec![0u8; rows];
        for col in 0..cols {
            for row in 0..rows {
                if is_missing(&data[row + col * rows]) {
                    removed[row] = 1;
                }
            }
        }
        let kept_rows: Vec<usize> = removed
            .iter()
            .enumerate()
            .filter_map(|(idx, flag)| (*flag == 0).then_some(idx))
            .collect();
        let mut out = Vec::with_capacity(kept_rows.len() * cols);
        for col in 0..cols {
            for row in &kept_rows {
                out.push(data[*row + col * rows].clone());
            }
        }
        Ok((
            build(out, vec![kept_rows.len(), cols])?,
            LogicalArray::new(removed, vec![rows, 1]).map_err(internal_error)?,
        ))
    }
}

pub(super) fn empty_like(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => {
            Tensor::new_with_dtype(Vec::new(), vec![0, 0], tensor.numeric_dtype())
                .map(Value::Tensor)
                .map_err(internal_error)
        }
        Value::StringArray(_) | Value::String(_) => StringArray::new(Vec::new(), vec![0, 0])
            .map(Value::StringArray)
            .map_err(internal_error),
        Value::Cell(_) => CellArray::new(Vec::new(), 0, 0)
            .map(Value::Cell)
            .map_err(internal_error),
        _ => Ok(Value::OutputList(Vec::new())),
    }
}
