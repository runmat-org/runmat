//! Owned value-to-worksheet projection with one invocation cooperation owner.

use super::*;

impl XlsTable {
    pub(in crate::builtins::io::tabular::xlswrite) async fn from_value_cooperative(
        value: Value,
        runtime: Option<&crate::context::RuntimeContext>,
        cooperation: &mut TabularCooperation<'_>,
        selected: &TabularFsPath,
    ) -> BuiltinResult<Self> {
        match value {
            Value::Cell(cell) => Self::from_cell_array(cell, runtime, cooperation, selected).await,
            Value::Tensor(tensor) => Self::from_tensor(tensor, cooperation, selected).await,
            Value::SparseTensor(sparse) => Self::from_sparse(sparse, cooperation, selected).await,
            Value::LogicalArray(logical) => {
                Self::from_logical(logical, cooperation, selected).await
            }
            Value::StringArray(strings) => {
                Self::from_string_array(strings, cooperation, selected).await
            }
            Value::String(runmat_value::StringElement::Text(text)) => {
                Ok(Self::single(CellValue::Text(text.into_string_unbounded())))
            }
            Value::CharArray(chars) => Self::from_char_array(chars, cooperation, selected).await,
            Value::Num(value) => Ok(Self::single(CellValue::Number(value))),
            Value::Int(value) => Ok(Self::single(CellValue::Integer(value))),
            Value::Bool(value) => Ok(Self::single(CellValue::Boolean(value))),
            Value::Complex(_, _) | Value::ComplexTensor(_) => Err(xlswrite_error_with(
                &XLSWRITE_ERROR_DATA,
                "xlswrite: complex values are not supported; split real and imaginary parts first",
            )),
            other => Err(xlswrite_error_with(
                &XLSWRITE_ERROR_DATA,
                format!("xlswrite: unsupported data value {other:?}"),
            )),
        }
    }

    async fn from_cell_array(
        cell: CellArray,
        runtime: Option<&crate::context::RuntimeContext>,
        cooperation: &mut TabularCooperation<'_>,
        selected: &TabularFsPath,
    ) -> BuiltinResult<Self> {
        ensure_shape_2d(&cell.shape, "cell array")?;
        checked_cell_count(cell.rows, cell.cols, "cell array")?;
        let mut cells = reserved_cells(cell.data().len())?;
        let mut work = 0;
        for value in cell.data().iter() {
            let materialized = if matches!(value, Value::GpuTensor(_)) {
                checkpoint_input(cooperation, selected, work).await?;
                work = 0;
                Some(
                    await_io(runtime, BUILTIN_NAME, gather_if_needed_async(value))
                        .await?
                        .map_err(map_control_flow)?,
                )
            } else {
                None
            };
            let scalar = materialized.as_ref().unwrap_or(value);
            cells.push(cell_value_from_scalar(scalar, cooperation, selected).await?);
            work += 1;
            if work == 1024 {
                checkpoint_input(cooperation, selected, work).await?;
                work = 0;
            }
        }
        checkpoint_input(cooperation, selected, work).await?;
        Ok(Self {
            rows: cell.rows,
            cols: cell.cols,
            cells,
        })
    }

    async fn from_tensor(
        tensor: Tensor,
        cooperation: &mut TabularCooperation<'_>,
        selected: &TabularFsPath,
    ) -> BuiltinResult<Self> {
        ensure_shape_2d(tensor.shape(), "numeric array")?;
        let rows = tensor.rows();
        let cols = tensor.cols();
        let count = checked_cell_count(rows, cols, "numeric array")?;
        if count == 0 {
            checkpoint_input(cooperation, selected, 0).await?;
            return Ok(Self {
                rows,
                cols,
                cells: Vec::new(),
            });
        }
        let mut cells = reserved_cells(count)?;
        let mut work = 0;
        for row in 0..rows {
            for col in 0..cols {
                let value = tensor
                    .numeric_value_at(row + col * rows)
                    .expect("index within authoritative numeric storage");
                if let Some(value) = value.into_int_value() {
                    cells.push(CellValue::Integer(value));
                } else {
                    cells.push(CellValue::Number(value.materialize_f64()));
                }
                work += 1;
                if work == 1024 {
                    checkpoint_input(cooperation, selected, work).await?;
                    work = 0;
                }
            }
        }
        checkpoint_input(cooperation, selected, work).await?;
        Ok(Self { rows, cols, cells })
    }

    async fn from_sparse(
        sparse: SparseTensor,
        cooperation: &mut TabularCooperation<'_>,
        selected: &TabularFsPath,
    ) -> BuiltinResult<Self> {
        let rows = sparse.rows;
        let cols = sparse.cols;
        let count = checked_cell_count(rows, cols, "sparse array")?;
        if count == 0 {
            checkpoint_input(cooperation, selected, 0).await?;
            return Ok(Self {
                rows,
                cols,
                cells: Vec::new(),
            });
        }
        let mut cells = reserved_cells(count)?;
        while cells.len() < count {
            let chunk = (count - cells.len()).min(1024);
            let default = if sparse.is_logical() {
                CellValue::Boolean(false)
            } else {
                CellValue::Number(0.0)
            };
            cells.resize(cells.len() + chunk, default);
            checkpoint_input(cooperation, selected, chunk).await?;
        }
        let mut work = 0;
        for col in 0..cols {
            work += 1;
            if work >= 1024 {
                checkpoint_input(cooperation, selected, work).await?;
                work = 0;
            }
            let start = sparse.col_ptrs[col];
            let end = sparse.col_ptrs[col + 1];
            for entry in start..end {
                let row = sparse.row_indices[entry];
                cells[row * cols + col] = if sparse.is_logical() {
                    CellValue::Boolean(true)
                } else if let Some(storage) = sparse.integer_storage() {
                    CellValue::Integer(storage.value_at(entry).ok_or_else(|| {
                        xlswrite_error_with(
                            &XLSWRITE_ERROR_DATA,
                            format!("xlswrite: sparse integer value buffer missing entry {entry}"),
                        )
                    })?)
                } else {
                    CellValue::Number(
                        sparse
                            .numeric_value_at(entry)
                            .expect("sparse storage index is valid")
                            .materialize_f64(),
                    )
                };
                work += 1;
                if work == 1024 {
                    checkpoint_input(cooperation, selected, work).await?;
                    work = 0;
                }
            }
        }
        checkpoint_input(cooperation, selected, work).await?;
        Ok(Self { rows, cols, cells })
    }

    async fn from_logical(
        logical: LogicalArray,
        cooperation: &mut TabularCooperation<'_>,
        selected: &TabularFsPath,
    ) -> BuiltinResult<Self> {
        ensure_shape_2d(&logical.shape, "logical array")?;
        let (rows, cols) = shape_rows_cols(&logical.shape);
        let count = checked_cell_count(rows, cols, "logical array")?;
        if count == 0 {
            checkpoint_input(cooperation, selected, 0).await?;
            return Ok(Self {
                rows,
                cols,
                cells: Vec::new(),
            });
        }
        let mut cells = reserved_cells(count)?;
        let mut work = 0;
        for row in 0..rows {
            for col in 0..cols {
                cells.push(CellValue::Boolean(logical.data[row + col * rows] != 0));
                work += 1;
                if work == 1024 {
                    checkpoint_input(cooperation, selected, work).await?;
                    work = 0;
                }
            }
        }
        checkpoint_input(cooperation, selected, work).await?;
        Ok(Self { rows, cols, cells })
    }

    async fn from_string_array(
        strings: StringArray,
        cooperation: &mut TabularCooperation<'_>,
        selected: &TabularFsPath,
    ) -> BuiltinResult<Self> {
        if strings.data().iter().any(|text| text.is_missing()) {
            return Err(xlswrite_error_with(
                &XLSWRITE_ERROR_DATA,
                "xlswrite: typed missing strings cannot be represented as worksheet text",
            ));
        }
        ensure_shape_2d(&strings.shape, "string array")?;
        let rows = strings.rows();
        let cols = strings.cols();
        let count = checked_cell_count(rows, cols, "string array")?;
        let mut cells = reserved_cells(count)?;
        while cells.len() < count {
            let chunk = (count - cells.len()).min(1024);
            cells.resize(cells.len() + chunk, CellValue::Empty);
            checkpoint_input(cooperation, selected, chunk).await?;
        }
        let mut work = 0;
        for (index, text) in strings.data().iter().enumerate() {
            let row = index % rows;
            let col = index / rows;
            cells[row * cols + col] =
                CellValue::Text(text.as_str().expect("checked nonmissing string").to_owned());
            work += 1;
            if work == 1024 {
                checkpoint_input(cooperation, selected, work).await?;
                work = 0;
            }
        }
        checkpoint_input(cooperation, selected, work).await?;
        Ok(Self { rows, cols, cells })
    }

    async fn from_char_array(
        chars: CharArray,
        cooperation: &mut TabularCooperation<'_>,
        selected: &TabularFsPath,
    ) -> BuiltinResult<Self> {
        if chars.rows == 0 || chars.cols == 0 {
            return Ok(Self::empty());
        }
        if chars.rows > 1 {
            checked_cell_count(chars.rows, 1, "character array")?;
        }
        let rows = if chars.rows == 1 { 1 } else { chars.rows };
        let mut cells = reserved_cells(rows)?;
        let character_count = chars.data().len();
        let mut work = 0;
        for row in 0..rows {
            let used_columns = if rows == 1 {
                character_count
            } else {
                chars.cols
            };
            let start = if rows == 1 { 0 } else { row * chars.cols };
            let text = crate::builtins::io::utf16::decode_utf16_with_error(
                &chars.data()[start..start + used_columns],
                || {
                    xlswrite_error_with(
                        &XLSWRITE_ERROR_DATA,
                        "xlswrite: character input contains malformed UTF-16 text",
                    )
                },
            )?;
            for _ in 0..used_columns {
                work += 1;
                if work == 1024 {
                    checkpoint_input(cooperation, selected, work).await?;
                    work = 0;
                }
            }
            cells.push(CellValue::Text(text));
        }
        checkpoint_input(cooperation, selected, work).await?;
        Ok(Self {
            rows,
            cols: 1,
            cells,
        })
    }
}

fn reserved_cells(count: usize) -> BuiltinResult<Vec<CellValue>> {
    let mut cells = Vec::new();
    cells.try_reserve_exact(count).map_err(|source| {
        xlswrite_error_with_source(
            &XLSWRITE_ERROR_DATA,
            "xlswrite: cell table allocation failed",
            source,
        )
    })?;
    Ok(cells)
}

#[cfg(test)]
#[path = "input/tests.rs"]
mod tests;

async fn cell_value_from_scalar(
    value: &Value,
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
) -> BuiltinResult<CellValue> {
    admit_cell_value(value)?;
    match value {
        Value::Num(n) => Ok(CellValue::Number(*n)),
        Value::Int(i) => Ok(CellValue::Integer(i.clone())),
        Value::Bool(b) => Ok(CellValue::Boolean(*b)),
        Value::String(runmat_value::StringElement::Text(s)) => {
            Ok(CellValue::Text(s.as_str().to_owned()))
        }
        Value::CharArray(chars) if chars.rows == 1 => {
            let text = crate::builtins::io::utf16::decode_utf16_with_error(chars.data(), || {
                xlswrite_error_with(
                    &XLSWRITE_ERROR_DATA,
                    "xlswrite: character cell contains malformed UTF-16 text",
                )
            })?;
            let mut work = 0;
            for _ in chars.data() {
                work += 1;
                if work == 1024 {
                    checkpoint_input(cooperation, selected, work).await?;
                    work = 0;
                }
            }
            checkpoint_input(cooperation, selected, work).await?;
            Ok(CellValue::Text(text))
        }
        Value::StringArray(strings) if strings.data().is_empty() => Ok(CellValue::Empty),
        Value::StringArray(strings) if !strings.data()[0].is_missing() => Ok(CellValue::Text(
            strings.data()[0]
                .as_str()
                .expect("checked nonmissing string")
                .to_owned(),
        )),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            writecell::scalar_tensor_cell_value(tensor)
        }
        Value::Tensor(tensor) if tensor::tensor_element_len(tensor) == 0 => Ok(CellValue::Empty),
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(CellValue::Boolean(logical.data[0] != 0))
        }
        Value::LogicalArray(logical) if logical.data.is_empty() => Ok(CellValue::Empty),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            "xlswrite: complex cell values are not supported",
        )),
        Value::Cell(_) => Err(xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            "xlswrite: nested cell arrays are not supported",
        )),
        other => Err(xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            format!("xlswrite: unsupported cell value {other:?}"),
        )),
    }
}
