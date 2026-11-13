use super::*;

#[path = "table/overlay.rs"]
mod overlay;
#[path = "table/input.rs"]
mod input;
#[path = "table/existing.rs"]
mod existing;

#[derive(Debug, Clone)]
pub(super) struct XlsTable {
    rows: usize,
    cols: usize,
    cells: Vec<CellValue>,
}

impl XlsTable {
    pub(super) async fn preflight_host_known(
        value: &Value,
        cooperation: &mut TabularCooperation<'_>,
    ) -> BuiltinResult<()> {
        match value {
            Value::GpuTensor(_) => Ok(()),
            Value::Cell(cell) => {
                ensure_shape_2d(&cell.shape, "cell array")?;
                checked_cell_count(cell.rows, cell.cols, "cell array")?;
                for (index, member) in cell.data().iter().enumerate() {
                    if !matches!(member, Value::GpuTensor(_)) {
                        admit_cell_value(member)?;
                    }
                    if (index + 1).is_multiple_of(1024) {
                        cooperation.completed(1024).await?;
                    }
                }
                cooperation.completed(cell.data().len() % 1024).await
            }
            Value::Tensor(tensor) => {
                ensure_shape_2d(tensor.shape(), "numeric array")?;
                checked_cell_count(tensor.rows(), tensor.cols(), "numeric array")?;
                Ok(())
            }
            Value::SparseTensor(sparse) => {
                checked_cell_count(sparse.rows, sparse.cols, "sparse array")?;
                Ok(())
            }
            Value::LogicalArray(logical) => {
                ensure_shape_2d(&logical.shape, "logical array")?;
                let (rows, cols) = shape_rows_cols(&logical.shape);
                checked_cell_count(rows, cols, "logical array")?;
                Ok(())
            }
            Value::StringArray(strings) => {
                ensure_shape_2d(&strings.shape, "string array")?;
                checked_cell_count(strings.rows(), strings.cols(), "string array")?;
                Ok(())
            }
            Value::CharArray(chars) => {
                if chars.rows > 1 && chars.cols > 0 {
                    checked_cell_count(chars.rows, 1, "character array")?;
                }
                Ok(())
            }
            Value::String(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(()),
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

    fn single(value: CellValue) -> Self {
        Self {
            rows: 1,
            cols: 1,
            cells: vec![value],
        }
    }

    pub(super) fn empty() -> Self {
        Self {
            rows: 0,
            cols: 0,
            cells: Vec::new(),
        }
    }

    pub(super) fn into_cell_table(self) -> BuiltinResult<CellTable> {
        CellTable::from_cells(self.rows, self.cols, self.cells).map_err(map_writecell_error)
    }
}

async fn checkpoint_input(
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
    work: usize,
) -> BuiltinResult<()> {
    cooperation.completed(work).await?;
    selected.check().map_err(|source| {
        xlswrite_error_with_source(
            &XLSWRITE_ERROR_IO,
            "xlswrite: captured filesystem authority was revoked",
            source,
        )
    })
}

fn admit_cell_value(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::String(runmat_value::StringElement::Text(_)) => Ok(()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(()),
        Value::StringArray(strings) if strings.data().len() <= 1 => Ok(()),
        Value::Tensor(tensor)
            if tensor::is_scalar_tensor(tensor) || tensor::tensor_element_len(tensor) == 0 =>
        {
            Ok(())
        }
        Value::LogicalArray(logical) if logical.data.len() <= 1 => Ok(()),
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

fn ensure_shape_2d(shape: &[usize], context: &str) -> BuiltinResult<()> {
    if shape.len() <= 2 || shape[2..].iter().all(|&dim| dim == 1) {
        Ok(())
    } else {
        Err(xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            format!("xlswrite: {context} must be 2-D"),
        ))
    }
}

fn shape_rows_cols(shape: &[usize]) -> (usize, usize) {
    match shape {
        [] => (1, 1),
        [cols] => (1, *cols),
        [rows, cols, ..] => (*rows, *cols),
    }
}

fn checked_cell_count(rows: usize, cols: usize, context: &str) -> BuiltinResult<usize> {
    if rows > MAX_EXCEL_ROW_INDEX + 1 || cols > MAX_EXCEL_COLUMN_INDEX + 1 {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            format!("xlswrite: {context} exceeds Excel worksheet limits"),
        ));
    }
    let cells = rows.checked_mul(cols).ok_or_else(|| {
        xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            format!("xlswrite: {context} cell count overflows supported bounds"),
        )
    })?;
    if cells > MAX_XLSWRITE_CELLS {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_DATA,
            format!(
                "xlswrite: {context} exceeds maximum supported size of {MAX_XLSWRITE_CELLS} cells"
            ),
        ));
    }
    Ok(cells)
}

pub(super) fn validate_write_bounds(table: &XlsTable, start: RangeStart) -> BuiltinResult<()> {
    let end_row = start.row.checked_add(table.rows).ok_or_else(|| {
        xlswrite_error_with(&XLSWRITE_ERROR_RANGE, "xlswrite: range row overflow")
    })?;
    let end_col = start.col.checked_add(table.cols).ok_or_else(|| {
        xlswrite_error_with(&XLSWRITE_ERROR_RANGE, "xlswrite: range column overflow")
    })?;
    if end_row > MAX_EXCEL_ROW_INDEX + 1 || end_col > MAX_EXCEL_COLUMN_INDEX + 1 {
        return Err(xlswrite_error_with(
            &XLSWRITE_ERROR_RANGE,
            "xlswrite: range exceeds Excel worksheet limits",
        ));
    }
    Ok(())
}
